"""Smoke test for the importable pipeline (``kannada_epub.pipeline``).

Everything runs against FAKE translation/editor/QA components and the real
Sherlock Holmes EPUB: no models, no network. It also proves the package imports
and builds a cloud provider with the optional local ML stack blocked.

Run: .venv/bin/python scripts/test_pipeline.py
"""

import io
import json
import math
import os
import posixpath
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import zipfile
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lxml.etree as ET
from bs4 import BeautifulSoup
from epub_fixture import build_epub

from kannada_epub import pipeline as pipeline_module
from kannada_epub.book_translator import (
    BookTranslator,
    TranslatedBatch,
    chapter_context_tail,
    detect_chapter_context,
)
from kannada_epub.config import BookConfig
from kannada_epub.consistency_editor import (
    ConsistencyEditor,
    EditorOutputError,
    _parse_numbered_output,
)
from kannada_epub.epub_io import (
    BLOCK_TAGS,
    Chapter,
    Paragraph,
    load_epub_chapters,
)
from kannada_epub.epub_writer import find_gutenberg_cover
from kannada_epub.epubcheck_runner import EpubcheckResult
from kannada_epub.glossary import GlossaryStore
from kannada_epub.pipeline import PipelineComponents, RunOptions, run_book
from kannada_epub.providers.base import OutputTruncatedError
from kannada_epub.qa import FLAGGED_FOR_REVIEW, PASS, RETRY
from kannada_epub.translation.base import TranslationProvider

ROOT = Path(__file__).resolve().parent.parent
EPUB = ROOT / "data" / "sherlock_holmes.epub"
CHAPTER_ID = "item4"
N_PARAGRAPHS = 6
BATCH_SIZE = 3

_OPF_NS = "http://www.idpf.org/2007/opf"
_CONTAINER_NS = "urn:oasis:names:tc:opendocument:xmlns:container"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeTranslationEngine(TranslationProvider):
    """Returns deterministic "Kannada" text and counts every translate call."""

    def __init__(self, prefix: str = "ಕನ್ನಡ"):
        self._prefix = prefix
        self.calls = 0
        self.batch_sizes: list[int] = []

    def translate_paragraphs(
        self, paragraphs: list[str], src_lang: str, tgt_lang: str
    ) -> list[str]:
        self.calls += 1
        self.batch_sizes.append(len(paragraphs))
        return [f"{self._prefix} {text[:10]}" for text in paragraphs]


class RecordingCoverEngine(FakeTranslationEngine):
    """Records every request; can return a wrong count for the cover text."""

    def __init__(self, *, mismatch: bool = False):
        super().__init__()
        self.inputs: list[list[str]] = []
        self.mismatch = mismatch

    def translate_paragraphs(
        self, paragraphs: list[str], src_lang: str, tgt_lang: str
    ) -> list[str]:
        self.inputs.append(list(paragraphs))
        if self.mismatch and paragraphs and paragraphs[0] == "Cover Test":
            self.calls += 1
            self.batch_sizes.append(len(paragraphs))
            return ["ಒಂದೇ"]  # deliberately the wrong count
        return super().translate_paragraphs(paragraphs, src_lang, tgt_lang)


class FakeEditorProvider:
    """Returns the draft paragraphs unchanged, in the editor's numbered format."""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        draft = user_prompt.split("DRAFT_CHAPTER:\n", 1)[1]
        parts = re.split(r"\[P(\d+)\]\n", draft)
        blocks: list[str] = []
        for i in range(1, len(parts), 2):
            number, text = parts[i], parts[i + 1].strip()
            blocks.append(f"[P{number}]\nEMOTION: Narration\n{text}")
        return "\n\n".join(blocks)


class RecordingEditorProvider:
    """Wraps another editor provider and records every prompt it is given."""

    def __init__(self, inner):
        self._inner = inner
        self.system_prompts: list[str] = []
        self.user_prompts: list[str] = []

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.system_prompts.append(system_prompt)
        self.user_prompts.append(user_prompt)
        return self._inner.complete(system_prompt, user_prompt)


def _headings_in_prompt(prompt: str) -> list[str]:
    """The ``P<n>`` tags listed in a prompt's HEADINGS block ([] if absent)."""
    match = re.search(r"HEADINGS:\n([^\n]*)", prompt)
    if not match:
        return []
    return [tag.strip() for tag in match.group(1).split(",") if tag.strip()]


class _EchoEditorProviderBase:
    """Parses the numbered draft and re-emits it, appending "[EDITED]" markers.

    Subclasses decide which paragraph counts to fail on, so the test can drive
    BookTranslator's split-and-retry either through truncation
    (`OutputTruncatedError`) or through a paragraph-count mismatch.
    """

    def __init__(self):
        self.batch_sizes: list[int] = []

    def _blocks(self, user_prompt: str) -> list[tuple[str, str]]:
        draft = user_prompt.split("DRAFT_CHAPTER:\n", 1)[1]
        parts = re.split(r"\[P(\d+)\]\n", draft)
        return [(parts[i], parts[i + 1].strip()) for i in range(1, len(parts), 2)]

    def _emit(self, blocks: list[tuple[str, str]]) -> str:
        return "\n\n".join(
            f"[P{number}]\nEMOTION: Narration\n{text} [EDITED]" for number, text in blocks
        )


class TruncatingEditorProvider(_EchoEditorProviderBase):
    """Raises OutputTruncatedError whenever asked to edit more than `max_ok`."""

    def __init__(self, max_ok: int):
        super().__init__()
        self._max_ok = max_ok

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        blocks = self._blocks(user_prompt)
        self.batch_sizes.append(len(blocks))
        if len(blocks) > self._max_ok:
            raise OutputTruncatedError(f"{len(blocks)} paragraphs is more than {self._max_ok}")
        return self._emit(blocks)


class DropLastEditorProvider(_EchoEditorProviderBase):
    """Drops the final paragraph when asked to edit more than `max_ok`.

    The missing number makes `_parse_numbered_output` raise an EditorOutputError
    (a RuntimeError), exercising the mismatch path rather than truncation.
    """

    def __init__(self, max_ok: int):
        super().__init__()
        self._max_ok = max_ok

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        blocks = self._blocks(user_prompt)
        self.batch_sizes.append(len(blocks))
        if len(blocks) > self._max_ok:
            blocks = blocks[:-1]
        return self._emit(blocks)


class AlwaysFailEditorProvider:
    """Fails with a retryable error even for a single paragraph."""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        raise OutputTruncatedError("every batch is too large")


class ExplodingTranslationEngine(TranslationProvider):
    """Raises if translated, proving a chapter was reused from checkpoints."""

    def translate_paragraphs(
        self, paragraphs: list[str], src_lang: str, tgt_lang: str
    ) -> list[str]:
        raise AssertionError("translation ran despite resumable checkpoints")


class BlockChapterOneEngine(FakeTranslationEngine):
    """Fails only for chapter 1, so a later chapter still translates on resume."""

    def translate_paragraphs(
        self, paragraphs: list[str], src_lang: str, tgt_lang: str
    ) -> list[str]:
        if paragraphs and paragraphs[0] == "CHAPTER I":
            raise AssertionError("chapter 1 was retranslated despite checkpoints")
        return super().translate_paragraphs(paragraphs, src_lang, tgt_lang)


class FakeBackTranslator:
    """Scores paragraph 0 RETRY (improving on retry) and paragraph 1 FLAGGED.

    The vector quality is encoded in the returned marker ("q:<cos>") and read
    by FakeEmbedder, so no real model is involved.
    """

    def __init__(self):
        self.n_calls = 0

    def back_translate(self, kannada: list[str]) -> list[str]:
        self.n_calls += 1
        if self.n_calls == 1:
            output: list[str] = []
            for i in range(len(kannada)):
                if i == 0:
                    output.append("q:0.75")  # RETRY (0.70 <= 0.75 < 0.85)
                elif i == 1:
                    output.append("q:0.5")  # FLAGGED_FOR_REVIEW
                else:
                    output.append("q:1.0")  # PASS
            return output
        return ["q:0.95" for _ in kannada]  # improved retry -> PASS


class FakeEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            if text.startswith("q:"):
                x = float(text[2:])
                vectors.append([x, math.sqrt(max(0.0, 1.0 - x * x))])
            else:
                vectors.append([1.0, 0.0])
        return vectors


class RetryBackTranslator:
    """Two paragraphs score RETRY; on retry one improves, one gets worse.

    The first call (the whole chapter, three paragraphs) scores p0/p1 RETRY
    and p2 PASS. The second call (the two retried paragraphs) scores p0 higher
    and p1 lower, so the pipeline must keep p0's retry and discard p1's.
    """

    def __init__(self):
        self.n_calls = 0

    def back_translate(self, kannada: list[str]) -> list[str]:
        self.n_calls += 1
        if self.n_calls == 1:
            return ["q:0.75", "q:0.80", "q:1.0"]
        return ["q:0.95", "q:0.5"]


class RetryRecordingEngine(FakeTranslationEngine):
    """Records QA retries and returns text visibly different from the draft."""

    retry_description = "test variation"

    def __init__(self):
        super().__init__()
        self.retry_calls = 0
        self.retry_batch_sizes: list[int] = []

    def translate_paragraphs_retry(
        self, paragraphs: list[str], src_lang: str, tgt_lang: str
    ) -> list[str]:
        self.retry_calls += 1
        self.retry_batch_sizes.append(len(paragraphs))
        return [f"RETRY {text}" for text in paragraphs]


class MismatchedRetryEngine(FakeTranslationEngine):
    """A retry that returns the wrong number of strings."""

    def translate_paragraphs_retry(
        self, paragraphs: list[str], src_lang: str, tgt_lang: str
    ) -> list[str]:
        return ["only one"]


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else ROOT / p


def _make_components(tmp: Path) -> tuple[PipelineComponents, FakeTranslationEngine]:
    engine = FakeTranslationEngine()
    components = PipelineComponents(
        translation_engine=engine,
        editor=ConsistencyEditor(FakeEditorProvider()),
        glossary_store=GlossaryStore(tmp / "glossary.db"),
        qa=(FakeBackTranslator(), FakeEmbedder()),
    )
    return components, engine


def _make_cfg(output_dir: Path) -> BookConfig:
    return BookConfig(
        epub_path="data/sherlock_holmes.epub",
        output_dir=str(output_dir),
        glossary_db=str(output_dir / "glossary.db"),
    )


def _chapter_document(epub_path: Path, chapter_id: str) -> bytes:
    with zipfile.ZipFile(epub_path) as zf:
        container = ET.fromstring(zf.read("META-INF/container.xml"))
        rootfile = container.find(f".//{{{_CONTAINER_NS}}}rootfile")
        opf_path = rootfile.get("full-path")
        opf = ET.fromstring(zf.read(opf_path))
        href = next(
            item.get("href")
            for item in opf.findall(f".//{{{_OPF_NS}}}manifest/{{{_OPF_NS}}}item")
            if item.get("id") == chapter_id
        )
        doc_path = posixpath.normpath(
            posixpath.join(posixpath.dirname(opf_path), posixpath.normpath(href))
        )
        return zf.read(doc_path)


def _check_cloud_only_import() -> None:
    src = str(ROOT / "src")
    code = r'''
import sys
sys.path.insert(0, "__SRC__")

BLOCKED = {"torch", "transformers", "ctranslate2", "sentencepiece",
           "IndicTransToolkit", "parler_tts", "huggingface_hub"}


class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError("blocked local ML dependency: " + name)
        return None


sys.meta_path.insert(0, Blocker())

import kannada_epub.pipeline  # noqa: F401
import kannada_epub.translation  # noqa: F401
import kannada_epub.tts  # noqa: F401
import kannada_epub.qa  # noqa: F401
from kannada_epub.translation import IndicTrans2Engine, build_translation_provider
from kannada_epub.config import TranslationModelConfig

assert IndicTrans2Engine is not None

provider = build_translation_provider(
    TranslationModelConfig(
        provider="openai_compatible",
        base_url="http://example.invalid/v1",
        model="dummy-model",
        api_key_env="DUMMY_API_KEY",
    )
)
assert provider is not None

try:
    build_translation_provider(
        TranslationModelConfig(provider="indictrans2_local"),
        ct2_model_dir="/tmp/does-not-matter",
    )
except RuntimeError as exc:
    assert ".[local]" in str(exc), str(exc)
else:
    raise AssertionError("expected RuntimeError for indictrans2_local without local deps")

print("cloud-only-import-ok")
'''.replace("__SRC__", src)

    env = dict(os.environ)
    env["DUMMY_API_KEY"] = "dummy-key"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env=env,
    )
    assert proc.returncode == 0, f"subprocess failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert "cloud-only-import-ok" in proc.stdout


def _load_chapter(chapter_id: str, n: int):
    """The first ``n`` translatable paragraphs of a Sherlock chapter.

    Number-only paragraphs ("I.") bypass the engine and editor (#27), so the
    split-and-retry tests use only paragraphs that reach them.
    """
    from kannada_epub.book_translator import needs_translation

    chapter = next(c for c in load_epub_chapters(EPUB) if c.id == chapter_id)
    chapter.paragraphs = [p for p in chapter.paragraphs if needs_translation(p.text)][:n]
    return chapter


def _chapter(chapter_id: str, title: str | None, paragraphs: tuple[str, ...] = ()) -> Chapter:
    return Chapter(
        id=chapter_id,
        title=title,
        paragraphs=[Paragraph(i, text) for i, text in enumerate(paragraphs)],
    )


def _multi_chapter_epub(path: Path, titles: list[str], *, paragraphs: int = 5) -> None:
    """A small EPUB with one heading + `paragraphs` body paragraphs per chapter."""
    documents = []
    for index, title in enumerate(titles, start=1):
        body = "".join(
            f"<p>Chapter {index} body paragraph {n}.</p>" for n in range(1, paragraphs + 1)
        )
        documents.append({
            "id": f"ch{index}",
            "href": f"ch{index}.xhtml",
            "content": (
                '<?xml version="1.0" encoding="utf-8"?>'
                '<html xmlns="http://www.w3.org/1999/xhtml">'
                f"<head><title>{title}</title></head>"
                f"<body><h1>{title}</h1>{body}</body></html>"
            ),
        })
    build_epub(path, documents)


def _expected_edited(source_texts: list[str]) -> list[str]:
    return [f"ಕನ್ನಡ {text[:10]} [EDITED]" for text in source_texts]


def _check_split_and_resume(tmp: Path) -> None:
    """Editor split-and-retry (truncation + mismatch) and batch-size-free resume."""
    # --- truncating editor: one 10-paragraph batch splits down and lines up --
    chapter = _load_chapter(CHAPTER_ID, 10)
    source_texts = [p.text for p in chapter.paragraphs]
    engine = FakeTranslationEngine()
    truncating = TruncatingEditorProvider(max_ok=3)
    translator = BookTranslator(
        translation_engine=engine,
        glossary_store=GlossaryStore(tmp / "split.db"),
        consistency_editor=ConsistencyEditor(truncating),
        batch_size=10,
    )
    batches = translator.translate_chapters([chapter])
    assert len(batches) == 1, batches
    batch = batches[0]
    assert (batch.paragraph_start, batch.paragraph_end) == (0, 10), batch
    assert batch.prior_context_used == ""
    assert batch.source_english == source_texts
    assert batch.draft_kannada == [f"ಕನ್ನಡ {t[:10]}" for t in source_texts]
    assert batch.edited_kannada == _expected_edited(source_texts), batch.edited_kannada
    assert len(batch.edited_emotions) == 10
    # The editor really did split: it was asked for 10, then 5s, then 2s/3s.
    assert truncating.batch_sizes[0] == 10
    assert 5 in truncating.batch_sizes and 2 in truncating.batch_sizes
    assert engine.calls == 1, engine.calls  # draft translation not redone

    # --- dropping the last paragraph of big batches hits the mismatch path ---
    chapter2 = _load_chapter(CHAPTER_ID, 10)
    source_texts2 = [p.text for p in chapter2.paragraphs]
    engine2 = FakeTranslationEngine()
    dropping = DropLastEditorProvider(max_ok=2)
    translator2 = BookTranslator(
        translation_engine=engine2,
        glossary_store=GlossaryStore(tmp / "drop.db"),
        consistency_editor=ConsistencyEditor(dropping),
        batch_size=10,
    )
    batches2 = translator2.translate_chapters([chapter2])
    assert len(batches2) == 1, batches2
    assert (batches2[0].paragraph_start, batches2[0].paragraph_end) == (0, 10), batches2[0]
    assert batches2[0].edited_kannada == _expected_edited(source_texts2), batches2[0].edited_kannada
    assert engine2.calls == 1, engine2.calls

    # --- draft translation is one call per ORIGINAL batch, not per split ----
    chapter3 = _load_chapter(CHAPTER_ID, 10)
    source_texts3 = [p.text for p in chapter3.paragraphs]
    engine3 = FakeTranslationEngine()
    translator3 = BookTranslator(
        translation_engine=engine3,
        glossary_store=GlossaryStore(tmp / "multi.db"),
        consistency_editor=ConsistencyEditor(TruncatingEditorProvider(max_ok=3)),
        batch_size=4,
    )
    batches3 = translator3.translate_chapters([chapter3])
    assert engine3.batch_sizes == [4, 4, 2], engine3.batch_sizes
    assert engine3.calls == 3, engine3.calls
    assert [(b.paragraph_start, b.paragraph_end) for b in batches3] == [(0, 4), (4, 8), (8, 10)]
    flat = [text for b in batches3 for text in b.edited_kannada]
    assert flat == _expected_edited(source_texts3), flat

    # --- an always-failing editor propagates and leaves no partial state ----
    fail_dir = tmp / "always_fail"
    fail_components = PipelineComponents(
        translation_engine=FakeTranslationEngine(),
        editor=ConsistencyEditor(AlwaysFailEditorProvider()),
        glossary_store=GlossaryStore(fail_dir / "glossary.db"),
    )
    try:
        run_book(
            _make_cfg(fail_dir),
            resolve_path=_resolve,
            options=RunOptions(limit_chapters=[CHAPTER_ID], max_paragraphs=3, batch_size=3),
            components=fail_components,
            progress=lambda _msg: None,
        )
    except OutputTruncatedError as exc:
        assert "every batch is too large" in str(exc)
    else:
        raise AssertionError("expected OutputTruncatedError to propagate")
    checkpoints_dir = fail_dir / "checkpoints"
    assert not list(checkpoints_dir.iterdir()), list(checkpoints_dir.iterdir())
    assert not (fail_dir / "chapters" / f"{CHAPTER_ID}.json").exists()
    assert not (fail_dir / "manifest.json").exists()

    # --- resume works across a changed batch_size --------------------------
    resume_dir = tmp / "batch_size_resume"
    resume_components = PipelineComponents(
        translation_engine=FakeTranslationEngine(),
        editor=ConsistencyEditor(FakeEditorProvider()),
        glossary_store=GlossaryStore(resume_dir / "glossary.db"),
    )
    first = run_book(
        _make_cfg(resume_dir),
        resolve_path=_resolve,
        options=RunOptions(limit_chapters=[CHAPTER_ID], max_paragraphs=12, batch_size=4),
        components=resume_components,
        progress=lambda _msg: None,
    )
    assert first.epub_path is not None and first.epub_path.exists()
    saved = sorted(p.name for p in (resume_dir / "checkpoints").iterdir())
    assert saved == ["item4_0000.json", "item4_0004.json", "item4_0008.json"], saved

    exploding = ExplodingTranslationEngine()
    second_components = PipelineComponents(
        translation_engine=exploding,
        editor=ConsistencyEditor(FakeEditorProvider()),
        glossary_store=GlossaryStore(resume_dir / "glossary.db"),
    )
    second = run_book(
        _make_cfg(resume_dir),
        resolve_path=_resolve,
        options=RunOptions(limit_chapters=[CHAPTER_ID], max_paragraphs=12, batch_size=3),
        components=second_components,
        progress=lambda _msg: None,
    )
    assert second.epub_path is not None and second.epub_path.exists()
    manifest = json.loads((resume_dir / "manifest.json").read_text(encoding="utf-8"))
    assert CHAPTER_ID in manifest["skipped"], manifest["skipped"]


def _check_source_problems(tmp: Path) -> None:
    """A defective source EPUB is reported in the manifest but doesn't stop the run."""
    fixture = tmp / "defective.epub"
    chapter = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<html xmlns="http://www.w3.org/1999/xhtml">'
        "<head><title>One</title></head>"
        "<body><h1>One</h1><p>Hello there.</p></body></html>"
    )
    build_epub(fixture, [
        {"id": "ch1", "href": "ch1.xhtml", "content": chapter},
        {
            "id": "img",
            "href": "images/pic",
            "content": b"\x89PNG\r\n\x1a\n" + b"not a real png body",
            "media_type": "image/png",
        },
    ])
    out_dir = tmp / "defective_out"
    components = PipelineComponents(
        translation_engine=FakeTranslationEngine(),
        editor=ConsistencyEditor(FakeEditorProvider()),
        glossary_store=GlossaryStore(out_dir / "glossary.db"),
    )
    cfg = BookConfig(
        epub_path=str(fixture),
        output_dir=str(out_dir),
        glossary_db=str(out_dir / "glossary.db"),
    )
    lines: list[str] = []
    result = run_book(
        cfg,
        resolve_path=_resolve,
        options=RunOptions(limit_chapters=["ch1"], max_paragraphs=1, batch_size=1),
        components=components,
        progress=lines.append,
    )
    # The run still finished and wrote its EPUB.
    assert result.epub_path is not None and result.epub_path.exists()
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    problems = manifest["source_problems"]
    assert len(problems) == 1, problems
    assert problems[0]["path"] == "EPUB/images/pic", problems[0]
    assert "file extension" in problems[0]["message"], problems[0]
    assert any(
        line.startswith("Source EPUB check: 1 problem") for line in lines
    ), lines
    assert any(line.strip().startswith("- pic:") for line in lines), lines


def _check_cover(tmp: Path) -> None:
    """A generated Gutenberg cover is replaced and reported; mismatch is safe."""
    from PIL import Image

    chapter = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<html xmlns="http://www.w3.org/1999/xhtml">'
        "<head><title>One</title></head>"
        "<body><h1>One</h1><p>Alpha.</p><p>Beta.</p><p>Gamma.</p></body></html>"
    )
    cover_name = "111_222-cover.png"
    buffer = io.BytesIO()
    Image.new("RGB", (1600, 2400), (10, 20, 30)).save(buffer, format="PNG")
    cover_png = buffer.getvalue()

    def build_fixture(path: Path) -> None:
        build_epub(
            path,
            [
                {"id": "ch1", "href": "ch1.xhtml", "content": chapter},
                {
                    "id": "cover-img",
                    "href": cover_name,
                    "zip_name": cover_name,
                    "content": cover_png,
                    "media_type": "image/png",
                    "properties": "cover-image",
                    "in_spine": False,
                },
            ],
            title="The Project Gutenberg eBook of Cover Test | Project Gutenberg",
            creator="Cover Author",
        )

    # --- successful replacement ------------------------------------------
    source = tmp / "cover_success.epub"
    build_fixture(source)
    out_dir = tmp / "cover_success_out"
    engine = RecordingCoverEngine()
    components = PipelineComponents(
        translation_engine=engine,
        editor=ConsistencyEditor(FakeEditorProvider()),
        glossary_store=GlossaryStore(out_dir / "glossary.db"),
    )
    cfg = BookConfig(
        epub_path=str(source),
        output_dir=str(out_dir),
        glossary_db=str(out_dir / "glossary.db"),
        strip_gutenberg=True,
    )
    lines: list[str] = []
    result = run_book(
        cfg,
        resolve_path=_resolve,
        options=RunOptions(limit_chapters=["ch1"], batch_size=3),
        components=components,
        progress=lines.append,
    )
    assert ["Cover Test", "Cover Author"] in engine.inputs, engine.inputs
    assert any(
        "Replaced Project Gutenberg's generated cover" in line for line in lines
    ), lines
    assert result.epub_path is not None and result.epub_path.exists()
    assert find_gutenberg_cover(result.epub_path) is None
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["cover_replaced"] is True, manifest.get("cover_replaced")

    # --- strict count mismatch: run finishes, nothing misaligned ----------
    source_bad = tmp / "cover_mismatch.epub"
    build_fixture(source_bad)
    out_dir_bad = tmp / "cover_mismatch_out"
    engine_bad = RecordingCoverEngine(mismatch=True)
    components_bad = PipelineComponents(
        translation_engine=engine_bad,
        editor=ConsistencyEditor(FakeEditorProvider()),
        glossary_store=GlossaryStore(out_dir_bad / "glossary.db"),
    )
    cfg_bad = BookConfig(
        epub_path=str(source_bad),
        output_dir=str(out_dir_bad),
        glossary_db=str(out_dir_bad / "glossary.db"),
        strip_gutenberg=True,
    )
    lines_bad: list[str] = []
    result_bad = run_book(
        cfg_bad,
        resolve_path=_resolve,
        options=RunOptions(limit_chapters=["ch1"], batch_size=3),
        components=components_bad,
        progress=lines_bad.append,
    )
    assert any("Cover title not translated:" in line for line in lines_bad), lines_bad
    assert result_bad.epub_path is not None and result_bad.epub_path.exists()
    manifest_bad = json.loads((out_dir_bad / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_bad["cover_replaced"] is True, manifest_bad.get("cover_replaced")


@contextmanager
def _patch_module(module, **attributes):
    """Temporarily replace module attributes, restoring them afterwards."""
    saved = {name: getattr(module, name) for name in attributes}
    for name, value in attributes.items():
        setattr(module, name, value)
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(module, name, value)


def _check_epubcheck(tmp: Path) -> None:
    """Optional EPUBCheck wiring: available, unavailable, failure and preview."""
    output_result = EpubcheckResult(
        0,
        2,
        1,
        [
            "ERROR(RSC-005): EPUB/ch1.xhtml(1,7): boom",
            "ERROR(RSC-005): EPUB/ch1.xhtml(9,9): a new problem",
        ],
        "5.1.0",
    )
    source_result = EpubcheckResult(
        0,
        1,
        0,
        ["ERROR(RSC-005): EPUB/ch1.xhtml(3,4): boom"],
        "5.1.0",
    )

    # --- available: output and source are both validated -----------------
    available_dir = tmp / "epubcheck_available"
    available_calls: list[str] = []

    def fake_run(path, *, timeout=180):
        available_calls.append(str(path))
        return output_result if str(path).endswith(".kn.epub") else source_result

    cfg = _make_cfg(available_dir).model_copy(update={"epubcheck": True})
    lines: list[str] = []
    with _patch_module(
        pipeline_module,
        find_epubcheck=lambda: ("java", Path("epubcheck.jar")),
        run_epubcheck=fake_run,
    ):
        result = run_book(
            cfg,
            resolve_path=_resolve,
            options=RunOptions(limit_chapters=[CHAPTER_ID]),
            components=_make_components(available_dir)[0],
            progress=lines.append,
        )
    assert result.epub_path is not None and result.epub_path.exists()
    assert len(available_calls) == 2, available_calls
    assert available_calls[0].endswith(".kn.epub"), available_calls
    assert available_calls[1].endswith("sherlock_holmes.epub"), available_calls
    manifest = json.loads((available_dir / "manifest.json").read_text(encoding="utf-8"))
    block = manifest["epubcheck"]
    assert block["version"] == "5.1.0", block
    assert block["output"] == {
        "fatals": 0, "errors": 2, "warnings": 1,
        "messages": output_result.messages,
    }, block
    assert block["source"]["errors"] == 1, block
    assert block["new_errors"] == 1, block
    assert any(
        line == "EPUBCheck: 2 errors, 1 warnings in the translated EPUB "
        "(1 of the errors are also in the source)"
        for line in lines
    ), lines

    # --- unavailable: null in the manifest and a skip line ----------------
    missing_dir = tmp / "epubcheck_missing"
    cfg = _make_cfg(missing_dir).model_copy(update={"epubcheck": True})
    lines = []
    with _patch_module(pipeline_module, find_epubcheck=lambda: None):
        result = run_book(
            cfg,
            resolve_path=_resolve,
            options=RunOptions(limit_chapters=[CHAPTER_ID]),
            components=_make_components(missing_dir)[0],
            progress=lines.append,
        )
    assert result.epub_path is not None and result.epub_path.exists()
    manifest = json.loads((missing_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["epubcheck"] is None, manifest.get("epubcheck")
    assert "EPUBCheck not installed; skipped" in lines, lines

    # --- a raising validator does not fail the book -----------------------
    failing_dir = tmp / "epubcheck_failing"

    def exploding_run(path, *, timeout=180):
        raise RuntimeError("validator exploded")

    cfg = _make_cfg(failing_dir).model_copy(update={"epubcheck": True})
    lines = []
    with _patch_module(
        pipeline_module,
        find_epubcheck=lambda: ("java", Path("epubcheck.jar")),
        run_epubcheck=exploding_run,
    ):
        result = run_book(
            cfg,
            resolve_path=_resolve,
            options=RunOptions(limit_chapters=[CHAPTER_ID]),
            components=_make_components(failing_dir)[0],
            progress=lines.append,
        )
    assert result.epub_path is not None and result.epub_path.exists()
    manifest = json.loads((failing_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["epubcheck"] == {"error": "validator exploded"}, manifest["epubcheck"]
    assert any("validator exploded" in line for line in lines), lines

    # --- a preview does not run it at all ---------------------------------
    preview_dir = tmp / "epubcheck_preview"
    preview_calls = 0

    def counting_run(path, *, timeout=180):
        nonlocal preview_calls
        preview_calls += 1
        return output_result

    cfg = _make_cfg(preview_dir).model_copy(update={"epubcheck": True})
    with _patch_module(
        pipeline_module,
        find_epubcheck=lambda: ("java", Path("epubcheck.jar")),
        run_epubcheck=counting_run,
    ):
        result = run_book(
            cfg,
            resolve_path=_resolve,
            options=RunOptions(limit_chapters=[CHAPTER_ID], max_paragraphs=2),
            components=_make_components(preview_dir)[0],
            progress=lambda _msg: None,
        )
    assert result.epub_path is not None
    assert preview_calls == 0, preview_calls
    manifest = json.loads((preview_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["epubcheck"] is None, manifest.get("epubcheck")


def _check_editor_parser() -> None:
    """A paragraph whose EMOTION line the model dropped is kept, not "missing"."""
    raw = (
        "[P1]\nEMOTION: Anger\nಒಂದು\n\n"
        "[P2]\nಎರಡು\nಮುಂದುವರಿಕೆ\n\n"
        "[P3]\n\nEMOTION: sad\nಮೂರು\n\n"
        "[P4]\nEMOTION:\nನಾಲ್ಕು"
    )
    parsed = _parse_numbered_output(raw, 4)
    assert [p.text for p in parsed] == ["ಒಂದು", "ಎರಡು\nಮುಂದುವರಿಕೆ", "ಮೂರು", "ನಾಲ್ಕು"], parsed
    assert [p.emotion for p in parsed] == ["Anger", "Narration", "Sad", "Narration"], parsed

    # A tag with no text, or a missing tag, is still a mismatch.
    for bad in ("[P1]\nEMOTION: Narration\nಒಂದು\n\n[P2]\nEMOTION: Narration\n",
                "[P1]\nಒಂದು"):
        try:
            _parse_numbered_output(bad, 2)
        except EditorOutputError as exc:
            assert "missing [2]" in str(exc), exc
        else:
            raise AssertionError(f"expected a mismatch for {bad!r}")


def _check_context_detection() -> None:
    """detect_chapter_context: numbered chapters carry, named stories reset."""
    # Numbered chapters of one novel -> carry.
    assert detect_chapter_context([
        _chapter("c1", "CHAPTER I"),
        _chapter("c2", "Chapter 2"),
        _chapter("c3", "CHAPTER III THE DRAWERS OF WATER"),
        _chapter("c4", "IV"),
    ]) == "carry"

    # Named stories (a collection) -> reset.
    assert detect_chapter_context([
        _chapter("c1", "I. A SCANDAL IN BOHEMIA"),
        _chapter("c2", "The Red-Headed League"),
        _chapter("c3", "A Case of Identity"),
    ]) == "reset"

    # Front matter is ignored before the count: Preface + 3 numbered -> carry.
    assert detect_chapter_context([
        _chapter("c0", "Preface"),
        _chapter("c1", "Chapter 1"),
        _chapter("c2", "Chapter 2"),
        _chapter("c3", "Chapter 3"),
    ]) == "carry"

    # Too little signal: fewer than 3 remaining chapters -> reset.
    assert detect_chapter_context([
        _chapter("c1", "Chapter 1"),
        _chapter("c2", "Chapter 2"),
    ]) == "reset"

    # Untitled chapters give no signal -> reset.
    assert detect_chapter_context([
        _chapter("c1", None),
        _chapter("c2", None),
        _chapter("c3", None),
    ]) == "reset"

    # A descriptive TOC label whose document starts with its own chapter
    # heading (as rajmohan.epub loads) still counts as numbered.
    assert detect_chapter_context([
        _chapter("c1", "The Drawers of Water", ("RAJMOHAN'S WIFE",)),
        _chapter("c2", "The Two Cousins", ("CHAPTER II THE TWO COUSINS",)),
        _chapter("c3", "The Truant's Return Home", ("CHAPTER III THE TRUANT'S RETURN HOME",)),
    ]) == "carry"

    # The tail helper honours the configured paragraph count (and 0 is empty).
    chapter = _chapter("c1", "Chapter 1", ("one", "two", "three", "four"))
    assert chapter_context_tail(chapter, 2) == "three four"
    assert chapter_context_tail(chapter, 0) == ""

    # Real books (git-ignored ones skipped when absent).
    assert detect_chapter_context(load_epub_chapters(EPUB)) == "reset"
    rajmohan = ROOT / ".omc" / "showcase" / "books" / "rajmohan.epub"
    if rajmohan.exists():
        assert detect_chapter_context(load_epub_chapters(rajmohan)) == "carry"


def _run_context_book(
    epub: Path,
    output_dir: Path,
    *,
    chapter_context: str,
    translation_engine=None,
    limit_chapters: list[str] | None = None,
) -> list[str]:
    components = PipelineComponents(
        translation_engine=translation_engine or FakeTranslationEngine(),
        editor=ConsistencyEditor(FakeEditorProvider()),
        glossary_store=GlossaryStore(output_dir / "glossary.db"),
    )
    cfg = BookConfig(
        epub_path=str(epub),
        output_dir=str(output_dir),
        glossary_db=str(output_dir / "glossary.db"),
        chapter_context=chapter_context,
        context_tail_paragraphs=3,
        batch_size=10,
    )
    lines: list[str] = []
    run_book(
        cfg,
        resolve_path=_resolve,
        options=RunOptions(limit_chapters=limit_chapters),
        components=components,
        progress=lines.append,
    )
    return lines


def _checkpoint_prior_context(output_dir: Path, chapter_id: str) -> str:
    data = json.loads(
        (output_dir / "checkpoints" / f"{chapter_id}_0000.json").read_text(encoding="utf-8")
    )
    return data["prior_context_used"]


def _check_chapter_context_pipeline(tmp: Path) -> None:
    """Cross-chapter context: carry, reset, configured tail, resume, manifest."""
    epub = tmp / "continuous.epub"
    _multi_chapter_epub(epub, ["CHAPTER I", "CHAPTER II", "CHAPTER III"])
    source = {c.id: c for c in load_epub_chapters(epub)}

    # --- carry: chapter 2's first batch sees chapter 1's English tail ------
    carry_dir = tmp / "carry"
    _run_context_book(epub, carry_dir, chapter_context="carry")
    manifest = json.loads((carry_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["chapter_context"] == "carry", manifest["chapter_context"]
    expected1 = chapter_context_tail(source["ch1"], 3)
    assert expected1  # non-empty: 3 body paragraphs
    assert _checkpoint_prior_context(carry_dir, "ch1") == ""
    assert _checkpoint_prior_context(carry_dir, "ch2") == expected1, (
        _checkpoint_prior_context(carry_dir, "ch2")
    )
    assert _checkpoint_prior_context(carry_dir, "ch3") == chapter_context_tail(
        source["ch2"], 3
    )

    # --- reset: chapter 2's first batch gets no context -------------------
    reset_dir = tmp / "reset"
    _run_context_book(epub, reset_dir, chapter_context="reset")
    manifest = json.loads((reset_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["chapter_context"] == "reset", manifest["chapter_context"]
    assert _checkpoint_prior_context(reset_dir, "ch2") == ""

    # --- auto detects a continuous novel and says so ----------------------
    auto_dir = tmp / "auto"
    lines = _run_context_book(epub, auto_dir, chapter_context="auto")
    assert "Chapter context: carry (detected continuous novel)" in lines, lines
    manifest = json.loads((auto_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["chapter_context"] == "carry", manifest["chapter_context"]

    # --- auto detects a story collection and resets -----------------------
    stories = tmp / "stories.epub"
    _multi_chapter_epub(
        stories,
        ["I. A SCANDAL IN BOHEMIA", "II. THE RED-HEADED LEAGUE", "III. A CASE OF IDENTITY"],
    )
    stories_dir = tmp / "stories_out"
    story_lines = _run_context_book(stories, stories_dir, chapter_context="auto")
    assert "Chapter context: reset (detected separate stories)" in story_lines, story_lines
    manifest = json.loads((stories_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["chapter_context"] == "reset", manifest["chapter_context"]
    assert _checkpoint_prior_context(stories_dir, "ch2") == ""

    # --- resume in carry mode: chapter 1 from checkpoints, ch2 still gets its tail
    resume_dir = tmp / "carry_resume"
    _run_context_book(epub, resume_dir, chapter_context="carry", limit_chapters=["ch1"])
    _run_context_book(
        epub,
        resume_dir,
        chapter_context="carry",
        translation_engine=BlockChapterOneEngine(),
    )
    manifest = json.loads((resume_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "ch1" in manifest["skipped"], manifest["skipped"]
    assert _checkpoint_prior_context(resume_dir, "ch2") == chapter_context_tail(
        source["ch1"], 3
    )


def _check_number_only_paragraphs(tmp: Path) -> None:
    """Number-only paragraphs skip the engine and editor and stay verbatim (#27)."""
    from kannada_epub.book_translator import needs_translation
    from kannada_epub.epub_io import Chapter, Paragraph

    for text in ("7.", "II", "XIV.", "— 12 —", "iv", "(12)", "..."):
        assert not needs_translation(text), text
    for text in ("CIVIL", "DID", "A.", "Chapter 1", "I went home.", "3rd"):
        assert needs_translation(text), text

    texts = ["1.", "The first poem.", "II", "The second poem.", "7."]
    chapter = Chapter(
        id="poems", title="Poems",
        paragraphs=[Paragraph(i, text) for i, text in enumerate(texts)],
    )

    class RecordingEngine(FakeTranslationEngine):
        def __init__(self):
            super().__init__()
            self.inputs: list[list[str]] = []

        def translate_paragraphs(self, paragraphs, src_lang, tgt_lang):
            self.inputs.append(list(paragraphs))
            return super().translate_paragraphs(paragraphs, src_lang, tgt_lang)

    engine = RecordingEngine()
    translator = BookTranslator(
        translation_engine=engine,
        glossary_store=GlossaryStore(tmp / "numbers-glossary.db"),
        consistency_editor=ConsistencyEditor(FakeEditorProvider()),
        batch_size=10,
    )
    [batch] = translator.translate_chapters([chapter])
    assert engine.inputs == [["The first poem.", "The second poem."]], engine.inputs
    assert batch.paragraph_start == 0 and batch.paragraph_end == 5
    assert batch.edited_kannada[0] == "1." and batch.edited_kannada[2] == "II"
    assert batch.edited_kannada[4] == "7.", batch.edited_kannada
    assert batch.edited_kannada[1] == "ಕನ್ನಡ The first", batch.edited_kannada
    assert batch.edited_kannada[3] == "ಕನ್ನಡ The second", batch.edited_kannada
    assert batch.draft_kannada[0] == "1." and batch.edited_emotions[0] == "Narration"

    # A batch with nothing to translate never calls the engine or the editor.
    only_numbers = Chapter(
        id="numbers", title=None, paragraphs=[Paragraph(0, "3."), Paragraph(1, "IV")]
    )
    engine.inputs.clear()
    [batch] = translator.translate_chapters([only_numbers])
    assert engine.inputs == [] and batch.edited_kannada == ["3.", "IV"]

    # Older checkpoints with invented text for "7." are repaired on resume.
    from kannada_epub.book_translator import restore_untranslatable
    from kannada_epub.pipeline import _matching_checkpoints, _write_checkpoint

    [good] = translator.translate_chapters([chapter])
    good.edited_kannada[4] = "7ರಂದು ನಡೆಯಿತು."
    good.draft_kannada[4] = "7ರಂದು ನಡೆಯಿತು."
    good.edited_emotions[4] = "Happy"
    checkpoints = tmp / "old-checkpoints"
    checkpoints.mkdir()
    _write_checkpoint(checkpoints, good)
    [resumed] = _matching_checkpoints(checkpoints, chapter)
    assert resumed.edited_kannada[4] == "7." and resumed.draft_kannada[4] == "7."
    assert resumed.edited_emotions[4] == "Narration"
    assert resumed.edited_kannada[1] == "ಕನ್ನಡ The first", resumed.edited_kannada
    assert restore_untranslatable(resumed) == 0


def _check_heading_routing(tmp: Path) -> None:
    """Headings reach the editor flagged, numbered within the editor's slice."""
    from kannada_epub.consistency_editor import SYSTEM_PROMPT, _build_user_prompt

    texts = [
        ("A Scandal in Bohemia", "heading"),
        ("Some body text.", "text"),
        ("More body text.", "text"),
        ("II", "heading"),
        ("The Red-Headed League", "heading"),
        ("Final body text.", "text"),
    ]
    chapter = Chapter(
        id="headings",
        title="Headings",
        paragraphs=[Paragraph(i, text, kind) for i, (text, kind) in enumerate(texts)],
    )

    # --- one batch, no split: HEADINGS numbers skip the number-only "II" ----
    recording = RecordingEditorProvider(FakeEditorProvider())
    translator = BookTranslator(
        translation_engine=FakeTranslationEngine(),
        glossary_store=GlossaryStore(tmp / "headings.db"),
        consistency_editor=ConsistencyEditor(recording),
        batch_size=10,
    )
    [batch] = translator.translate_chapters([chapter])
    assert len(recording.user_prompts) == 1, recording.user_prompts
    prompt = recording.user_prompts[0]
    # "II" bypasses the editor (#27), so the wanted list is
    # [Scandal, body, body, Red-Headed, body] and its headings are P1, P4.
    assert _headings_in_prompt(prompt) == ["P1", "P4"], prompt
    assert "HEADINGS:" in prompt
    assert batch.edited_kannada[3] == "II", batch.edited_kannada

    # --- a chapter without headings: no HEADINGS block, prompt unchanged ----
    plain = Chapter(
        id="plain",
        title="Plain",
        paragraphs=[Paragraph(0, "First body."), Paragraph(1, "Second body.")],
    )
    recording_plain = RecordingEditorProvider(FakeEditorProvider())
    translator_plain = BookTranslator(
        translation_engine=FakeTranslationEngine(),
        glossary_store=GlossaryStore(tmp / "plain.db"),
        consistency_editor=ConsistencyEditor(recording_plain),
        batch_size=10,
    )
    translator_plain.translate_chapters([plain])
    assert len(recording_plain.user_prompts) == 1, recording_plain.user_prompts
    captured = recording_plain.user_prompts[0]
    assert "HEADINGS:" not in captured, captured
    draft = [f"ಕನ್ನಡ {p.text[:10]}" for p in plain.paragraphs]
    expected = _build_user_prompt(
        "\n\n".join(draft), {}, "", "neutral, standard written Kannada"
    )
    assert captured == expected, (captured, expected)

    # --- forced split: each half numbers headings within its own slice ------
    truncating = RecordingEditorProvider(TruncatingEditorProvider(max_ok=2))
    translator_split = BookTranslator(
        translation_engine=FakeTranslationEngine(),
        glossary_store=GlossaryStore(tmp / "headings_split.db"),
        consistency_editor=ConsistencyEditor(truncating),
        batch_size=10,
    )
    translator_split.translate_chapters([chapter])
    # Call order: whole 5 (fails), left 2, right 3 (fails), its left 1, its
    # right 2 — each prompt numbers headings relative to the paragraphs it has.
    assert [_headings_in_prompt(p) for p in truncating.user_prompts] == [
        ["P1", "P4"],
        ["P1"],
        ["P2"],
        [],
        ["P1"],
    ], truncating.user_prompts

    # --- the system prompt carries the new heading rule ---------------------
    assert "HEADINGS" in SYSTEM_PROMPT, SYSTEM_PROMPT
    assert "A Scandal in Bohemia" in SYSTEM_PROMPT, SYSTEM_PROMPT
    assert "HEADINGS" in recording.system_prompts[0], recording.system_prompts[0]


def _qa_chapter() -> tuple[Chapter, list[TranslatedBatch]]:
    """One chapter of three paragraphs and its single batch."""
    chapter = Chapter(
        id="qa-chapter",
        title="QA",
        paragraphs=[Paragraph(0, "Alpha."), Paragraph(1, "Beta."), Paragraph(2, "Gamma.")],
    )
    batch = TranslatedBatch(
        chapter_id="qa-chapter",
        chapter_title="QA",
        paragraph_start=0,
        paragraph_end=3,
        source_english=["Alpha.", "Beta.", "Gamma."],
        draft_kannada=["draft A", "draft B", "draft C"],
        edited_kannada=["kn A", "kn B", "kn C"],
        edited_emotions=["Narration"] * 3,
        prior_context_used="",
    )
    return chapter, [batch]


def _qa_components(engine, out_dir: Path) -> PipelineComponents:
    return PipelineComponents(
        translation_engine=engine,
        editor=ConsistencyEditor(FakeEditorProvider()),
        glossary_store=GlossaryStore(out_dir / "glossary.db"),
        qa=(RetryBackTranslator(), FakeEmbedder()),
    )


def _check_qa_retry(tmp: Path) -> None:
    """QA retries vary settings, keep only improvements, and never misalign."""
    # --- vary_retry: retry path used, improved kept, worse discarded ------
    run_dir = tmp / "qa_retry"
    engine = RetryRecordingEngine()
    cfg = _make_cfg(run_dir)
    chapter, batches = _qa_chapter()
    lines: list[str] = []
    results = pipeline_module._run_chapter_qa(
        cfg, _qa_components(engine, run_dir), chapter, batches, run_dir, lines.append
    )
    assert engine.retry_calls == 1, engine.retry_calls
    assert engine.retry_batch_sizes == [2], engine.retry_batch_sizes
    # Position 0 improved -> retry kept; position 1 worsened -> draft kept.
    assert results[0].status == PASS, results[0]
    assert results[1].status == RETRY, results[1]  # worse retry discarded
    assert batches[0].edited_kannada == ["RETRY Alpha.", "kn B", "kn C"], (
        batches[0].edited_kannada
    )
    assert any(
        line == "[qa-chapter] QA retry: 2 paragraph(s) re-translated "
        "(test variation); 1 improved"
        for line in lines
    ), lines
    cache = json.loads((run_dir / "qa" / "qa-chapter.json").read_text(encoding="utf-8"))
    assert cache["retries"] == [
        {"paragraph_index": 0, "kept": True},
        {"paragraph_index": 1, "kept": False},
    ], cache["retries"]

    # --- vary_retry=False: the plain method is used, not the retry --------
    off_dir = tmp / "qa_retry_off"
    off_engine = RetryRecordingEngine()
    off_cfg = _make_cfg(off_dir)
    off_cfg.qa.vary_retry = False
    off_chapter, off_batches = _qa_chapter()
    off_lines: list[str] = []
    pipeline_module._run_chapter_qa(
        off_cfg,
        _qa_components(off_engine, off_dir),
        off_chapter,
        off_batches,
        off_dir,
        off_lines.append,
    )
    assert off_engine.retry_calls == 0, off_engine.retry_calls
    assert off_engine.calls == 1, off_engine.calls  # the plain method ran
    assert any(
        line == "[qa-chapter] QA retry: 2 paragraph(s) re-translated "
        "(same settings); 1 improved"
        for line in off_lines
    ), off_lines

    # --- a wrong retry count raises instead of shifting paragraphs --------
    bad_dir = tmp / "qa_retry_bad"
    bad_cfg = _make_cfg(bad_dir)
    bad_chapter, bad_batches = _qa_chapter()
    try:
        pipeline_module._run_chapter_qa(
            bad_cfg,
            _qa_components(MismatchedRetryEngine(), bad_dir),
            bad_chapter,
            bad_batches,
            bad_dir,
            lambda _msg: None,
        )
    except RuntimeError as exc:
        assert "refusing to misalign" in str(exc), exc
    else:
        raise AssertionError("expected RuntimeError for a misaligned retry")


def main() -> None:
    source = {c.id: c for c in load_epub_chapters(EPUB)}
    source_item4 = source[CHAPTER_ID]
    # Number-only paragraphs ("I.") are copied verbatim, not translated (#27).
    from kannada_epub.book_translator import needs_translation

    expected_kannada = [
        f"ಕನ್ನಡ {p.text[:10]}" if needs_translation(p.text) else p.text
        for p in source_item4.paragraphs[:N_PARAGRAPHS]
    ]
    assert len(expected_kannada) == N_PARAGRAPHS

    tmp = Path(tempfile.mkdtemp())
    try:
        components, engine = _make_components(tmp)
        cfg = _make_cfg(tmp)
        events: list[dict] = []
        options = RunOptions(
            limit_chapters=[CHAPTER_ID],
            max_paragraphs=N_PARAGRAPHS,
            batch_size=BATCH_SIZE,
        )

        result = run_book(
            cfg,
            resolve_path=_resolve,
            options=options,
            components=components,
            progress=lambda _msg: None,
            on_event=events.append,
        )

        event_types = [event["type"] for event in events]
        assert event_types == ["start", "chapter", "stage", "stage", "done"], event_types
        assert events[0]["chapters"] == [{
            "id": CHAPTER_ID,
            "title": source_item4.title or source_item4.id,
            "paragraphs": N_PARAGRAPHS,
        }]
        assert events[1] == {
            "type": "chapter", "id": CHAPTER_ID, "index": 1, "total": 1, "resumed": False
        }
        assert events[2] == {"type": "stage", "stage": "qa"}
        assert events[3] == {"type": "stage", "stage": "epub"}

        # --- EPUB written, first 6 paragraphs Kannada, rest English --------
        assert result.epub_path is not None
        assert result.epub_path.exists()
        assert result.cancelled is False
        assert result.chapters == [CHAPTER_ID]

        reloaded = {c.id: c for c in load_epub_chapters(result.epub_path)}[CHAPTER_ID]
        assert [p.text for p in reloaded.paragraphs[:N_PARAGRAPHS]] == expected_kannada
        assert [p.text for p in reloaded.paragraphs[N_PARAGRAPHS:]] == [
            p.text for p in source_item4.paragraphs[N_PARAGRAPHS:]
        ]

        # --- paragraph 1's element is flagged ------------------------------
        soup = BeautifulSoup(_chapter_document(result.epub_path, CHAPTER_ID), "lxml")
        blocks = soup.find_all(BLOCK_TAGS)
        flagged_element = blocks[source_item4.paragraphs[1].index]
        assert "qa-review-flag" in (flagged_element.get("class") or []), (
            f"missing qa-review-flag on paragraph 1: {flagged_element.get('class')!r}"
        )

        # --- qa_report: total 6, flagged 1, paragraph 0 retried to PASS ----
        assert result.qa_report_path is not None
        report = json.loads(result.qa_report_path.read_text(encoding="utf-8"))
        assert report["summary"]["total"] == N_PARAGRAPHS, report["summary"]
        assert report["summary"]["flagged"] == 1, report["summary"]
        assert report["summary"]["retry"] == 0, report["summary"]
        idx0 = source_item4.paragraphs[0].index
        result0 = next(r for r in report["results"] if r["paragraph_index"] == idx0)
        assert result0["status"] == PASS, result0
        assert result0["similarity_score"] > 0.75
        # Paragraph 1 stays flagged in the report.
        idx1 = source_item4.paragraphs[1].index
        result1 = next(r for r in report["results"] if r["paragraph_index"] == idx1)
        assert result1["status"] == FLAGGED_FOR_REVIEW, result1

        # Initial translation: 2 batches of 3 -> 2 calls; QA retry: 1 -> total 3.
        assert engine.calls == 3, engine.calls

        # --- second run resumes: no extra translate calls ------------------
        result2 = run_book(
            cfg,
            resolve_path=_resolve,
            options=options,
            components=components,
            progress=lambda _msg: None,
        )
        assert engine.calls == 3, f"resume re-ran translation: {engine.calls} calls"
        assert result2.epub_path is not None and result2.epub_path.exists()

        # --- cancellation: manifest only, no EPUB --------------------------
        cancel_dir = tmp / "cancelled"
        cancel_components = PipelineComponents(
            translation_engine=FakeTranslationEngine(),
            editor=ConsistencyEditor(FakeEditorProvider()),
            glossary_store=GlossaryStore(cancel_dir / "glossary.db"),
        )
        cancel_event = threading.Event()
        cancel_event.set()
        cancelled = run_book(
            _make_cfg(cancel_dir),
            resolve_path=_resolve,
            options=options,
            components=cancel_components,
            progress=lambda _msg: None,
            cancel=cancel_event,
        )
        assert cancelled.cancelled is True
        assert cancelled.epub_path is None
        assert not (cancel_dir / "sherlock_holmes.kn.epub").exists()
        assert (cancel_dir / "manifest.json").exists()

        # --- the first run was a preview: labelled as one --------------------
        assert result.epub_path.name == "sherlock_holmes.kn.preview.epub", result.epub_path
        manifest = json.loads((tmp / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["preview"] is True
        assert manifest["paragraphs_translated"] == N_PARAGRAPHS
        assert manifest["paragraphs_total"] == len(source_item4.paragraphs)
        assert manifest["source_problems"] == [], manifest["source_problems"]
        assert manifest["edition"]["needs_review"] is False, manifest["edition"]

        # --- regression: preview with QA, then a full run in the same folder --
        # The preview's checkpoint and QA cache share names with the full run's
        # first batch. They used to be reused, leaving most of the book English.
        chapter_id = "item6"
        full_source = source[chapter_id]
        both_dir = tmp / "preview_then_full"
        for max_paragraphs in (2, None):
            run_components, _ = _make_components(both_dir)
            last = run_book(
                _make_cfg(both_dir),
                resolve_path=_resolve,
                options=RunOptions(
                    limit_chapters=[chapter_id],
                    max_paragraphs=max_paragraphs,
                    batch_size=BATCH_SIZE,
                ),
                components=run_components,
                progress=lambda _msg: None,
            )
        assert last.epub_path.name == "sherlock_holmes.kn.epub", last.epub_path
        full = {c.id: c for c in load_epub_chapters(last.epub_path)}[chapter_id]
        untranslated = [p.text for p in full.paragraphs if not p.text.startswith("ಕನ್ನಡ")]
        assert not untranslated, (
            f"{len(untranslated)} of {len(full.paragraphs)} paragraphs left in English"
        )
        assert len(full.paragraphs) == len(full_source.paragraphs)
        report = json.loads(last.qa_report_path.read_text(encoding="utf-8"))
        assert report["summary"]["total"] == len(full_source.paragraphs), report["summary"]
        manifest = json.loads((both_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["preview"] is False
        assert manifest["paragraphs_translated"] == manifest["paragraphs_total"]

        # --- editor split-and-retry, and resume across a changed batch_size -
        _check_split_and_resume(tmp)
        # --- a defective source is reported but the run still completes -----
        _check_source_problems(tmp)
        # --- a generated Gutenberg cover is replaced, not misaligned --------
        _check_cover(tmp)
        # --- optional EPUBCheck wiring --------------------------------------
        _check_epubcheck(tmp)
        # --- cross-chapter context: carry/reset, tail size, resume, manifest -
        _check_chapter_context_pipeline(tmp)
        # --- QA retry: varied settings, improved-only, strict count --------
        _check_qa_retry(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    _check_context_detection()
    _check_editor_parser()
    _number_tmp = Path(tempfile.mkdtemp())
    try:
        _check_number_only_paragraphs(_number_tmp)
    finally:
        shutil.rmtree(_number_tmp, ignore_errors=True)

    _heading_tmp = Path(tempfile.mkdtemp())
    try:
        _check_heading_routing(_heading_tmp)
    finally:
        shutil.rmtree(_heading_tmp, ignore_errors=True)

    # --- cloud-only import check (no local ML libs) ------------------------
    _check_cloud_only_import()

    print("test_pipeline: all assertions passed")


if __name__ == "__main__":
    main()
