"""Smoke test for the importable pipeline (``kannada_epub.pipeline``).

Everything runs against FAKE translation/editor/QA components and the real
Sherlock Holmes EPUB: no models, no network. It also proves the package imports
and builds a cloud provider with the optional local ML stack blocked.

Run: .venv/bin/python scripts/test_pipeline.py
"""

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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import lxml.etree as ET
from bs4 import BeautifulSoup

from kannada_epub.config import BookConfig
from kannada_epub.consistency_editor import ConsistencyEditor
from kannada_epub.epub_io import BLOCK_TAGS, load_epub_chapters
from kannada_epub.glossary import GlossaryStore
from kannada_epub.pipeline import PipelineComponents, RunOptions, run_book
from kannada_epub.qa import FLAGGED_FOR_REVIEW, PASS
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


def main() -> None:
    source = {c.id: c for c in load_epub_chapters(EPUB)}
    source_item4 = source[CHAPTER_ID]
    expected_kannada = [
        f"ಕನ್ನಡ {p.text[:10]}" for p in source_item4.paragraphs[:N_PARAGRAPHS]
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
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # --- cloud-only import check (no local ML libs) ------------------------
    _check_cloud_only_import()

    print("test_pipeline: all assertions passed")


if __name__ == "__main__":
    main()
