"""Tests for target-language support (GitHub issue #23).

Fast, offline, no models: the language table, prompt templates, EPUB writer,
output names, cover rendering and a pipeline run with fake engines.

Run: .venv/bin/python scripts/test_languages.py
"""

import io
import json
import re
import shutil
import sys
import tempfile
import warnings
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lxml.etree as ET
import test_pipeline as tp
from bs4 import BeautifulSoup

from kannada_epub import consistency_editor as ce
from kannada_epub import pipeline as pipeline_module
from kannada_epub import translation
from kannada_epub.config import BookConfig
from kannada_epub.epub_check import check_source_epub
from kannada_epub.epub_io import find_opf_path, load_epub_chapters
from kannada_epub.epub_writer import (
    MACHINE_TRANSLATION_CONTRIBUTOR,
    write_translated_epub,
)
from kannada_epub.languages import LANGUAGES, get_language
from kannada_epub.pipeline import RunOptions, output_file_names, run_book

ROOT = Path(__file__).resolve().parent.parent
EPUB = ROOT / "data" / "sherlock_holmes.epub"
FONTS = ROOT / "assets" / "fonts"
DC_NS = "http://purl.org/dc/elements/1.1/"
OPF_NS = "http://www.idpf.org/2007/opf"

# ---------------------------------------------------------------------------
# The pre-#23 Kannada prompt constants, copied verbatim as expected values.
# ---------------------------------------------------------------------------
EXPECTED_EMOTIONS = [
    "Command", "Anger", "Narration", "Conversation", "Disgust", "Fear",
    "Happy", "Neutral", "Proper Noun", "News", "Sad", "Surprise",
]

EXPECTED_EDITOR_SYSTEM_PROMPT = """You are a Kannada-language consistency editor working on one chapter of a book \
that has already been machine-translated from English into Kannada.

You are NOT translating from scratch. You are given a draft Kannada chapter and must return an \
edited version that only fixes:
1. Glossary terms — every term in GLOSSARY must appear exactly as given wherever its English \
   source term occurs, replacing whatever the draft used instead.
2. Pronoun / referent consistency — resolve ambiguous or inconsistent pronouns and character \
   references using PRIOR_CHAPTER_CONTEXT.
3. Register — keep the tone consistent with REGISTER across the whole chapter.
4. Headings and titles — paragraphs listed under HEADINGS are headings or titles. Translate their \
   meaning into natural Kannada instead of transliterating English words into Kannada script. For \
   example, "A Scandal in Bohemia" should become a Kannada phrase meaning "a scandal in Bohemia", \
   with only the place name "Bohemia" transliterated. Names of people and places may stay \
   transliterated.

Do not rewrite sentences that are already correct. Do not change meaning, add content, or remove \
content.

DRAFT_CHAPTER's paragraphs are each numbered with a [P<n>] tag, e.g. "[P1]", "[P2]". Your output \
MUST have exactly the same number of paragraphs, in the same order, each carrying the same [P<n>] \
tag it had in the input. This holds even when DRAFT_CHAPTER has only one paragraph, and even when \
a paragraph is long or contains many sentences: one input paragraph always produces exactly one \
output entry. Never split one input paragraph into several output entries and never merge several \
input paragraphs into one, regardless of how much the tone or subject varies within a paragraph.

Additionally, classify the emotional tone each paragraph should be narrated in for an audiobook \
reading. Choose exactly one tag per paragraph from this fixed set: """ + ", ".join(EXPECTED_EMOTIONS) + """. \
If a paragraph's tone varies internally, pick the single tag that best represents it as a whole — \
do not split it to give different parts different tags. Use "Narration" for ordinary \
descriptive/narrative prose; use a more specific tag only when the paragraph's content clearly \
signals it (e.g. dialogue expressing anger -> "Anger").

Return each paragraph as its [P<n>] tag, then "EMOTION: <tag>" on its own line, then the edited \
paragraph text — no commentary, preamble, or markdown anywhere in your answer. Example shape for \
two paragraphs:

[P1]
EMOTION: Narration
<edited paragraph 1 text>

[P2]
EMOTION: Happy
<edited paragraph 2 text>"""

EXPECTED_EDITOR_INLINE_MARKUP = """

This chapter was translated from a source with inline formatting. Words that were bold, italic, a \
link or a footnote reference in the source are wrapped in numbered markers: ⟦1⟧ … ⟦/1⟧, ⟦2⟧ … \
⟦/2⟧, and so on. Keep every marker pair around the Kannada words that translate the marked English \
words. Never add, remove or renumber markers, never move a marker onto different words, and never \
let one marker pair cross another."""

EXPECTED_CLOUD_SYSTEM_PROMPT = (
    "You are an English-to-Kannada literary translator. Translate each numbered paragraph "
    "below into natural, standard written Kannada. Preserve meaning exactly: do not add, "
    "remove, or explain anything. Keep proper nouns, acronyms, and code in their original "
    "form. Return ONLY a JSON array of translated strings, one per input paragraph, in the "
    "same order."
)

EXPECTED_CLOUD_RETRY = (
    "This is a second attempt at the same passage: produce a faithful, complete translation "
    "that keeps every detail of the original, and translate ordinary English words into "
    "Kannada rather than transliterating them."
)

EXPECTED_QA_SYSTEM_PROMPT = (
    "You are a Kannada-to-English translator. Translate each numbered Kannada paragraph below "
    "into literal, faithful English. Preserve meaning exactly: do not add, remove, summarize, "
    "or explain anything. Return ONLY a JSON array of translated strings, one per input "
    "paragraph, in the same order."
)


# ---------------------------------------------------------------------------
# The language table
# ---------------------------------------------------------------------------
def check_language_table() -> None:
    assert set(LANGUAGES) == {"kn", "ta", "te", "ml", "hi"}, sorted(LANGUAGES)
    expected = {
        "kn": ("Kannada", "kan_Knda", "kn-IN", "Noto Sans Kannada", "ಯಂತ್ರ ಅನುವಾದ"),
        "ta": ("Tamil", "tam_Taml", "ta-IN", "Noto Sans Tamil", "இயந்திர மொழிபெயர்ப்பு"),
        "te": ("Telugu", "tel_Telu", "te-IN", "Noto Sans Telugu", "యంత్ర అనువాదం"),
        "ml": ("Malayalam", "mal_Mlym", "ml-IN", "Noto Sans Malayalam", "യന്ത്ര വിവർത്തനം"),
        "hi": ("Hindi", "hin_Deva", "hi-IN", "Noto Sans Devanagari", "मशीनी अनुवाद"),
    }
    for key, (name, flores, bcp47, family, label) in expected.items():
        language = LANGUAGES[key]
        assert language.key == key
        assert language.name == name, (key, language.name)
        assert language.flores == flores, (key, language.flores)
        assert language.bcp47 == bcp47, (key, language.bcp47)
        assert language.font_family == family, (key, language.font_family)
        assert language.font_regular == f"{family.replace(' ', '')}-Regular.ttf", key
        assert language.font_bold == f"{family.replace(' ', '')}-Bold.ttf", key
        assert language.machine_translation_label == label, (key, language.machine_translation_label)
        # The script regex matches a character from the language's block.
        assert re.search(language.script_re, {"kn": "ಕ", "ta": "க", "te": "క", "ml": "ക", "hi": "क"}[key])

    assert get_language("ta") is LANGUAGES["ta"]
    try:
        get_language("xx")
    except ValueError as exc:
        assert "xx" in str(exc), exc
    else:
        raise AssertionError("expected ValueError for an unknown language key")

    # BookConfig validates the key.
    try:
        BookConfig(epub_path="b.epub", output_dir="o", glossary_db="g.db", target_language="xx")
    except Exception as exc:  # pydantic ValidationError
        assert "target_language" in str(exc), exc
    else:
        raise AssertionError("expected an invalid target_language to be rejected")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
def check_prompts() -> None:
    assert ce.SYSTEM_PROMPT == EXPECTED_EDITOR_SYSTEM_PROMPT
    assert ce.system_prompt_for("Kannada") == EXPECTED_EDITOR_SYSTEM_PROMPT
    assert ce.INLINE_MARKUP_RULE == EXPECTED_EDITOR_INLINE_MARKUP
    assert ce.inline_markup_rule_for("Kannada") == EXPECTED_EDITOR_INLINE_MARKUP

    assert translation.cloud._SYSTEM_PROMPT == EXPECTED_CLOUD_SYSTEM_PROMPT
    assert translation.cloud.system_prompt_for("Kannada") == EXPECTED_CLOUD_SYSTEM_PROMPT
    assert translation.cloud._RETRY_INSTRUCTION == EXPECTED_CLOUD_RETRY
    assert translation.cloud.retry_instruction_for("Kannada") == EXPECTED_CLOUD_RETRY

    from kannada_epub.qa import backtranslate

    assert backtranslate._SYSTEM_PROMPT == EXPECTED_QA_SYSTEM_PROMPT
    assert backtranslate.system_prompt_for("Kannada") == EXPECTED_QA_SYSTEM_PROMPT

    # A non-Kannada name replaces every Kannada mention, and the editor's
    # constructor stores it.
    tamil_editor_prompt = ce.system_prompt_for("Tamil")
    assert "Tamil" in tamil_editor_prompt
    assert "Kannada" not in tamil_editor_prompt
    assert "Tamil" in ce.inline_markup_rule_for("Tamil")
    assert "Kannada" not in translation.cloud.system_prompt_for("Tamil")
    assert "Tamil" in translation.cloud.retry_instruction_for("Tamil")
    assert "Kannada" not in backtranslate.system_prompt_for("Tamil")

    class _Provider:
        def complete(self, system_prompt, user_prompt):  # pragma: no cover - not called
            raise AssertionError("not called")

    assert ce.ConsistencyEditor(_Provider(), language_name="Tamil")._system_prompt == ce.system_prompt_for("Tamil")


# ---------------------------------------------------------------------------
# Output names
# ---------------------------------------------------------------------------
def check_output_file_names() -> None:
    assert output_file_names("book", False) == ("book.kn.epub", "book.kn.wav")
    assert output_file_names("book", True) == ("book.kn.preview.epub", "book.kn.preview.wav")
    assert output_file_names("book", False, "ta") == ("book.ta.epub", "book.ta.wav")
    assert output_file_names("book", True, "ta") == (
        "book.ta.preview.epub",
        "book.ta.preview.wav",
    )


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------
def _fake_translations(chapters, prefix: str) -> dict[str, dict[int, str]]:
    return {
        chapter.id: {p.index: f"{prefix} {i}" for i, p in enumerate(chapter.paragraphs)}
        for chapter in chapters
    }


def _write(tmp: Path, language_key: str | None, name: str) -> tuple[Path, list[str]]:
    chapters = load_epub_chapters(EPUB)
    translations = _fake_translations(chapters, "தமிழ்" if language_key == "ta" else "ಪಠ್ಯ")
    out = tmp / name
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if language_key is None:
            write_translated_epub(EPUB, translations, out)
        else:
            write_translated_epub(EPUB, translations, out, language=LANGUAGES[language_key])
    return out, [str(item.message) for item in caught]


def check_writer(tmp: Path) -> None:
    # --- Tamil: no font bundled, warning, lang/metadata switched ----------
    chapters = load_epub_chapters(EPUB)
    translations = _fake_translations(chapters, "தமிழ்")
    out = tmp / "sherlock.ta.epub"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        write_translated_epub(EPUB, translations, out, language=LANGUAGES["ta"])
    messages = [str(item.message) for item in caught]
    assert any(
        "No bundled font for Tamil; readers will use their own fonts." == message
        for message in messages
    ), messages

    assert check_source_epub(out) == [], check_source_epub(out)

    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        assert not any(name.endswith(".ttf") for name in names), names
        assert not any(name.endswith("OFL.txt") for name in names), names
        css_name = next(name for name in names if name.endswith("kannada.css"))
        css = zf.read(css_name)
        assert b"@font-face" not in css, css
        assert b'"Noto Sans Tamil", serif' in css, css
        opf_path = find_opf_path(zf)
        opf = ET.fromstring(zf.read(opf_path))
        languages = [element.text for element in opf.findall(f".//{{{DC_NS}}}language")]
        assert languages == ["ta"], languages
        contributors = [
            element.text for element in opf.findall(f".//{{{DC_NS}}}contributor")
        ]
        assert "Unreviewed machine translation into Tamil (Indic Book Translator)" in contributors
        assert MACHINE_TRANSLATION_CONTRIBUTOR not in contributors

        manifest = {
            item.get("id"): item.get("href")
            for item in opf.findall(f".//{{{OPF_NS}}}manifest/{{{OPF_NS}}}item")
        }
        doc_path = (Path(opf_path).parent / manifest["item4"]).as_posix()
        soup = BeautifulSoup(zf.read(doc_path), "lxml")
        assert soup.html.get("lang") == "ta", soup.html.get("lang")
        assert soup.html.get("xml:lang") == "ta"

    # --- Kannada: passing the language is byte-identical to the default ----
    default_out, default_warnings = _write(tmp, None, "default.epub")
    explicit_out, explicit_warnings = _write(tmp, "kn", "explicit.epub")
    assert not any("No bundled font" in w for w in default_warnings), default_warnings
    assert not any("No bundled font" in w for w in explicit_warnings), explicit_warnings
    with zipfile.ZipFile(default_out) as a, zipfile.ZipFile(explicit_out) as b:
        assert set(a.namelist()) == set(b.namelist())
        for name in a.namelist():
            assert a.read(name) == b.read(name), f"entry changed: {name}"


# ---------------------------------------------------------------------------
# Cover
# ---------------------------------------------------------------------------
def check_cover() -> None:
    try:
        from PIL import Image, features
    except ImportError:
        print("test_languages: Pillow missing; skipping cover render checks")
        return
    if not features.check("raqm"):
        # The Tamil font is absent, so shaping never runs for this check; the
        # English-only fallback is exactly what is under test.
        print("test_languages: raqm missing; cover falls back to English only")

    from kannada_epub import cover as cover_module
    from kannada_epub.cover import CoverText

    drawn_blocks: list[str] = []
    drawn_segments: list[list[str]] = []
    original_block = cover_module._draw_block
    original_segments = cover_module._draw_segments

    def fake_block(draw, text, *_args, **_kwargs):
        drawn_blocks.append(text)

    def fake_segments(draw, segments, *_args, **_kwargs):
        drawn_segments.append(list(segments))

    cover_module._draw_block = fake_block
    cover_module._draw_segments = fake_segments
    try:
        data = cover_module.render_cover(
            CoverText("தமிழ் தலைப்பு", "ஆசிரியர்", "English Title", "Author"),
            FONTS,
            language=LANGUAGES["ta"],
        )
    finally:
        cover_module._draw_block = original_block
        cover_module._draw_segments = original_segments

    image = Image.open(io.BytesIO(data))
    assert image.format == "PNG" and image.size == (1600, 2400)
    # No Tamil font: English-only title and footer; Tamil text would render
    # as missing glyphs in the image.
    assert "English Title" in drawn_blocks, drawn_blocks
    assert "தமிழ் தலைப்பு" not in drawn_blocks, drawn_blocks
    assert "Machine translation" in drawn_blocks, drawn_blocks
    label = LANGUAGES["ta"].machine_translation_label
    assert not any(label in segments for segments in drawn_segments), drawn_segments


# ---------------------------------------------------------------------------
# Pipeline with fakes
# ---------------------------------------------------------------------------
class _RecordingLanguageEngine(tp.FakeTranslationEngine):
    """Records the (src, tgt) FLORES codes of every translation call."""

    def __init__(self):
        super().__init__()
        self.targets: list[tuple[str, str]] = []

    def translate_paragraphs(self, paragraphs, src_lang, tgt_lang):
        self.targets.append((src_lang, tgt_lang))
        return super().translate_paragraphs(paragraphs, src_lang, tgt_lang)


def check_pipeline_ta(tmp: Path) -> None:
    engine = _RecordingLanguageEngine()
    editor_provider = tp.RecordingEditorProvider(tp.FakeEditorProvider())
    out_dir = tmp / "pipeline_ta"
    cfg = BookConfig(
        epub_path="data/sherlock_holmes.epub",
        output_dir=str(out_dir),
        glossary_db=str(out_dir / "glossary.db"),
        target_language="ta",
        provider_config="config/consistency_editor.example.yaml",
    )
    lines: list[str] = []
    with tp._patch_module(
        pipeline_module,
        build_translation_provider=lambda config, ct2_model_dir=None, language_name="Kannada": engine,
        build_provider=lambda provider_cfg: editor_provider,
    ):
        result = run_book(
            cfg,
            resolve_path=tp._resolve,
            options=RunOptions(limit_chapters=["item4"]),
            progress=lines.append,
        )

    assert ("eng_Latn", "tam_Taml") in engine.targets, engine.targets
    assert all(tgt == "tam_Taml" for _src, tgt in engine.targets), engine.targets
    assert editor_provider.system_prompts, "editor was never asked to edit"
    assert "Tamil" in editor_provider.system_prompts[0], editor_provider.system_prompts[0]
    assert "Kannada" not in editor_provider.system_prompts[0]

    assert result.epub_path is not None
    assert result.epub_path.name == "sherlock_holmes.ta.epub", result.epub_path
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["target_language"] == "ta", manifest.get("target_language")
    assert (
        "No bundled font for Tamil; readers will use their own fonts." in lines
    ), lines


def main() -> None:
    check_language_table()
    check_prompts()
    check_output_file_names()
    check_cover()
    tmp = Path(tempfile.mkdtemp())
    try:
        check_writer(tmp)
        check_pipeline_ta(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("test_languages: all assertions passed")


if __name__ == "__main__":
    main()
