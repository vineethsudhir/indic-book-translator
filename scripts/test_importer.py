"""Tests for ``kannada_epub.importer`` (text/HTML/OCR -> EPUB importer).

Everything here is fast and offline: small inline strings, stdlib + bs4/lxml,
no models, no network, no Java. The real sources under ``.omc/showcase/books``
are git-ignored, so the cases that use them run only when those files exist and
print a skip note otherwise.

Run: .venv/bin/python scripts/test_importer.py
"""

import re
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from kannada_epub import importer  # noqa: E402
from kannada_epub.epub_check import check_source_epub  # noqa: E402
from kannada_epub.epub_io import load_epub_chapters, read_epub_metadata  # noqa: E402
from kannada_epub.importer import (  # noqa: E402
    ImportedChapter,
    build_epub,
    import_html,
    import_text,
)

BOOKS = ROOT / ".omc" / "showcase" / "books"


def paragraphs(result) -> list[str]:
    return [p for chapter in result.chapters for p in chapter.paragraphs]


def titles(result) -> list[str]:
    return [chapter.title for chapter in result.chapters]


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="importer-test-"))
    saved_words = importer._system_words_cache
    try:
        # --- plain text: blank-line paragraphs, collapsed whitespace ------
        plain = "First paragraph\nwrapped over two lines.\n\nSecond   one\nwith  spaces.\n"
        result = import_text(plain)
        assert paragraphs(result) == [
            "First paragraph wrapped over two lines.",
            "Second one with spaces.",
        ], paragraphs(result)
        assert titles(result) == ["Front matter"], titles(result)

        # --- plain text: no blank lines means one paragraph per line -------
        no_blanks = "Line one\nLine two\nLine three"
        assert paragraphs(import_text(no_blanks)) == [
            "Line one",
            "Line two",
            "Line three",
        ]

        # --- non-breaking spaces are normalised ---------------------------
        assert paragraphs(import_text("one\u00a0two\u00a0three")) == ["one two three"]

        # --- OCR: page-number lines are dropped ---------------------------
        ocr_pages = "Hello there.\n\n12\n\nxiv\n\nl2\n\nGoodbye now.\n"
        result = import_text(ocr_pages, ocr=True)
        assert paragraphs(result) == ["Hello there.", "Goodbye now."], paragraphs(result)

        # --- OCR: running headers repeat and are dropped ------------------
        running = (
            "RUNNING HEAD\n\nBody one.\n\n"
            "RUNNING HEAD\n\nBody two.\n\n"
            "RUNNING HEAD\n\nBody three.\n"
        )
        result = import_text(running, ocr=True)
        # The first occurrence is kept (it doubles as a chapter title page);
        # the other two are dropped.
        assert paragraphs(result).count("RUNNING HEAD") == 1, paragraphs(result)
        assert any("running page header" in w for w in result.warnings), result.warnings
        assert any(
            "Dropped 2 running page header" in w for w in result.warnings
        ), result.warnings

        # --- OCR: hyphen joining needs the word to be known ---------------
        hyphen = "The re-\ncent years were\n\nquite hard."
        importer._system_words_cache = set()
        without = import_text(hyphen, ocr=True)
        assert paragraphs(without) == ["The re-cent years were quite hard."], without
        importer._system_words_cache = {"recent"}
        with_words = import_text(hyphen, ocr=True)
        assert paragraphs(with_words) == ["The recent years were quite hard."], with_words

        # Book vocabulary makes a joined word known even without /usr words.
        importer._system_words_cache = set()
        in_book = "The re-\ncent word.\n\nA recent arrival."
        assert paragraphs(import_text(in_book, ocr=True)) == [
            "The recent word.",
            "A recent arrival.",
        ]

        # --- OCR: paragraph split by a page break is rejoined -------------
        page_break = "The long walk home\n\n42\n\ncontinued past the mill."
        assert paragraphs(import_text(page_break, ocr=True)) == [
            "The long walk home continued past the mill."
        ]

        # A paragraph that already ends a sentence is not rejoined.
        no_rejoin = "The walk ended here.\n\n42\n\nA new paragraph begins."
        assert paragraphs(import_text(no_rejoin, ocr=True)) == [
            "The walk ended here.",
            "A new paragraph begins.",
        ]

        # --- OCR: quote junk and space-before-punctuation are fixed -------
        junk = "He said ''Hello'' , then left .\n"
        assert paragraphs(import_text(junk, ocr=True)) == [
            'He said "Hello", then left.'
        ]

        # --- OCR: punctuation/digit/one-char paragraphs are skipped --------
        skipped = "Hello there.\n\n--\n\nx\n\n12345\n\nWorld here.\n"
        assert paragraphs(import_text(skipped, ocr=True)) == [
            "Hello there.",
            "World here.",
        ]

        # --- OCR: leading front-matter garbage is skipped ------------------
        garbage = ";oo\n\n\u25a0CD\n\n^*  ^^\n\nThe real start of the book.\n"
        assert paragraphs(import_text(garbage, ocr=True)) == [
            "The real start of the book."
        ]

        # --- contents list: detected, then matched after the list ----------
        contents = (
            "CONTENTS\n\nPAGE\n\n"
            "Alpha Story 1\n\nBeta Story 15\n\nGamma Story 29\n\n"
            "Alpha Story\n\nThe alpha body.\n\n"
            "Beta Story\n\nThe beta body.\n\n"
            "Gamma Story\n\nThe gamma body.\n"
        )
        result = import_text(contents)
        assert result.detected_headings == [
            "Alpha Story",
            "Beta Story",
            "Gamma Story",
        ], result.detected_headings
        assert titles(result) == [
            "Front matter",
            "Alpha Story",
            "Beta Story",
            "Gamma Story",
        ], titles(result)
        assert result.chapters[1].paragraphs == ["The alpha body."]
        assert result.warnings == [], result.warnings

        # --- caller headings take precedence, contents still detected ------
        result = import_text(contents, headings=["Beta Story"])
        assert titles(result) == ["Front matter", "Beta Story"], titles(result)
        assert result.detected_headings == [
            "Alpha Story",
            "Beta Story",
            "Gamma Story",
        ]

        # --- a missing heading is warned about ----------------------------
        result = import_text("Alpha Story\n\nbody.\n", headings=["Ghost Story"])
        assert any("Ghost Story" in w and "not found" in w for w in result.warnings)

        # --- pattern headings: CHAPTER with run-together paragraph ---------
        long_chapter = (
            "CHAPTER I. The morning was bright and early when the travellers "
            "set out across the wide and dusty plain, past the tall trees, and "
            "into the distant blue hills beyond the river.\n\nThe rest of the "
            "chapter body lives here.\n"
        )
        result = import_text(long_chapter)
        assert titles(result) == ["CHAPTER I."], titles(result)
        assert result.chapters[0].paragraphs[0].startswith("The morning was bright")

        # --- pattern headings: a short heading stays whole -----------------
        result = import_text("BOOK 2\n\nThe second book begins.\n")
        assert titles(result) == ["BOOK 2"], titles(result)

        # --- pattern headings: roman numerals match exactly ----------------
        roman = "In the beginning\n\nI\n\nThe first body.\n\nII\n\nThe second body.\n"
        result = import_text(roman)
        assert titles(result) == ["Front matter", "I", "II"], titles(result)
        assert result.chapters[0].paragraphs == ["In the beginning"]

        # --- suspicious chapter openings are reported, not fixed -----------
        suspicious = "CHAPTER I.\n\nGEZHE whole night was dark.\n\nA second paragraph.\n"
        result = import_text(suspicious)
        assert titles(result) == ["CHAPTER I."], titles(result)
        assert any(
            'starts with "GEZHE whole night' in w and "check the first word" in w
            for w in result.warnings
        ), result.warnings
        known_start = import_text("IT was a dark night.\n\nIT returned again.")
        assert not any("check the first word" in w for w in known_start.warnings)

        # --- HTML: <p> blocks ---------------------------------------------
        html_p = (
            "<html><body><div>"
            "<p>First para.</p><p>Second para.</p>"
            "</div></body></html>"
        )
        assert paragraphs(import_html(html_p)) == ["First para.", "Second para."]

        # --- HTML: <br>-separated paragraphs ------------------------------
        html_br = (
            "<html><body><span property=\"sioc:content\">"
            "Line one.<br /><br />Line two.<br /><br />Line three."
            "</span></body></html>"
        )
        assert paragraphs(import_html(html_br)) == [
            "Line one.",
            "Line two.",
            "Line three.",
        ]

        # --- HTML: hard-wrapped lines are joined per block ----------------
        wrapped = (
            "The quick brown fox jumps over the lazy dog and keeps running on"
            "<br />"
            "and on across the field until it reaches the far fence at last"
            "<br /><br />"
            "A second paragraph also runs on for quite a while without stopping"
            "<br />"
            "and continues further still before it finally comes to a close now"
        )
        result = import_html(f"<html><body><div>{wrapped}</div></body></html>")
        assert len(paragraphs(result)) == 2, paragraphs(result)
        assert paragraphs(result)[0].startswith("The quick brown fox")
        assert paragraphs(result)[1].startswith("A second paragraph")

        # --- HTML: sioc:content is preferred over a denser decoy ----------
        decoy = "Decoy text " * 50
        html_scalar = (
            "<html><body>"
            f"<div id='decoy'>{decoy}</div>"
            "<span property=\"sioc:content\">Real one.<br /><br />Real two.</span>"
            "</body></html>"
        )
        assert paragraphs(import_html(html_scalar)) == ["Real one.", "Real two."]

        # --- build_epub: round trip, metadata, escaping, checks ------------
        chapters = [
            ImportedChapter("Chapter One", ["Hello world.", "Second paragraph."]),
            ImportedChapter("Chapter Two", ["Third & <fourth> clause."]),
        ]
        out = tmp / "round-trip.epub"
        build_epub(
            out,
            title="A Title & More",
            author="An Author",
            date="1900",
            source="https://example.org/book",
            chapters=chapters,
        )
        assert check_source_epub(out) == [], check_source_epub(out)
        assert read_epub_metadata(out) == ("A Title & More", "An Author")
        loaded = load_epub_chapters(out)
        assert [c.title for c in loaded] == ["Chapter One", "Chapter Two"]
        loaded_paragraphs = [p.text for c in loaded for p in c.paragraphs]
        assert "Hello world." in loaded_paragraphs
        assert "Second paragraph." in loaded_paragraphs
        assert "Third & <fourth> clause." in loaded_paragraphs

        # Optional metadata is omitted when not supplied.
        bare = tmp / "bare.epub"
        build_epub(
            bare,
            title="Bare",
            author=None,
            date=None,
            source=None,
            chapters=[ImportedChapter("Only", ["Body."])],
        )
        assert read_epub_metadata(bare) == ("Bare", None)
        assert check_source_epub(bare) == []

        # --- real sources: only when the git-ignored files exist -----------
        real_sources = [
            ("dance_of_siva", "siva.txt", True),
            ("cradle", "cradle_ocr.txt", True),
            ("kamala", "kamala.html", False),
            ("saguna", "saguna.html", False),
        ]
        ran_real = False
        for name, filename, ocr in real_sources:
            source = BOOKS / filename
            if not source.exists():
                continue
            ran_real = True
            text = source.read_text(encoding="utf-8", errors="replace")
            result = import_text(text, ocr=True) if ocr else import_html(text)
            assert result.chapters, name
            assert all(
                not re.fullmatch(r"[\divxlIl]{1,4}", p) for p in paragraphs(result)
            ), f"{name} has a page-number-only paragraph"
            real_out = tmp / f"{name}.epub"
            build_epub(
                real_out,
                title=name,
                author=None,
                date=None,
                source=None,
                chapters=result.chapters,
            )
            assert check_source_epub(real_out) == [], check_source_epub(real_out)
        if not ran_real:
            print("test_importer: real sources not present, skipping those cases")
    finally:
        importer._system_words_cache = saved_words
        shutil.rmtree(tmp, ignore_errors=True)

    print("test_importer: all assertions passed")


if __name__ == "__main__":
    main()
