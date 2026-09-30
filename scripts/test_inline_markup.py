"""Offline tests for FR-1.3 inline-markup markers.

Covers the pure helpers in ``kannada_epub.inline_markup`` (strip/parse) and the
reader's ``Paragraph.marked_text`` extraction, then proves the hard requirement
on real books: removing every marker from ``marked_text`` reproduces ``text``.

Run: .venv/bin/python scripts/test_inline_markup.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bs4 import BeautifulSoup
from epub_fixture import build_epub

from kannada_epub.epub_io import (
    BLOCK_TAGS,
    _marked_elements,
    _marked_text,
    _spine_documents,
    load_epub_chapters,
)
from kannada_epub.inline_markup import (
    MARKER_RE,
    MarkerSpan,
    parse_markers,
    strip_markers,
)

ROOT = Path(__file__).resolve().parent.parent


def _raw_doc(inner: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE html>\n'
        '<html xmlns="http://www.w3.org/1999/xhtml">'
        "<head><title>Book title</title></head>"
        f"<body>{inner}</body></html>"
    )


def check_strip_markers() -> None:
    # Whitespace collapsing and marker removal.
    assert strip_markers("") == ""
    assert strip_markers("  a\t b \n c ") == "a b c"
    assert strip_markers("He read ⟦1⟧The⟦/1⟧ Times.") == "He read The Times."
    assert strip_markers("⟦1⟧⟦2⟧x⟦/2⟧⟦/1⟧") == "x"
    # The marker regex matches both forms and nothing else.
    assert [m.group(0) for m in MARKER_RE.finditer("⟦1⟧a⟦/2⟧")] == ["⟦1⟧", "⟦/2⟧"]


def check_parse_markers() -> None:
    # Valid tree: text and one span.
    assert parse_markers("a⟦1⟧b⟦/1⟧c", {1}) == [
        "a",
        MarkerSpan(1, ["b"]),
        "c",
    ]
    # Nested spans.
    assert parse_markers("⟦1⟧a⟦2⟧b⟦/2⟧c⟦/1⟧", {1, 2}) == [
        MarkerSpan(1, ["a", MarkerSpan(2, ["b"]), "c"]),
    ]
    # Two sibling spans with text between.
    assert parse_markers("⟦1⟧a⟦/1⟧ and ⟦2⟧b⟦/2⟧", {1, 2}) == [
        MarkerSpan(1, ["a"]),
        " and ",
        MarkerSpan(2, ["b"]),
    ]
    # A span present in the source but missing from the translation is allowed.
    assert parse_markers("a⟦2⟧b⟦/2⟧c", {1, 2}) == [
        "a",
        MarkerSpan(2, ["b"]),
        "c",
    ]
    # No markers at all parses to plain text segments.
    assert parse_markers("plain text", {1, 2}) == ["plain text"]

    # Malformed cases all return None.
    assert parse_markers("a⟦3⟧b⟦/3⟧", {1, 2}) is None  # unknown id
    assert parse_markers("⟦1⟧a⟦1⟧b⟦/1⟧⟦/1⟧", {1}) is None  # opened twice
    assert parse_markers("⟦1⟧a⟦/1⟧⟦1⟧b⟦/1⟧", {1}) is None  # reused id
    assert parse_markers("a⟦/1⟧b", {1}) is None  # close without open
    assert parse_markers("⟦1⟧a⟦2⟧b⟦/1⟧c⟦/2⟧", {1, 2}) is None  # crossing
    assert parse_markers("a⟦1⟧b", {1}) is None  # unclosed
    assert parse_markers("⟦1⟧a⟦2⟧b⟦/2⟧", {1, 2}) is None  # outer unclosed
    print("  strip/parse helpers ok")


def check_extraction() -> None:
    tmpdir = Path(tempfile.mkdtemp())
    try:
        book = tmpdir / "markup.epub"
        build_epub(
            book,
            [
                {
                    "id": "c1",
                    "href": "c1.xhtml",
                    "content": _raw_doc(
                        '<p>He read <i>The <b>Times</b></i> and a '
                        '<a href="#fn1" id="r1">note</a>.</p>'
                        "<p>Plain text only.</p>"
                        "<p>Whitespace<i>   </i>inside.</p>"
                    ),
                }
            ],
        )
        chapter = load_epub_chapters(book)[0]
        marked, plain, whitespace = chapter.paragraphs

        assert marked.text == "He read The Times and a note.", marked.text
        assert marked.marked_text == (
            "He read ⟦1⟧The ⟦2⟧Times⟦/2⟧⟦/1⟧ and a ⟦3⟧note⟦/3⟧."
        ), marked.marked_text
        assert strip_markers(marked.marked_text) == marked.text
        # An inline element with only whitespace is not marked, so the block
        # has no markup at all.
        assert plain.marked_text is None, plain.marked_text
        assert whitespace.text == "Whitespace inside.", whitespace.text
        assert whitespace.marked_text is None, whitespace.marked_text

        # The public-ish helpers agree with the dataclass.
        soup = BeautifulSoup(_raw_doc("<p>Plain.</p>"), "lxml")
        block = soup.find("p")
        assert _marked_elements(block) == []
        assert _marked_text(block, "Plain.") is None
        print("  extraction fixture ok")
    finally:
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)


def _fallback_blocks(path: Path) -> int:
    """Blocks whose inline markup could not be represented and fell back to None."""
    documents, _labels, _ws = _spine_documents(path)
    fallbacks = 0
    for _id, _name, content in documents:
        soup = BeautifulSoup(content, "lxml")
        for tag in soup.find_all(BLOCK_TAGS):
            text = " ".join(tag.get_text().split())
            if _marked_elements(tag) and _marked_text(tag, text) is None:
                fallbacks += 1
    return fallbacks


def check_real_books() -> None:
    """Every marked_text must strip back to text exactly; report the counts."""
    books = [ROOT / "data" / "sherlock_holmes.epub"]
    books += sorted((ROOT / ".omc" / "pg").glob("*.epub"))
    for book in books:
        if not book.exists():
            continue
        with_markup = 0
        total = 0
        for chapter in load_epub_chapters(book):
            for paragraph in chapter.paragraphs:
                total += 1
                if paragraph.marked_text is not None:
                    with_markup += 1
                    assert strip_markers(paragraph.marked_text) == paragraph.text, (
                        f"{book.name}: {paragraph.text!r} != "
                        f"strip({paragraph.marked_text!r})"
                    )
        print(
            f"  {book.name}: {with_markup}/{total} paragraphs have markup; "
            f"{_fallback_blocks(book)} block(s) fell back to None"
        )


def main() -> None:
    check_strip_markers()
    check_parse_markers()
    check_extraction()
    check_real_books()
    print("test_inline_markup: all assertions passed")


if __name__ == "__main__":
    main()
