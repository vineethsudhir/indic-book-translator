"""Smoke test for table-cell translation (PRD FR-3.2).

Builds a small EPUB 3 fixture (heading, two paragraphs and a
table exercising caption/th/colspan/rowspan/align/empty-cell/nested-<p>),
translates every paragraph with fake Kannada, rewrites the book and proves the
cells round-trip in place with their structure and attributes intact. No model
or network needed.

Run: .venv/bin/python scripts/test_tables.py
"""

import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bs4 import BeautifulSoup
from epub_fixture import build_epub

from kannada_epub.epub_io import load_epub_chapters
from kannada_epub.epub_writer import write_translated_epub

CHAPTER_ID = "chapter1"

# A heading, two ordinary paragraphs, and a table whose cells cover every
# shape FR-3.2 needs: caption, a header row, colspan, rowspan+align, an empty
# cell, and a cell wrapping a block (<p>).
CHAPTER_XHTML = """<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>Chapter One</title></head>
<body>
  <h1>Chapter One</h1>
  <p>First paragraph.</p>
  <p>Second paragraph.</p>
  <table>
    <caption>Table caption</caption>
    <tr><th>Header A</th><th>Header B</th><th>Header C</th></tr>
    <tr><td colspan="2">Wide cell</td><td>Normal cell</td></tr>
    <tr><td rowspan="2" align="center">Tall cell</td><td>Row two cell</td><td></td></tr>
    <tr><td>Row three cell</td><td><p>Nested paragraph cell</p></td></tr>
  </table>
</body>
</html>
"""


def _build_fixture(path: Path) -> None:
    build_epub(
        path,
        [{"id": CHAPTER_ID, "href": "chap_01.xhtml", "content": CHAPTER_XHTML}],
        title="Tables Test",
    )


def _read_doc(epub_path: Path, suffix: str) -> bytes:
    with zipfile.ZipFile(epub_path) as zf:
        name = next(n for n in zf.namelist() if n.endswith(suffix))
        return zf.read(name)


def _table_snapshot(epub_path: Path) -> tuple[int, list[tuple]]:
    """(row count, per-cell (tag, colspan, rowspan, align)) for the chapter table."""
    soup = BeautifulSoup(_read_doc(epub_path, "chap_01.xhtml"), "lxml")
    table = soup.find("table")
    cells = [
        (cell.name, cell.get("colspan"), cell.get("rowspan"), cell.get("align"))
        for cell in table.find_all(["td", "th"])
    ]
    return len(table.find_all("tr")), cells


def main() -> None:
    tmpdir = Path(tempfile.mkdtemp())
    try:
        src = tmpdir / "fixture.epub"
        _build_fixture(src)

        chapter = {c.id: c for c in load_epub_chapters(src)}[CHAPTER_ID]
        paragraphs = chapter.paragraphs

        # --- extraction: cells are translatable, tagged units --------------
        by_text = {p.text: p for p in paragraphs}

        # Ordinary blocks and the caption stay "text"; the h1 is a heading.
        assert by_text["Chapter One"].kind == "heading"
        assert by_text["First paragraph."].kind == "text"
        assert by_text["Second paragraph."].kind == "text"
        assert by_text["Table caption"].kind == "text"

        # Every td/th with text is a table cell.
        for text in (
            "Header A",
            "Header B",
            "Header C",
            "Wide cell",
            "Normal cell",
            "Tall cell",
            "Row two cell",
            "Row three cell",
            "Nested paragraph cell",
        ):
            assert by_text[text].kind == "table_cell", text

        # The empty cell is dropped by min_paragraph_chars.
        assert all(p.text.strip() for p in paragraphs)
        cells = [p for p in paragraphs if p.kind == "table_cell"]
        assert len(cells) == 9, [p.text for p in cells]

        # <td><p>x</p></td> yields exactly one unit: the <p>, as a table cell.
        nested = [p for p in paragraphs if p.text == "Nested paragraph cell"]
        assert len(nested) == 1
        assert nested[0].kind == "table_cell"

        # --- round-trip: fake Kannada lands back in the same units ---------
        fake = [f"ಕನ್ನಡ ಪಠ್ಯ {i}" for i in range(len(paragraphs))]
        translations = {
            CHAPTER_ID: {p.index: text for p, text in zip(paragraphs, fake, strict=True)}
        }
        out = tmpdir / "fixture.kn.epub"
        write_translated_epub(src, translations, out)

        reloaded = {c.id: c for c in load_epub_chapters(out)}[CHAPTER_ID]
        assert [p.text for p in reloaded.paragraphs] == fake
        assert [p.kind for p in reloaded.paragraphs] == [p.kind for p in paragraphs]

        # Structure and attributes are untouched, on the same cells.
        assert _table_snapshot(out) == _table_snapshot(src)

        # --- stylesheet keeps expanded Kannada inside its cell -------------
        with zipfile.ZipFile(out) as zf:
            css_name = next(n for n in zf.namelist() if n.endswith("kannada.css"))
            css = zf.read(css_name).decode("utf-8")
        assert "overflow-wrap: anywhere" in css
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print("test_tables: all assertions passed")


if __name__ == "__main__":
    main()
