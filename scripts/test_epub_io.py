"""Offline tests for the zipfile/lxml EPUB reader in `kannada_epub.epub_io`.

Covers the spine rules the reader must keep (they match the EbookLib-based
reader it replaced, so existing checkpoints stay valid): linear="no" and
non-XHTML items are skipped, hrefs are URL-unquoted, the nav is skipped, a
missing spine document raises, entities are not expanded, and metadata is
read from the package document. No model or network needed.

Run: .venv/bin/python scripts/test_epub_io.py
"""

import logging
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from epub_fixture import build_epub

from kannada_epub.epub_io import load_epub_chapters, read_epub_metadata


def _doc(heading: str, body: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE html>\n'
        '<html xmlns="http://www.w3.org/1999/xhtml">'
        f"<head><title>Book title</title></head>"
        f"<body><h1>{heading}</h1>{body}</body></html>"
    )


class _RecordingHandler(logging.Handler):
    """Collect log records instead of printing them (plain-script test)."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def main() -> None:
    tmpdir = Path(tempfile.mkdtemp())
    try:
        src = tmpdir / "book.epub"
        build_epub(
            src,
            [
                {"id": "c1", "href": "c1.xhtml", "content": _doc("One", "<p>Alpha.</p>")},
                {
                    "id": "aside",
                    "href": "aside.xhtml",
                    "content": _doc("Aside", "<p>Skipped.</p>"),
                    "linear": "no",
                },
                {
                    "id": "c2",
                    "href": "chapter%20two.xhtml",
                    "zip_name": "chapter two.xhtml",
                    "content": _doc("Two", "<blockquote><p>Beta.</p></blockquote><p>Gamma.</p>"),
                },
                {
                    "id": "style",
                    "href": "style.css",
                    "content": "p { margin: 0 }",
                    "media_type": "text/css",
                },
                {
                    "id": "unlisted",
                    "href": "unlisted.xhtml",
                    "content": _doc("Unlisted", "<p>Not in spine.</p>"),
                    "in_spine": False,
                },
                {
                    "id": "cover-page",
                    "href": "cover.xhtml",
                    "content": _doc("Cover", "<p>Cover text.</p>"),
                    "properties": "cover",
                },
            ],
            title="  The Title  ",
            creator="An Author",
        )

        chapters = load_epub_chapters(src)
        # Spine order; nav, non-linear, non-XHTML, unlisted and "cover"
        # documents are all skipped.
        assert [c.id for c in chapters] == ["c1", "c2"], [c.id for c in chapters]
        # The chapter title comes from the body, not the head's <title>.
        assert [c.title for c in chapters] == ["One", "Two"]
        # An URL-quoted href resolves to the real zip entry; the blockquote
        # wrapping a <p> is skipped but still advances the index.
        c2 = chapters[1]
        assert [(p.index, p.text) for p in c2.paragraphs] == [
            (0, "Two"),
            (2, "Beta."),
            (3, "Gamma."),
        ], c2.paragraphs
        assert read_epub_metadata(src) == ("The Title", "An Author")
        assert [c.id for c in load_epub_chapters(src, exclude_ids=["c1"])] == ["c2"]

        pg = tmpdir / "gutenberg.epub"
        build_epub(
            pg,
            [
                {
                    "id": "item14",
                    "href": "item14.xhtml",
                    "content": _doc(
                        "Story",
                        '<p>Before.</p><div class="pg-boilerplate extra">'
                        '<p>License text.</p></div>'
                        '<p class="not-pg-boilerplate">Ordinary text.</p>'
                        '<footer id="pg-footer"><p>Footer license.</p></footer>'
                        '<p>After.</p>',
                    ),
                },
                {
                    "id": "pg-header",
                    "href": "single.xhtml",
                    "content": _doc(
                        "Single",
                        '<header id="pg-header" class="pg-boilerplate"><p>Header.</p>'
                        '</header><p>Actual book paragraph one.</p>'
                        '<p>Actual book paragraph two.</p>',
                    ),
                },
            ],
        )
        skipped = {chapter.id: chapter for chapter in load_epub_chapters(pg)}
        assert [(p.index, p.text) for p in skipped["item14"].paragraphs] == [
            (0, "Story"),
            (1, "Before."),
            (3, "Ordinary text."),
            (5, "After."),
        ]
        assert [p.text for p in skipped["pg-header"].paragraphs] == [
            "Single",
            "Actual book paragraph one.",
            "Actual book paragraph two.",
        ]
        restored = {chapter.id: chapter for chapter in load_epub_chapters(
            pg, skip_gutenberg_boilerplate=False
        )}
        assert restored["item14"].paragraphs[2].text == "License text."
        assert restored["item14"].paragraphs[4].text == "Footer license."
        assert restored["item14"].paragraphs[3].text == "Ordinary text."

        # Missing title/creator come back as None.
        bare = tmpdir / "bare.epub"
        build_epub(
            bare,
            [{"id": "c1", "href": "c1.xhtml", "content": _doc("One", "<p>A.</p>")}],
            title=None,
            creator=None,
            opf_dir="",
        )
        assert read_epub_metadata(bare) == (None, None)
        assert [c.id for c in load_epub_chapters(bare)] == ["c1"]

        # A spine that lists the same idref twice yields the document once, in
        # the position of its first occurrence, and logs one warning naming the
        # skipped idref. A document first listed linear="no" and later linear
        # is returned (once), because duplicates are checked after filtering.
        dupes = tmpdir / "duplicates.epub"
        build_epub(
            dupes,
            [
                {"id": "c1", "href": "c1.xhtml", "content": _doc("One", "<p>Alpha.</p>")},
                {"id": "c2", "href": "c2.xhtml", "content": _doc("Two", "<p>Beta.</p>")},
                {
                    "id": "aside",
                    "href": "aside.xhtml",
                    "content": _doc("Aside", "<p>Aside.</p>"),
                    "linear": "no",
                },
            ],
            extra_spine=["c1", "aside", "c2"],
        )
        epub_io_logger = logging.getLogger("kannada_epub.epub_io")
        handler = _RecordingHandler()
        epub_io_logger.addHandler(handler)
        try:
            dup_chapters = load_epub_chapters(dupes)
        finally:
            epub_io_logger.removeHandler(handler)
        assert [c.id for c in dup_chapters] == ["c1", "c2", "aside"], [
            c.id for c in dup_chapters
        ]
        assert [p.text for p in dup_chapters[0].paragraphs] == ["One", "Alpha."]
        assert len(handler.records) == 1, [r.getMessage() for r in handler.records]
        message = handler.records[0].getMessage()
        assert handler.records[0].levelno == logging.WARNING, message
        assert "c1" in message and "c2" in message, message
        assert "aside" not in message, message

        # A spine document missing from the zip is an error, not a silent skip.
        missing = tmpdir / "missing.epub"
        build_epub(
            missing,
            [{"id": "c1", "href": "c1.xhtml", "content": "", "write": False}],
        )
        try:
            load_epub_chapters(missing)
        except ValueError as exc:
            assert "missing entry" in str(exc), exc
        else:
            raise AssertionError("missing spine document did not raise")

        # Entities in the package document are not expanded.
        entity = tmpdir / "entity.epub"
        build_epub(
            entity,
            [{"id": "c1", "href": "c1.xhtml", "content": _doc("One", "<p>A.</p>")}],
            title="&xxe;",
        )
        with zipfile.ZipFile(entity) as zf:
            entries = {n: zf.read(n) for n in zf.namelist()}
        opf = entries["EPUB/content.opf"].decode().replace("&amp;xxe;", "&xxe;")
        opf = opf.replace(
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<!DOCTYPE package [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>',
        )
        entries["EPUB/content.opf"] = opf.encode()
        with zipfile.ZipFile(entity, "w") as zf:
            for name, data in entries.items():
                zf.writestr(name, data)
        title, _ = read_epub_metadata(entity)
        assert title is None or "root:" not in title, title
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print("test_epub_io: all assertions passed")


if __name__ == "__main__":
    main()
