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

from epub_fixture import build_epub, build_ncx

from kannada_epub.epub_io import load_epub_chapters, read_epub_metadata


def _doc(heading: str, body: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE html>\n'
        '<html xmlns="http://www.w3.org/1999/xhtml">'
        f"<head><title>Book title</title></head>"
        f"<body><h1>{heading}</h1>{body}</body></html>"
    )


def _raw_doc(inner: str) -> str:
    """A body whose blocks are given verbatim (no implicit heading)."""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE html>\n'
        '<html xmlns="http://www.w3.org/1999/xhtml">'
        f"<head><title>Book title</title></head>"
        f"<body>{inner}</body></html>"
    )


def _nav(entries: list[tuple[str, str]]) -> str:
    """An EPUB 3 nav document with one ``<a>`` per ``(label, href)``."""
    items = "".join(
        f'<li><a href="{href}">{label}</a></li>' for label, href in entries
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<html xmlns="http://www.w3.org/1999/xhtml" '
        'xmlns:epub="http://www.idpf.org/2007/ops">'
        "<head><title>Contents</title></head>"
        f'<body><nav epub:type="toc"><ol>{items}</ol></nav></body></html>'
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

        # Paragraph kind: h1-h6 blocks (and a block nested inside one) are
        # "heading"; a <p> inside a heading-less <div> stays "text"; and the
        # table rule still wins over the heading rule. Classification never
        # changes which blocks are extracted or their indices, so the index
        # list is exactly the enumerate() positions of the fixture.
        kinds = tmpdir / "kinds.epub"
        build_epub(
            kinds,
            [
                {
                    "id": "k1",
                    "href": "k1.xhtml",
                    "content": _raw_doc(
                        "<h1>Top</h1><p>Body one.</p><div><p>Body two.</p></div>"
                        "<h2>Sub</h2><h3>Sub sub</h3><p>Body three.</p>"
                        "<table><tr><th>Head cell</th><td>Data cell</td></tr></table>"
                    ),
                }
            ],
        )
        k1 = load_epub_chapters(kinds)[0]
        assert [p.index for p in k1.paragraphs] == list(range(8)), k1.paragraphs
        assert [(p.text, p.kind) for p in k1.paragraphs] == [
            ("Top", "heading"),
            ("Body one.", "text"),
            ("Body two.", "text"),
            ("Sub", "heading"),
            ("Sub sub", "heading"),
            ("Body three.", "text"),
            ("Head cell", "table_cell"),
            ("Data cell", "table_cell"),
        ], k1.paragraphs

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

        # Wikisource ws-export boilerplate. A licenseContainer banner inside a
        # content document is skipped block-by-block (all other block indices
        # unchanged), and the generator's title/about spine documents are
        # skipped as whole chapters when the book is detected as a ws-export.
        ws = tmpdir / "wikisource.epub"
        build_epub(
            ws,
            [
                {
                    "id": "title",
                    "href": "title.xhtml",
                    "content": _doc(
                        "Sultana's Dream",
                        "<h3>Rokeya Sakhawat Hossain</h3>"
                        "<h6>Exported from Wikisource on January 1, 2026</h6>",
                    ),
                },
                {
                    "id": "c1",
                    "href": "c1.xhtml",
                    "content": _raw_doc(
                        "<p>Before.</p>"
                        '<div class="licenseContainer licenseBanner '
                        'dynlayout-exempt">'
                        "<div><p>This work is in the public domain in the "
                        "United States.</p><p>More licence text.</p></div>"
                        "<div><p>Yet more licence text.</p></div></div>"
                        "<p>After.</p>"
                    ),
                },
                {
                    "id": "about",
                    "href": "about.xhtml",
                    "content": _doc("About", "<p>About this digital edition.</p>"),
                },
            ],
            extra_metadata=[
                '<dc:contributor id="meta-bkp">Wikisource</dc:contributor>'
            ],
        )
        ws_default = {c.id: c for c in load_epub_chapters(ws)}
        assert set(ws_default) == {"c1"}, sorted(ws_default)
        assert [p.text for p in ws_default["c1"].paragraphs] == [
            "Before.",
            "After.",
        ], ws_default["c1"].paragraphs
        ws_full = {
            c.id: c
            for c in load_epub_chapters(ws, skip_wikisource_boilerplate=False)
        }
        # With the skip off, title, about and every banner paragraph return.
        assert set(ws_full) == {"title", "c1", "about"}, sorted(ws_full)
        assert any(
            "public domain" in p.text for p in ws_full["c1"].paragraphs
        ), ws_full["c1"].paragraphs
        # Non-banner indices are unchanged: the default indices and texts are
        # exactly the full load minus the banner's indices.
        banner = {
            p.index
            for p in ws_full["c1"].paragraphs
            if "licence" in p.text or "public domain" in p.text
        }
        assert len(banner) == 3, banner
        assert {p.index: p.text for p in ws_default["c1"].paragraphs} == {
            p.index: p.text
            for p in ws_full["c1"].paragraphs
            if p.index not in banner
        }

        # The same ws-export detection fires on a wikisource.org identifier
        # with no Wikisource contributor.
        ws_id = tmpdir / "wikisource-identifier.epub"
        build_epub(
            ws_id,
            [
                {
                    "id": "title",
                    "href": "title.xhtml",
                    "content": _doc("T", "<p>T.</p>"),
                },
                {
                    "id": "about",
                    "href": "about.xhtml",
                    "content": _doc("A", "<p>A.</p>"),
                },
                {"id": "c1", "href": "c1.xhtml", "content": _doc("C", "<p>C.</p>")},
            ],
            creator=None,
            identifier="https://en.wikisource.org/wiki/X",
        )
        assert [c.id for c in load_epub_chapters(ws_id)] == ["c1"]

        # A non-Wikisource book keeps chapters whose ids happen to be
        # title/about, but still drops licenseContainer blocks (the class name
        # is Wikisource-specific, so the rule is not gated on detection).
        plain = tmpdir / "plain.epub"
        build_epub(
            plain,
            [
                {
                    "id": "title",
                    "href": "title.xhtml",
                    "content": _doc("Title", "<p>T.</p>"),
                },
                {
                    "id": "about",
                    "href": "about.xhtml",
                    "content": _doc("About", "<p>A.</p>"),
                },
                {
                    "id": "c1",
                    "href": "c1.xhtml",
                    "content": _raw_doc(
                        '<p>Kept.</p><div class="licenseContainer">'
                        "<p>Banner.</p></div>"
                    ),
                },
            ],
            identifier="plain-001",
        )
        plain_chapters = {c.id: c for c in load_epub_chapters(plain)}
        assert set(plain_chapters) == {"title", "about", "c1"}, sorted(
            plain_chapters
        )
        assert [p.text for p in plain_chapters["c1"].paragraphs] == ["Kept."]

        # Chapter title fallbacks. The nav label beats a lower heading; an h1
        # always wins; a document with no nav entry falls back to its h3; and
        # the nav href may carry a URL-encoded path and a #fragment.
        titles = tmpdir / "titles.epub"
        build_epub(
            titles,
            [
                {
                    "id": "t1",
                    "href": "t1.xhtml",
                    "content": _raw_doc("<h3>Heading Three</h3><p>A.</p>"),
                },
                {
                    "id": "t2",
                    "href": "chapter%20two.xhtml",
                    "zip_name": "chapter two.xhtml",
                    "content": _raw_doc("<p>B.</p>"),
                },
                {
                    "id": "t3",
                    "href": "t3.xhtml",
                    "content": _raw_doc("<h3>Heading Three Only</h3><p>C.</p>"),
                },
                {
                    "id": "t4",
                    "href": "t4.xhtml",
                    "content": _raw_doc("<h1>Real H1</h1><p>D.</p>"),
                },
            ],
            nav_content=_nav(
                [
                    ("Nav One", "t1.xhtml"),
                    ("Nav Two", "chapter%20two.xhtml#sec-1"),
                    ("Different", "t4.xhtml"),
                ]
            ),
        )
        title_map = {c.id: c.title for c in load_epub_chapters(titles)}
        assert title_map == {
            "t1": "Nav One",
            "t2": "Nav Two",
            "t3": "Heading Three Only",
            "t4": "Real H1",
        }, title_map

        # An EPUB 2-style book has no nav: the NCX label is used.
        ncx_book = tmpdir / "ncx.epub"
        build_epub(
            ncx_book,
            [{"id": "n1", "href": "n1.xhtml", "content": _raw_doc("<p>Only.</p>")}],
            include_nav=False,
            ncx_content=build_ncx([("NCX Label", "n1.xhtml#part1")]),
            version="2.0",
        )
        assert [c.title for c in load_epub_chapters(ncx_book)] == ["NCX Label"]

        # Oversized titles are capped at 200 characters on a word boundary.
        long_title = " ".join(["word"] * 60)
        long_book = tmpdir / "long.epub"
        build_epub(
            long_book,
            [
                {
                    "id": "l1",
                    "href": "l1.xhtml",
                    "content": _raw_doc(f"<h1>{long_title}</h1><p>x</p>"),
                }
            ],
        )
        truncated = load_epub_chapters(long_book)[0].title
        assert len(truncated) <= 200, len(truncated)
        assert truncated.split() == ["word"] * 40, truncated

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
