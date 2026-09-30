"""Smoke test for the EPUB writer (PRD FR-1.3 / FR-5): replace translated
paragraphs in place, repackage, and prove everything else survived. Uses FAKE
translations and the real Sherlock Holmes EPUB — no model or network needed.

Run: .venv/bin/python scripts/test_epub_writer.py
"""

import posixpath
import shutil
import sys
import tempfile
import uuid
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import lxml.etree as ET
from bs4 import BeautifulSoup
from epub_fixture import build_epub

from kannada_epub.epub_io import BLOCK_TAGS, load_epub_chapters
from kannada_epub.epub_writer import (
    MACHINE_TRANSLATION_CONTRIBUTOR,
    _serialize_xhtml,
    _translate_document,
    find_gutenberg_mentions,
    translations_from_batches,
    write_translated_epub,
)

ROOT = Path(__file__).resolve().parent.parent
EPUB = ROOT / "data" / "sherlock_holmes.epub"

OPF_NS = "http://www.idpf.org/2007/opf"
DC_NS = "http://purl.org/dc/elements/1.1/"
CONTAINER_NS = "urn:oasis:names:tc:opendocument:xmlns:container"


def _opf_path(zf: zipfile.ZipFile) -> str:
    root = ET.fromstring(zf.read("META-INF/container.xml"))
    rootfile = root.find(f".//{{{CONTAINER_NS}}}rootfile")
    return rootfile.get("full-path")


def _manifest(zf: zipfile.ZipFile, opf_path: str) -> dict[str, str]:
    root = ET.fromstring(zf.read(opf_path))
    return {
        item.get("id"): item.get("href")
        for item in root.findall(f".//{{{OPF_NS}}}manifest/{{{OPF_NS}}}item")
    }


def _block_tags(data: bytes) -> list[str]:
    """The ``BLOCK_TAGS`` tag-name sequence of a document."""
    return [tag.name for tag in BeautifulSoup(data, "lxml").find_all(BLOCK_TAGS)]


def _chapter_indices(path, **kwargs) -> list[tuple[str, list[int]]]:
    """``(chapter id, [Paragraph.index ...])`` for a book."""
    return [
        (chapter.id, [p.index for p in chapter.paragraphs])
        for chapter in load_epub_chapters(path, **kwargs)
    ]


def _legacy_translate_document(
    content: bytes, translations_by_index: dict[int, str], css_href: str
) -> bytes:
    """The pre-FR-1.3 plain replacement path, for the byte-identical check."""
    soup = BeautifulSoup(content, "lxml")
    blocks = soup.find_all(BLOCK_TAGS)
    for index, text in sorted(translations_by_index.items()):
        element = blocks[index]
        preserved = [
            soup.new_tag(desc.name, attrs=dict(desc.attrs))
            for desc in element.find_all(attrs={"id": True})
        ]
        element.clear()
        for anchor in preserved:
            element.append(anchor)
        element.append(text)
    if soup.html is not None:
        soup.html["xmlns"] = "http://www.w3.org/1999/xhtml"
        soup.html["lang"] = "kn"
        soup.html["xml:lang"] = "kn"
    if soup.head is not None:
        soup.head.append(
            soup.new_tag("link", rel="stylesheet", type="text/css", href=css_href)
        )
    return _serialize_xhtml(content, soup)


def _check_inline_markup(tmpdir: Path) -> None:
    """FR-1.3: rebuild inline tags from markers, degrade safely otherwise."""
    content = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<html xmlns="http://www.w3.org/1999/xhtml"><head><title>t</title></head>'
        "<body>"
        '<p>He read <i>The <b>Times</b></i> and a '
        '<a href="#fn1" id="r1">note</a>.</p>'
        "<p>Plain paragraph.</p>"
        "</body></html>"
    ).encode("utf-8")
    source = tmpdir / "markup.epub"
    build_epub(source, [{"id": "c1", "href": "c1.xhtml", "content": content.decode()}])
    chapter = load_epub_chapters(source)[0]
    marked, plain = chapter.paragraphs
    assert marked.marked_text

    def doc_bytes(out: Path) -> bytes:
        with zipfile.ZipFile(out) as zf:
            return zf.read("EPUB/c1.xhtml")

    def block_of(data: bytes):
        return BeautifulSoup(data, "lxml").find_all(BLOCK_TAGS)[marked.index]

    # --- no markers: byte-identical to the old plain replacement ------------
    assert _translate_document(
        content, {marked.index: "ಸರಳ ಪಠ್ಯ.", plain.index: "ಇನ್ನೊಂದು."}, "css/kannada.css"
    ) == _legacy_translate_document(
        content, {marked.index: "ಸರಳ ಪಠ್ಯ.", plain.index: "ಇನ್ನೊಂದು."}, "css/kannada.css"
    )

    # --- valid marked translation: tags, attributes, nesting, id kept -------
    valid = (
        marked.marked_text.replace("The", "ದಿ")
        .replace("Times", "ಟೈಮ್ಸ್")
        .replace("note", "ಟಿಪ್ಪಣಿ")
    )
    valid_out = tmpdir / "valid.kn.epub"
    write_translated_epub(
        source, {"c1": {marked.index: valid, plain.index: "ಸರಳ."}}, valid_out
    )
    valid_data = doc_bytes(valid_out)
    valid_block = block_of(valid_data)
    assert [tag.name for tag in valid_block.find_all(["i", "b", "a"])] == [
        "i",
        "b",
        "a",
    ], valid_block
    assert valid_block.find("i").find("b") is not None, valid_block
    anchor = valid_block.find("a")
    assert anchor is not None and anchor.get("href") == "#fn1"
    assert anchor.get("id") == "r1"
    assert valid_block.get_text() == "He read ದಿ ಟೈಮ್ಸ್ and a ಟಿಪ್ಪಣಿ.", valid_block
    ids = [
        tag.get("id")
        for tag in BeautifulSoup(valid_data, "lxml").find_all(attrs={"id": True})
    ]
    assert ids.count("r1") == 1, ids  # rebuilt, not also preserved empty

    # --- malformed markers: plain text, no marker characters ----------------
    malformed_out = tmpdir / "malformed.kn.epub"
    write_translated_epub(
        source,
        {"c1": {marked.index: "⟦9⟧ಮೋಸ⟦/9⟧ ಪಠ್ಯ", plain.index: "ಸರಳ."}},
        malformed_out,
    )
    malformed_data = doc_bytes(malformed_out)
    malformed_block = block_of(malformed_data)
    assert malformed_block.get_text() == "ಮೋಸ ಪಠ್ಯ", malformed_block
    # The id anchor survives as an empty copy, exactly as before.
    assert malformed_block.find(["i", "b"]) is None, malformed_block
    assert malformed_block.find("a").get_text() == "", malformed_block
    assert "⟦" not in malformed_data.decode() and "⟧" not in malformed_data.decode()

    # --- one missing span: the others are still rebuilt ---------------------
    missing = valid.replace("⟦2⟧", "").replace("⟦/2⟧", "")
    missing_out = tmpdir / "missing.kn.epub"
    write_translated_epub(
        source, {"c1": {marked.index: missing, plain.index: "ಸರಳ."}}, missing_out
    )
    missing_data = doc_bytes(missing_out)
    missing_block = block_of(missing_data)
    assert missing_block.find("i") is not None, missing_block
    assert missing_block.find("b") is None, missing_block
    assert missing_block.find("a") is not None, missing_block
    missing_ids = [
        tag.get("id")
        for tag in BeautifulSoup(missing_data, "lxml").find_all(attrs={"id": True})
    ]
    assert missing_ids.count("r1") == 1, missing_ids


def main() -> None:
    chapters = load_epub_chapters(EPUB)
    by_id = {c.id: c for c in chapters}
    item4 = by_id["item4"]

    # --- translations_from_batches mapping ---------------------------------
    n = 10
    fake = [f"ಕನ್ನಡ ಪರೀಕ್ಷೆ {i}" for i in range(n)]
    batches = [
        {
            "chapter_id": "item4",
            "chapter_title": item4.title,
            "paragraph_start": 0,
            "paragraph_end": n,
            "source_english": [p.text for p in item4.paragraphs[:n]],
            "draft_kannada": fake,
            "edited_kannada": fake,
            "edited_emotions": ["neutral"] * n,
            "prior_context_used": "",
        }
    ]
    translations = translations_from_batches(chapters, batches)
    assert len(translations["item4"]) == n
    for k in range(n):
        assert translations["item4"][item4.paragraphs[k].index] == fake[k]

    # A batch starting mid-chapter maps k -> Paragraph.index of start + k.
    mid = translations_from_batches(
        chapters,
        [{"chapter_id": "item4", "paragraph_start": 5, "edited_kannada": ["ಐದು", "ಆರು"]}],
    )
    assert mid["item4"][item4.paragraphs[5].index] == "ಐದು"
    assert mid["item4"][item4.paragraphs[6].index] == "ಆರು"

    # Out-of-range position -> ValueError.
    try:
        translations_from_batches(
            chapters,
            [
                {
                    "chapter_id": "item4",
                    "paragraph_start": len(item4.paragraphs),
                    "edited_kannada": ["ಎಲ್ಲವೂ ಮೀರಿದೆ"],
                }
            ],
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for an out-of-range position")

    # Unknown chapter -> ValueError.
    try:
        translations_from_batches(
            chapters, [{"chapter_id": "does-not-exist", "paragraph_start": 0, "edited_kannada": ["x"]}]
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for an unknown chapter")

    # --- write the translated epub -----------------------------------------
    tmpdir = Path(tempfile.mkdtemp())
    try:
        out = tmpdir / "sherlock_holmes.kn.epub"
        write_translated_epub(EPUB, translations, out)
        assert out.exists()

        with zipfile.ZipFile(EPUB) as zsrc, zipfile.ZipFile(out) as zout:
            src_names = zsrc.namelist()
            out_names = zout.namelist()
            assert set(src_names) <= set(out_names), set(src_names) - set(out_names)

            out_infos = zout.infolist()
            assert out_infos[0].filename == "mimetype"
            assert out_infos[0].compress_type == zipfile.ZIP_STORED

            opf_path = _opf_path(zsrc)
            doc_name = posixpath.normpath(
                posixpath.join(posixpath.dirname(opf_path), _manifest(zsrc, opf_path)["item4"])
            )
            modified = {opf_path, doc_name}

            # Everything we did not touch is byte-for-byte identical.
            for name in src_names:
                if name in modified:
                    continue
                assert zsrc.read(name) == zout.read(name), f"entry changed: {name}"

            # Added package files are present...
            for required in (
                "NotoSansKannada-Regular.ttf",
                "NotoSansKannada-Bold.ttf",
                "OFL.txt",
                "kannada.css",
            ):
                assert any(n.endswith(required) for n in out_names), required

            # ... and listed in the OPF manifest, with language switched to kn.
            opf_out = ET.fromstring(zout.read(opf_path))
            items = opf_out.findall(f".//{{{OPF_NS}}}manifest/{{{OPF_NS}}}item")
            out_hrefs = {item.get("href") for item in items}
            assert any(h.endswith("kannada.css") for h in out_hrefs)
            assert any(h.endswith("NotoSansKannada-Regular.ttf") for h in out_hrefs)
            assert any(h.endswith("NotoSansKannada-Bold.ttf") for h in out_hrefs)
            assert any(h.endswith("OFL.txt") for h in out_hrefs)
            languages = opf_out.findall(f".//{{{DC_NS}}}language")
            assert languages and all(lang.text == "kn" for lang in languages)
            contributors = [
                c.text for c in opf_out.findall(f".//{{{DC_NS}}}contributor")
            ]
            assert contributors.count(MACHINE_TRANSLATION_CONTRIBUTOR) == 1, contributors

            # Every modified XHTML document is well-formed XML.
            for name in modified:
                if name.endswith((".xhtml", ".html", ".htm")):
                    ET.fromstring(zout.read(name))

            # Every id in the source document survives in the output.
            src_soup = BeautifulSoup(zsrc.read(doc_name), "lxml")
            out_soup = BeautifulSoup(zout.read(doc_name), "lxml")
            src_ids = {tag["id"] for tag in src_soup.find_all(attrs={"id": True})}
            out_ids = {tag["id"] for tag in out_soup.find_all(attrs={"id": True})}
            assert src_ids <= out_ids, f"lost ids: {src_ids - out_ids}"

            # Language is switched and the Kannada stylesheet is linked last.
            assert out_soup.html.get("lang") == "kn"
            assert out_soup.html.get("xml:lang") == "kn"
            links = out_soup.head.find_all("link")
            assert links and links[-1].get("href", "").endswith("kannada.css")
            assert links[-1].get("rel") == ["stylesheet"]

        # --- re-load the OUTPUT through the ordinary loader -----------------
        reloaded = {c.id: c for c in load_epub_chapters(out)}
        item4_out = reloaded["item4"]
        assert len(item4_out.paragraphs) == len(item4.paragraphs)
        for k in range(n):
            assert item4_out.paragraphs[k].text == fake[k], (
                f"paragraph {k}: {item4_out.paragraphs[k].text!r} != {fake[k]!r}"
            )
        assert [p.text for p in item4_out.paragraphs[n:]] == [
            p.text for p in item4.paragraphs[n:]
        ]

        # --- optional Project Gutenberg cleanup ---------------------------
        pg_source = tmpdir / "gutenberg.epub"
        pg_output = tmpdir / "gutenberg.kn.epub"
        unique_text = "Project Gutenberg ebook identifier"
        build_epub(
            pg_source,
            [
                {
                    "id": "pg-header",
                    "href": "book.xhtml",
                    "content": (
                        '<?xml version="1.0" encoding="UTF-8"?>'
                        '<html xmlns="http://www.w3.org/1999/xhtml"><head><title>'
                        'The Project Gutenberg eBook of Example | Project Gutenberg</title>'
                        '<meta name="generator" content="Ebookmaker by Project Gutenberg"/>'
                        '</head><body>'
                        '<header id="pg-header" class="pg-boilerplate extra">'
                        '<h1 id="header-heading">Project Gutenberg Header</h1>'
                        '<p>License <a id="license-fragment">text</a></p>'
                        '<ul><li>Item <a id="list-fragment">one</a></li></ul>'
                        '<table><tr><td>Cell <a id="cell-fragment">x</a></td></tr></table>'
                        '<p>Unlinked <a id="unlinked-anchor">anchor</a></p>'
                        '</header>'
                        '<p class="not-pg-boilerplate">The actual book text. '
                        '<a href="#license-fragment">See note</a>. An '
                        '<a href="#list-fragment">item</a> and a '
                        '<a href="#cell-fragment">cell</a> note. An '
                        '<a href="https://www.gutenberg.org/ebooks/1">illustrated edition</a>'
                        ' exists.</p>'
                        '<footer id="pg-footer"><p id="footer-fragment">License.</p>'
                        '<div id="project-gutenberg-license"><p>Full license.</p></div>'
                        '</footer></body></html>'
                    ),
                },
                {
                    "id": "ncx",
                    "href": "toc.ncx",
                    "media_type": "application/x-dtbncx+xml",
                    "in_spine": False,
                    "content": (
                        '<?xml version="1.0" encoding="UTF-8"?>'
                        '<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/">'
                        '<head><meta name="dtb:uid" content="Project Gutenberg old uid"/>'
                        '<meta name="dtb:generator" content="Ebookmaker by Project Gutenberg"/>'
                        '</head><navMap><navPoint id="pg"><navLabel><text>'
                        'Project Gutenberg</text></navLabel></navPoint>'
                        '<navPoint id="story"><navLabel><text>Story</text></navLabel>'
                        '</navPoint></navMap></ncx>'
                    ),
                },
            ],
            identifier=unique_text,
            extra_metadata=[
                '<dc:source>Project Gutenberg source</dc:source>',
                '<dc:publisher>Project Gutenberg</dc:publisher>',
                '<meta id="pg-meta" property="belongs-to-collection">'
                'Project Gutenberg collection</meta>',
                '<meta refines="#pg-meta" property="title">Referenced metadata</meta>',
            ],
            nav_content=(
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<html xmlns="http://www.w3.org/1999/xhtml" '
                'xmlns:epub="http://www.idpf.org/2007/ops"><head><title>Contents</title>'
                '</head><body><nav epub:type="toc"><ol>'
                '<li><a href="book.xhtml#pg-header">Project Gutenberg</a></li>'
                '<li><a href="book.xhtml">Story</a></li></ol>'
                '<ol><li><a href="book.xhtml">Project Gutenberg extra</a></li></ol>'
                '</nav></body></html>'
            ),
        )
        assert find_gutenberg_mentions(pg_source)
        write_translated_epub(pg_source, {}, pg_output, strip_gutenberg=True)
        assert find_gutenberg_mentions(pg_output) == []
        expected_uid = "urn:uuid:" + str(uuid.uuid5(uuid.NAMESPACE_URL, unique_text))
        with zipfile.ZipFile(pg_output) as pg_zip:
            opf_root = ET.fromstring(pg_zip.read("EPUB/content.opf"))
            identifier = opf_root.find(f".//{{{DC_NS}}}identifier")
            assert identifier.text == expected_uid
            metadata_text = " ".join(opf_root.find(f"{{{OPF_NS}}}metadata").itertext())
            assert "Gutenberg" not in metadata_text
            assert opf_root.find('.//*[@id="pg-meta"]') is None
            book_soup = BeautifulSoup(pg_zip.read("EPUB/book.xhtml"), "lxml")
            assert book_soup.title.get_text() == "Example"
            assert book_soup.find(id="pg-header").get_text(strip=True) == ""
            assert book_soup.find(id="pg-footer").get_text(strip=True) == ""
            # An id on a kept block element stays on it (now empty); an id on
            # a removed inline element becomes an empty retained span.
            for kept_id, kept_tag in (
                ("header-heading", "h1"),
                ("footer-fragment", "p"),
            ):
                kept = book_soup.find(id=kept_id)
                assert kept is not None and kept.name == kept_tag, kept_id
                assert kept.get_text(strip=True) == "", kept_id
            # A kept container whose id names Gutenberg loses the id (nothing
            # links to it) but stays, so the block sequence is unchanged.
            assert book_soup.find(id="project-gutenberg-license") is None
            assert not book_soup.find_all(attrs={"data-kn-kept-pg-id": True})
            fragment = book_soup.find(id="license-fragment")
            assert fragment is not None and fragment.name == "span"
            assert fragment.get_text(strip=True) == "" and not fragment.attrs.keys() - {"id"}
            # A boilerplate <ul><li> list and a <table><tr><td> survive,
            # emptied but with their markup and linked ids intact.
            ul = book_soup.find("ul")
            assert ul is not None and ul.find("li") is not None
            assert ul.get_text(strip=True) == ""
            list_fragment = book_soup.find(id="list-fragment")
            assert list_fragment is not None and list_fragment.name == "span"
            assert list_fragment.find_parent("li") is not None
            table = book_soup.find("table")
            assert table is not None and table.find("td") is not None
            assert table.get_text(strip=True) == ""
            cell_fragment = book_soup.find(id="cell-fragment")
            assert cell_fragment is not None and cell_fragment.name == "span"
            assert cell_fragment.find_parent("td") is not None
            # An unlinked id on a removed inline element is dropped.
            assert book_soup.find(id="unlinked-anchor") is None
            assert book_soup.find("meta", attrs={"name": "generator"}) is None
            # Links to gutenberg.org are unwrapped; their text stays.
            assert "illustrated edition" in book_soup.get_text()
            assert not [a for a in book_soup.find_all("a") if "gutenberg" in a.get("href", "")]
            nav_soup = BeautifulSoup(pg_zip.read("EPUB/nav.xhtml"), "lxml")
            assert "Gutenberg" not in nav_soup.get_text()
            assert len(nav_soup.find("nav").find_all("ol", recursive=False)) == 1
            ncx_root = ET.fromstring(pg_zip.read("EPUB/toc.ncx"))
            assert ncx_root.find('.//*[@name="dtb:uid"]') is not None
            assert ncx_root.find('.//*[@name="dtb:uid"]').get("content") == expected_uid
            assert ncx_root.find('.//*[@id="pg"]') is None
            assert ncx_root.find('.//*[@id="story"]') is not None
            assert ncx_root.find('.//*[@name="dtb:generator"]') is None

        # Stripping must leave every document's BLOCK_TAGS sequence unchanged,
        # so output and source paragraph indices line up.
        with zipfile.ZipFile(pg_source) as pg_zip:
            src_book = pg_zip.read("EPUB/book.xhtml")
        with zipfile.ZipFile(pg_output) as pg_zip:
            out_book = pg_zip.read("EPUB/book.xhtml")
        assert _block_tags(src_book) == _block_tags(out_book), (
            _block_tags(src_book),
            _block_tags(out_book),
        )
        assert len(_block_tags(src_book)) == len(_block_tags(out_book))
        # With boilerplate blocks retained as empty shells, both loads (with
        # boilerplate skipped) agree on chapter ids and paragraph indices.
        assert _chapter_indices(
            pg_output, skip_gutenberg_boilerplate=True
        ) == _chapter_indices(pg_source, skip_gutenberg_boilerplate=True)

        # --- URL-encoded manifest hrefs resolve like the reader resolves them
        encoded_source = tmpdir / "encoded.epub"
        encoded_output = tmpdir / "encoded.kn.epub"
        build_epub(
            encoded_source,
            [
                {
                    "id": "c1",
                    "href": "chapter%20one.xhtml",
                    "zip_name": "chapter one.xhtml",
                    "content": (
                        '<?xml version="1.0" encoding="UTF-8"?>'
                        '<html xmlns="http://www.w3.org/1999/xhtml"><head><title>t</title>'
                        "</head><body><p>Hello.</p></body></html>"
                    ),
                }
            ],
        )
        encoded_chapter = load_epub_chapters(encoded_source)[0]
        write_translated_epub(
            encoded_source,
            {"c1": {p.index: "ನಮಸ್ಕಾರ" for p in encoded_chapter.paragraphs}},
            encoded_output,
        )
        assert load_epub_chapters(encoded_output)[0].paragraphs[0].text == "ನಮಸ್ಕಾರ"

        # --- FR-1.3: inline markup rebuild --------------------------------
        _check_inline_markup(tmpdir)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print("test_epub_writer: all assertions passed")


if __name__ == "__main__":
    main()
