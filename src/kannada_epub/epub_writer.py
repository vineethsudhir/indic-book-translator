"""Write a translated Kannada EPUB from the ORIGINAL epub plus translations.

This is the reassembly half of FR-1.3 / FR-5. The translator
(``scripts/translate_book.py``) only produces per-chapter JSON keyed by
positions in the *filtered* paragraph list; this module maps those back onto
elements (via :func:`translations_from_batches` and ``Paragraph.index``) and
repackages the book.

Everything that is not a document carrying translations is copied from the
source archive byte-for-byte: CSS, images, the nav/NCX, every other document,
and every element attribute and ``id``. Only the documents with translations
and the OPF are rewritten, plus an embedded Kannada font and stylesheet.

Known prototype limitation: inline markup inside a *translated* paragraph
(``i``, ``strong``, ``span``, ``br``, …) is dropped when the Kannada text
replaces the element's contents. Descendants that carry an ``id`` (typically
empty ``<a>`` anchors that the TOC links to) are preserved, emptied of text,
so navigation keeps working.
"""

from __future__ import annotations

import os
import posixpath
import re
import tempfile
import time
import zipfile
from pathlib import Path

import lxml.etree as etree
from bs4 import BeautifulSoup

from .epub_io import BLOCK_TAGS, Chapter

_OPF_NS = "http://www.idpf.org/2007/opf"
_DC_NS = "http://purl.org/dc/elements/1.1/"
_CONTAINER_NS = "urn:oasis:names:tc:opendocument:xmlns:container"

# (filename, media-type) for the files added next to the OPF.
_FONT_FILES = [
    ("NotoSansKannada-Regular.ttf", "font/ttf"),
    ("NotoSansKannada-Bold.ttf", "font/ttf"),
    ("OFL.txt", "text/plain"),
]

KANNADA_CSS = """\
@font-face {
  font-family: "Noto Sans Kannada";
  font-style: normal;
  font-weight: 400;
  src: url("../fonts/NotoSansKannada-Regular.ttf") format("truetype");
}

@font-face {
  font-family: "Noto Sans Kannada";
  font-style: normal;
  font-weight: 700;
  src: url("../fonts/NotoSansKannada-Bold.ttf") format("truetype");
}

body, p, li, blockquote, h1, h2, h3, h4, h5, h6, td, th {
  font-family: "Noto Sans Kannada", sans-serif;
  line-height: 1.5em;
}

table {
  table-layout: auto;
}

td, th {
  overflow-wrap: anywhere;
  word-break: normal;
  line-height: 1.5em;
  vertical-align: top;
}

.qa-review-flag {
  background-color: #fff3cd;
}
"""


def _default_font_dir() -> Path:
    # <repo>/assets/fonts, resolved relative to this module:
    # src/kannada_epub/epub_writer.py -> src/kannada_epub -> src -> <repo>.
    return Path(__file__).resolve().parent.parent.parent / "assets" / "fonts"


def translations_from_batches(
    chapters: list[Chapter], batches: list[dict]
) -> dict[str, dict[int, str]]:
    """Map per-chapter batch payloads back onto ``Paragraph.index`` keys.

    Batch ``paragraph_start``/``paragraph_end`` are positions in
    ``chapter.paragraphs`` (the filtered list), so
    ``edited_kannada[k]`` translates ``chapter.paragraphs[paragraph_start + k]``.
    The writer needs the underlying ``Paragraph.index`` (position among ALL
    block tags), hence this translation of coordinates.

    Raises ``ValueError`` for an unknown chapter or an out-of-range position.
    """
    by_id = {c.id: c for c in chapters}
    result: dict[str, dict[int, str]] = {}

    for batch in batches:
        chapter_id = batch.get("chapter_id")
        chapter = by_id.get(chapter_id)
        if chapter is None:
            raise ValueError(f"batch refers to unknown chapter {chapter_id!r}")

        start = batch.get("paragraph_start", 0)
        texts = batch.get("edited_kannada") or []
        if start < 0:
            raise ValueError(
                f"chapter {chapter_id!r}: negative paragraph_start {start}"
            )
        for k, text in enumerate(texts):
            position = start + k
            if position >= len(chapter.paragraphs):
                raise ValueError(
                    f"chapter {chapter_id!r}: position {position} out of range "
                    f"(chapter has {len(chapter.paragraphs)} paragraphs)"
                )
            paragraph_index = chapter.paragraphs[position].index
            result.setdefault(chapter_id, {})[paragraph_index] = text

    return result


def _find_opf_path(zf: zipfile.ZipFile) -> str:
    try:
        container = zf.read("META-INF/container.xml")
    except KeyError as exc:
        raise ValueError("source epub has no META-INF/container.xml") from exc
    root = etree.fromstring(container)
    rootfile = root.find(f".//{{{_CONTAINER_NS}}}rootfile")
    if rootfile is None or not rootfile.get("full-path"):
        raise ValueError("container.xml has no rootfile with a full-path")
    return rootfile.get("full-path")


def _manifest_items(opf_root: etree._Element) -> dict[str, dict[str, str]]:
    items: dict[str, dict[str, str]] = {}
    for item in opf_root.findall(f".//{{{_OPF_NS}}}manifest/{{{_OPF_NS}}}item"):
        item_id = item.get("id")
        if item_id:
            items[item_id] = {
                "href": item.get("href", ""),
                "media-type": item.get("media-type", ""),
            }
    return items


def _serialize_xhtml(original: bytes, soup: BeautifulSoup) -> bytes:
    """Serialize a DOM parsed with the HTML parser back to well-formed XHTML.

    Keeps the original XML declaration / DOCTYPE (which ``BeautifulSoup``
    turns into comment nodes or drops) and lets ``minimal`` formatting emit
    self-closing void elements such as ``<br/>``, ``<link/>`` and ``<meta/>``.
    """
    text = original.decode("utf-8")
    parts: list[str] = []

    declaration = re.match(r"\s*(<\?xml[^>]*\?>)", text)
    if declaration:
        parts.append(declaration.group(1))

    doctype = re.search(r"<!DOCTYPE[^>]*>", text, re.IGNORECASE)
    if doctype:
        parts.append(doctype.group(0))

    html = soup.html.decode(formatter="minimal") if soup.html is not None else soup.decode(formatter="minimal")
    parts.append(html)
    return ("\n".join(parts) + "\n").encode("utf-8")


def _translate_document(
    content: bytes,
    translations_by_index: dict[int, str],
    css_href: str,
    flagged_indices: set[int] | frozenset[int] = frozenset(),
) -> bytes:
    soup = BeautifulSoup(content, "lxml")
    blocks = soup.find_all(BLOCK_TAGS)

    def _element_at(index: int):
        if index < 0 or index >= len(blocks):
            raise ValueError(
                f"paragraph index {index} out of range for document "
                f"({len(blocks)} block elements)"
            )
        return blocks[index]

    for index, text in sorted(translations_by_index.items()):
        element = _element_at(index)

        # Keep anchors/id-bearing descendants alive (TOC targets), emptied of
        # text, then replace the rest of the element's contents.
        preserved = [
            soup.new_tag(desc.name, attrs=dict(desc.attrs))
            for desc in element.find_all(attrs={"id": True})
        ]
        element.clear()
        for anchor in preserved:
            element.append(anchor)
        element.append(text)

    # QA review flags: append the class so we never clobber an existing one.
    for index in sorted(flagged_indices):
        element = _element_at(index)
        classes = element.get("class") or []
        if "qa-review-flag" not in classes:
            element["class"] = [*classes, "qa-review-flag"]

    if soup.html is not None:
        soup.html["xmlns"] = "http://www.w3.org/1999/xhtml"
        soup.html["lang"] = "kn"
        soup.html["xml:lang"] = "kn"

    if soup.head is not None:
        link = soup.new_tag("link", rel="stylesheet", type="text/css", href=css_href)
        soup.head.append(link)

    return _serialize_xhtml(content, soup)


def _update_opf(
    opf_bytes: bytes, added_items: list[tuple[str, str, str]], language: str = "kn"
) -> bytes:
    """Add manifest entries and force the package language, keeping the rest."""
    root = etree.fromstring(opf_bytes)
    manifest = root.find(f"{{{_OPF_NS}}}manifest")
    if manifest is None:
        raise ValueError("OPF has no manifest")

    existing_ids = {
        item.get("id")
        for item in manifest.findall(f"{{{_OPF_NS}}}item")
        if item.get("id")
    }
    for item_id, href, media_type in added_items:
        if item_id in existing_ids:
            raise ValueError(f"OPF manifest already has id {item_id!r}")
        item = etree.SubElement(manifest, f"{{{_OPF_NS}}}item")
        item.set("href", href)
        item.set("id", item_id)
        item.set("media-type", media_type)

    metadata = root.find(f"{{{_OPF_NS}}}metadata")
    if metadata is None:
        raise ValueError("OPF has no metadata")
    languages = metadata.findall(f"{{{_DC_NS}}}language")
    if languages:
        for lang in languages:
            lang.text = language
    else:
        lang = etree.SubElement(metadata, f"{{{_DC_NS}}}language")
        lang.text = language

    return etree.tostring(root, xml_declaration=True, encoding="utf-8", pretty_print=True)


def write_translated_epub(
    source_epub: str | Path,
    translations: dict[str, dict[int, str]],
    output_path: str | Path,
    *,
    font_dir: str | Path | None = None,
    flagged: dict[str, set[int]] | None = None,
) -> None:
    """Repackage ``source_epub`` with translated paragraphs replaced in place.

    ``translations`` maps chapter id -> {``Paragraph.index``: Kannada text}.
    ``flagged`` optionally maps chapter id -> the set of ``Paragraph.index``
    values to mark with the ``qa-review-flag`` CSS class (QA flagged them for
    human review).
    """
    source_epub = Path(source_epub)
    output_path = Path(output_path)
    font_dir = Path(font_dir) if font_dir is not None else _default_font_dir()
    flagged_by_chapter = {cid: set(indices) for cid, indices in (flagged or {}).items()}

    font_bytes: dict[str, bytes] = {}
    for filename, _media_type in _FONT_FILES:
        path = font_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"font asset not found: {path}")
        font_bytes[filename] = path.read_bytes()

    with zipfile.ZipFile(source_epub, "r") as zin:
        infos = {info.filename: info for info in zin.infolist()}
        source_data = {name: zin.read(name) for name in infos}
        opf_path = _find_opf_path(zin)

        opf_dir = posixpath.dirname(opf_path)
        opf_root = etree.fromstring(source_data[opf_path])
        manifest = _manifest_items(opf_root)

        # Names of the package files we add, relative to the OPF directory.
        css_rel = "css/kannada.css"
        css_zip_path = posixpath.join(opf_dir, css_rel)

        modified_docs: dict[str, bytes] = {}
        for chapter_id in sorted(set(translations) | set(flagged_by_chapter)):
            by_index = translations.get(chapter_id, {})
            item = manifest.get(chapter_id)
            if item is None:
                raise ValueError(
                    f"chapter {chapter_id!r} is not in the OPF manifest"
                )
            doc_path = posixpath.normpath(
                posixpath.join(opf_dir, posixpath.normpath(item["href"]))
            )
            if doc_path not in source_data:
                raise ValueError(
                    f"chapter {chapter_id!r} manifest href {item['href']!r} "
                    f"resolves to missing entry {doc_path!r}"
                )
            doc_dir = posixpath.dirname(doc_path)
            css_href = posixpath.relpath(css_zip_path, doc_dir or ".")
            modified_docs[doc_path] = _translate_document(
                source_data[doc_path],
                by_index,
                css_href,
                flagged_by_chapter.get(chapter_id, frozenset()),
            )

        added_items: list[tuple[str, str, str]] = [
            ("kannada-font-regular", f"fonts/{_FONT_FILES[0][0]}", "font/ttf"),
            ("kannada-font-bold", f"fonts/{_FONT_FILES[1][0]}", "font/ttf"),
            ("kannada-ofl", f"fonts/{_FONT_FILES[2][0]}", "text/plain"),
            ("kannada-css", css_rel, "text/css"),
        ]
        added_bytes: dict[str, bytes] = {
            posixpath.join(opf_dir, "fonts", name): data
            for name, data in font_bytes.items()
        }
        added_bytes[css_zip_path] = KANNADA_CSS.encode("utf-8")

        modified_opf = _update_opf(source_data[opf_path], added_items)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(output_path.parent), prefix=output_path.name + ".", suffix=".tmp"
    )
    os.close(fd)
    try:
        with zipfile.ZipFile(tmp_name, "w") as zout:
            # mimetype must be first and stored uncompressed.
            mimetype_info = zipfile.ZipInfo(
                "mimetype",
                date_time=infos["mimetype"].date_time
                if "mimetype" in infos
                else time.localtime()[:6],
            )
            mimetype_info.compress_type = zipfile.ZIP_STORED
            mimetype_info.external_attr = infos["mimetype"].external_attr if "mimetype" in infos else 0
            zout.writestr(mimetype_info, source_data.get("mimetype", b"application/epub+zip"))

            for name, info in infos.items():
                if name == "mimetype":
                    continue
                if name == opf_path:
                    data: bytes | None = modified_opf
                elif name in modified_docs:
                    data = modified_docs[name]
                else:
                    data = None
                out_info = zipfile.ZipInfo(filename=name, date_time=info.date_time)
                out_info.compress_type = zipfile.ZIP_DEFLATED
                out_info.external_attr = info.external_attr
                zout.writestr(out_info, data if data is not None else source_data[name])

            for name, data in added_bytes.items():
                if name in infos:
                    continue
                out_info = zipfile.ZipInfo(filename=name, date_time=time.localtime()[:6])
                out_info.compress_type = zipfile.ZIP_DEFLATED
                zout.writestr(out_info, data)

        # mkstemp creates the file 0600; give the finished book normal
        # permissions so other apps and users can open it like any document.
        os.chmod(tmp_name, 0o644)
        os.replace(tmp_name, output_path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise
