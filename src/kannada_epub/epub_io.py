import posixpath
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

import lxml.etree as etree
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

OPF_NS = "http://www.idpf.org/2007/opf"
DC_NS = "http://purl.org/dc/elements/1.1/"
_CONTAINER_NS = "urn:oasis:names:tc:opendocument:xmlns:container"
_XHTML_MEDIA_TYPE = "application/xhtml+xml"

# EPUBs are untrusted input: never expand entities or fetch DTDs.
_XML_PARSER = etree.XMLParser(resolve_entities=False, no_network=True)

BLOCK_TAGS = [
    "p",
    "li",
    "blockquote",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "td",
    "th",
    "caption",
]

# Table cells flow through the pipeline as ordinary paragraphs but are tagged
# with this kind so downstream consumers can tell them apart.
TABLE_CELL_TAGS = ("td", "th")


def _paragraph_kind(tag) -> str:
    """Classify a kept block as a table cell or ordinary text.

    A cell is either the ``td``/``th`` itself, or a block (e.g. the ``<p>`` in
    ``<td><p>x</p></td>``) whose nearest block-level ancestor is one. A
    ``caption`` stays ``"text"``.
    """
    if tag.name in TABLE_CELL_TAGS:
        return "table_cell"
    parent = tag.find_parent(BLOCK_TAGS)
    if parent is not None and parent.name in TABLE_CELL_TAGS:
        return "table_cell"
    return "text"


def find_opf_path(zf: zipfile.ZipFile) -> str:
    """Zip path of the package document named by ``META-INF/container.xml``."""
    try:
        container = zf.read("META-INF/container.xml")
    except KeyError as exc:
        raise ValueError("source epub has no META-INF/container.xml") from exc
    root = etree.fromstring(container, _XML_PARSER)
    rootfile = root.find(f".//{{{_CONTAINER_NS}}}rootfile")
    if rootfile is None or not rootfile.get("full-path"):
        raise ValueError("container.xml has no rootfile with a full-path")
    return rootfile.get("full-path")


def _read_opf(zf: zipfile.ZipFile) -> tuple[str, etree._Element]:
    opf_path = find_opf_path(zf)
    try:
        data = zf.read(opf_path)
    except KeyError as exc:
        raise ValueError(f"package document {opf_path!r} is missing") from exc
    return opf_path, etree.fromstring(data, _XML_PARSER)


def _spine_documents(path: str | Path) -> list[tuple[str, bytes]]:
    """``(manifest id, content)`` of each linear XHTML spine document, in order.

    Matches what the previous EbookLib-based reader returned, so chapter ids
    and paragraph indices of existing checkpoints stay valid:

    - spine items with ``linear="no"`` are skipped (nav/TOC aids, not content);
    - only ``application/xhtml+xml`` items are documents;
    - an XHTML item with the non-standard ``cover`` property is skipped
      (EbookLib typed it as a cover, not a document);
    - hrefs are URL-unquoted and resolved against the package directory;
    - a spine document missing from the zip is an error, not a silent skip.
    """
    with zipfile.ZipFile(path) as zf:
        opf_path, opf = _read_opf(zf)
        opf_dir = posixpath.dirname(opf_path)
        manifest = {
            item.get("id"): item
            for item in opf.iterfind(f"{{{OPF_NS}}}manifest/{{{OPF_NS}}}item")
            if item.get("id")
        }
        spine = opf.find(f"{{{OPF_NS}}}spine")
        if spine is None:
            raise ValueError("package document has no spine")

        documents: list[tuple[str, bytes]] = []
        for itemref in spine.iterfind(f"{{{OPF_NS}}}itemref"):
            if itemref.get("linear", "yes").lower() == "no":
                continue
            idref = itemref.get("idref")
            item = manifest.get(idref)
            if item is None or item.get("media-type") != _XHTML_MEDIA_TYPE:
                continue
            if "cover" in item.get("properties", "").split():
                continue
            name = posixpath.normpath(
                posixpath.join(opf_dir, unquote(item.get("href", "")))
            )
            try:
                documents.append((idref, zf.read(name)))
            except KeyError as exc:
                raise ValueError(
                    f"spine item {idref!r} points to missing entry {name!r}"
                ) from exc
        return documents


def read_epub_metadata(path: str | Path) -> tuple[str | None, str | None]:
    """The first ``dc:title`` and ``dc:creator`` of an EPUB, if present."""
    with zipfile.ZipFile(path) as zf:
        _, opf = _read_opf(zf)
    metadata = opf.find(f"{{{OPF_NS}}}metadata")
    if metadata is None:
        return None, None

    def first(tag: str) -> str | None:
        element = metadata.find(f"{{{DC_NS}}}{tag}")
        text = (element.text or "").strip() if element is not None else ""
        return text or None

    return first("title"), first("creator")


@dataclass
class Paragraph:
    index: int
    text: str
    kind: str = "text"


@dataclass
class Chapter:
    id: str
    title: str | None
    paragraphs: list[Paragraph]

    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.paragraphs)


def load_epub_chapters(
    path: str | Path,
    min_paragraph_chars: int = 1,
    exclude_ids: list[str] | set[str] | None = None,
) -> list[Chapter]:
    """Extract reading-order chapters as plain paragraph text.

    Deliberately minimal: no DOM/attribute preservation — just enough
    structure (chapter -> ordered paragraphs) to translate and review.
    Reassembly back into an .epub now lives in
    `kannada_epub.epub_writer`, which re-reads the source with this same
    parse (and same block enumeration) so a `Paragraph.index` maps back to
    its element. Table cells (``td``/``th``) are included as translatable
    units and tagged ``Paragraph.kind == "table_cell"``; ``caption`` is
    treated as ordinary text. Full AST-preserving extraction (FR-1.3)
    remains a separate, heavier piece of work for later.

    A block tag that itself contains a block tag (e.g.
    ``<blockquote><p>…</p></blockquote>``) is skipped, so its text is not
    emitted twice; ``enumerate`` still advances over the skipped tag, keeping
    the index of every other paragraph stable for the writer.

    Reading order comes from the spine (not manifest iteration order), since
    the spine is what the EPUB spec actually guarantees reflects intended
    reading order. Spine items marked non-linear (EPUB3 nav/TOC documents,
    typically linear="no") are skipped — they are navigation aids, not
    content. Additional spine IDs (e.g. Project Gutenberg boilerplate
    chapters like "pg-header"/"pg-footer") can be skipped via `exclude_ids`.
    """
    chapters: list[Chapter] = []
    excluded = set(exclude_ids or [])

    for idref, content in _spine_documents(path):
        if idref in excluded:
            continue

        soup = BeautifulSoup(content, "lxml")
        # Title from the body only: the head's <title> is usually the book
        # title, not the chapter's.
        title_tag = (soup.body or soup).find(["h1", "h2", "title"])
        title = (
            " ".join(title_tag.get_text(" ", strip=True).split())
            if title_tag
            else None
        )

        paragraphs = []
        for i, tag in enumerate(soup.find_all(BLOCK_TAGS)):
            # A block nested inside another block would otherwise be extracted
            # twice (once as the parent's text, once as its own). Skip it, but
            # keep enumerate() advancing so all other indices are unchanged.
            if tag.find(BLOCK_TAGS) is not None:
                continue
            text = " ".join(tag.get_text().split())
            if len(text) >= min_paragraph_chars:
                paragraphs.append(Paragraph(i, text, _paragraph_kind(tag)))

        if paragraphs:
            chapters.append(Chapter(id=idref, title=title, paragraphs=paragraphs))

    return chapters