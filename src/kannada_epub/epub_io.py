import logging
import posixpath
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

import lxml.etree as etree
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

logger = logging.getLogger(__name__)

OPF_NS = "http://www.idpf.org/2007/opf"
DC_NS = "http://purl.org/dc/elements/1.1/"
_CONTAINER_NS = "urn:oasis:names:tc:opendocument:xmlns:container"
_XHTML_MEDIA_TYPE = "application/xhtml+xml"
_NCX_MEDIA_TYPE = "application/x-dtbncx+xml"
_EPUB_OPS_NS = "http://www.idpf.org/2007/ops"

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


def _collapse(text: str) -> str:
    """Collapse every run of whitespace into a single space."""
    return " ".join(text.split())


def _truncate_title(title: str) -> str:
    """Cap a display title at 200 characters, preferring a word boundary."""
    if len(title) <= 200:
        return title
    cut = title[:200]
    boundary = cut.rfind(" ")
    if boundary > 0:
        cut = cut[:boundary]
    return cut


def _local_name(tag) -> str:
    """Local tag name with any ``{namespace}`` prefix removed (``""`` if not a str)."""
    if isinstance(tag, str):
        return tag.rsplit("}", 1)[-1]
    return ""


def _resolve_zip_path(base_dir: str, href: str) -> str:
    """Resolve a nav/NCX ``href`` against ``base_dir`` to a zip entry path.

    Drops any ``#fragment``, URL-decodes and normalises, matching the way
    ``_spine_documents`` computes spine document zip paths.
    """
    path = href.split("#", 1)[0]
    return posixpath.normpath(posixpath.join(base_dir, unquote(path)))


def _collect_nav_labels(
    zf: zipfile.ZipFile, opf_dir: str, nav_item, labels: dict[str, str]
) -> None:
    """Add EPUB 3 nav TOC entries to ``labels`` (``{zip path: label}``)."""
    nav_path = _resolve_zip_path(opf_dir, nav_item.get("href") or "")
    try:
        data = zf.read(nav_path)
    except KeyError:
        return
    try:
        root = etree.fromstring(data, _XML_PARSER)
    except etree.XMLSyntaxError:
        return
    nav_dir = posixpath.dirname(nav_path)
    for element in root.iter():
        if _local_name(element.tag) != "nav":
            continue
        epub_type = (
            element.get(f"{{{_EPUB_OPS_NS}}}type") or element.get("epub:type") or ""
        )
        if "toc" not in epub_type.split():
            continue
        for anchor in element.iter():
            if _local_name(anchor.tag) != "a":
                continue
            label = _collapse("".join(anchor.itertext()))
            href = anchor.get("href")
            if label and href:
                labels.setdefault(_resolve_zip_path(nav_dir, href), label)


def _collect_ncx_labels(
    zf: zipfile.ZipFile, opf_dir: str, ncx_item, labels: dict[str, str]
) -> None:
    """Add EPUB 2 NCX entries to ``labels`` (``{zip path: label}``)."""
    ncx_path = _resolve_zip_path(opf_dir, ncx_item.get("href") or "")
    try:
        data = zf.read(ncx_path)
    except KeyError:
        return
    try:
        root = etree.fromstring(data, _XML_PARSER)
    except etree.XMLSyntaxError:
        return
    ncx_dir = posixpath.dirname(ncx_path)

    def child(element, name):
        for candidate in element:
            if _local_name(candidate.tag) == name:
                return candidate
        return None

    for nav_point in root.iter():
        if _local_name(nav_point.tag) != "navPoint":
            continue
        label_el = child(nav_point, "navLabel")
        text_el = child(label_el, "text") if label_el is not None else None
        content_el = child(nav_point, "content")
        label = _collapse("".join(text_el.itertext())) if text_el is not None else ""
        src = content_el.get("src") if content_el is not None else None
        if label and src:
            labels.setdefault(_resolve_zip_path(ncx_dir, src), label)


def _navigation_labels(
    zf: zipfile.ZipFile, opf: etree._Element, opf_dir: str, manifest: dict
) -> dict[str, str]:
    """``{zip path: label}`` for every TOC entry, built once per book.

    Prefers the EPUB 3 navigation document (the manifest item whose
    ``properties`` include ``nav``) when present; otherwise falls back to the
    EPUB 2 NCX (the spine's ``toc`` item, else the first
    ``application/x-dtbncx+xml`` item). The first label for a document wins.
    """
    labels: dict[str, str] = {}
    nav_item = next(
        (
            item
            for item in manifest.values()
            if "nav" in (item.get("properties") or "").split()
        ),
        None,
    )
    if nav_item is not None:
        _collect_nav_labels(zf, opf_dir, nav_item, labels)
        return labels

    spine = opf.find(f"{{{OPF_NS}}}spine")
    ncx_item = None
    if spine is not None and spine.get("toc"):
        ncx_item = manifest.get(spine.get("toc"))
    if ncx_item is None:
        ncx_item = next(
            (
                item
                for item in manifest.values()
                if item.get("media-type") == _NCX_MEDIA_TYPE
            ),
            None,
        )
    if ncx_item is not None:
        _collect_ncx_labels(zf, opf_dir, ncx_item, labels)
    return labels


def _chapter_title(soup: BeautifulSoup, nav_label: str | None) -> str | None:
    """Best display title for a chapter.

    Tries, in order: the first ``h1``/``h2``/``title`` inside the body (the
    historical rule), the navigation label, then the first ``h3``-``h6``.
    Empty/whitespace-only candidates are skipped; the result is truncated to
    200 characters at a word boundary.
    """
    scope = soup.body or soup
    heading = scope.find(["h1", "h2", "title"])
    if heading is not None:
        title = _collapse(heading.get_text(" ", strip=True))
        if title:
            return _truncate_title(title)
    if nav_label:
        return _truncate_title(nav_label)
    lower = scope.find(["h3", "h4", "h5", "h6"])
    if lower is not None:
        title = _collapse(lower.get_text(" ", strip=True))
        if title:
            return _truncate_title(title)
    return None


def _spine_documents(
    path: str | Path,
) -> tuple[list[tuple[str, str, bytes]], dict[str, str]]:
    """Read each linear XHTML spine document plus the book's TOC labels.

    Returns ``(documents, nav_labels)`` where ``documents`` holds
    ``(manifest id, zip path, content)`` in order and ``nav_labels`` maps a
    document's zip path to its navigation label. Matches what the previous
    EbookLib-based reader returned, so chapter ids and paragraph indices of
    existing checkpoints stay valid:

    - spine items with ``linear="no"`` are skipped (nav/TOC aids, not content);
    - only ``application/xhtml+xml`` items are documents;
    - an XHTML item with the non-standard ``cover`` property is skipped
      (EbookLib typed it as a cover, not a document);
    - the first occurrence of each spine ``idref`` wins; a later duplicate is
      skipped and logged once per book (some exports, e.g. Wikisource, list
      the same document more than once). Duplicates are checked after the
      filters above, so a skipped non-linear first listing does not hide a
      later linear one;
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
        # Built once per book from the still-open zip, shared by all chapters.
        nav_labels = _navigation_labels(zf, opf, opf_dir, manifest)

        documents: list[tuple[str, str, bytes]] = []
        seen: set[str] = set()
        skipped_duplicates: list[str] = []
        for itemref in spine.iterfind(f"{{{OPF_NS}}}itemref"):
            if itemref.get("linear", "yes").lower() == "no":
                continue
            idref = itemref.get("idref")
            item = manifest.get(idref)
            if item is None or item.get("media-type") != _XHTML_MEDIA_TYPE:
                continue
            if "cover" in item.get("properties", "").split():
                continue
            # Checked after the filters above, so a skipped non-linear first
            # occurrence does not hide a later linear one.
            if idref in seen:
                skipped_duplicates.append(idref)
                continue
            seen.add(idref)
            name = posixpath.normpath(
                posixpath.join(opf_dir, unquote(item.get("href", "")))
            )
            try:
                documents.append((idref, name, zf.read(name)))
            except KeyError as exc:
                raise ValueError(
                    f"spine item {idref!r} points to missing entry {name!r}"
                ) from exc
        if skipped_duplicates:
            logger.warning(
                "Skipping duplicate spine idref(s) in %s: %s",
                path,
                ", ".join(skipped_duplicates),
            )
        return documents, nav_labels


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
    *,
    skip_gutenberg_boilerplate: bool = True,
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

    ``Chapter.title`` is display-only and is resolved in fallback order: the
    first ``h1``/``h2``/``title`` inside the body (unchanged from before),
    then the label of this document's entry in the book's table of contents
    (the EPUB 3 nav, else the EPUB 2 NCX), then the first body ``h3``-``h6``,
    otherwise ``None``. Candidates are whitespace-collapsed, must be
    non-empty, and are truncated to 200 characters at a word boundary. Titles
    never affect paragraph indices or checkpoint matching.

    Reading order comes from the spine (not manifest iteration order), since
    the spine is what the EPUB spec actually guarantees reflects intended
    reading order. Spine items marked non-linear (EPUB3 nav/TOC documents,
    typically linear="no") are skipped — they are navigation aids, not
    content. Additional spine IDs (e.g. Project Gutenberg boilerplate
    chapters like "pg-header"/"pg-footer") can be skipped via `exclude_ids`.
    By default, blocks inside Project Gutenberg boilerplate elements (the
    ``pg-boilerplate`` class token or ``pg-header``/``pg-footer`` IDs) are
    omitted while retaining their original block indices. Set
    ``skip_gutenberg_boilerplate=False`` to include them.
    """
    documents, nav_labels = _spine_documents(path)
    chapters: list[Chapter] = []
    excluded = set(exclude_ids or [])

    for idref, name, content in documents:
        if idref in excluded:
            continue

        soup = BeautifulSoup(content, "lxml")
        # Titles are display-only: they never affect paragraph indices or
        # checkpoint matching. Resolved in fallback order by _chapter_title:
        # a body h1/h2/title first, then the book's TOC label for this
        # document, then a body h3-h6.
        title = _chapter_title(soup, nav_labels.get(name))

        paragraphs = []
        for i, tag in enumerate(soup.find_all(BLOCK_TAGS)):
            # A block nested inside another block would otherwise be extracted
            # twice (once as the parent's text, once as its own). Skip it, but
            # keep enumerate() advancing so all other indices are unchanged.
            if tag.find(BLOCK_TAGS) is not None:
                continue
            if skip_gutenberg_boilerplate:
                current = tag
                in_boilerplate = False
                while current is not None:
                    classes = current.get("class") or []
                    if (
                        "pg-boilerplate" in classes
                        or current.get("id") in {"pg-header", "pg-footer"}
                    ):
                        in_boilerplate = True
                        break
                    current = current.parent
                if in_boilerplate:
                    continue
            text = " ".join(tag.get_text().split())
            if len(text) >= min_paragraph_chars:
                paragraphs.append(Paragraph(i, text, _paragraph_kind(tag)))

        if paragraphs:
            chapters.append(Chapter(id=idref, title=title, paragraphs=paragraphs))

    return chapters
