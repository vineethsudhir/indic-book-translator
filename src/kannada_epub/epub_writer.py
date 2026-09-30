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

Inline markup inside a *translated* paragraph is preserved when the incoming
translation carries the reader's ``⟦n⟧ … ⟦/n⟧`` markers (see
:mod:`kannada_epub.inline_markup`): the original tags and attributes are rebuilt
around the translated segments. A translation without markers — including one
from a run with markup preservation off — is written as plain text, as before.
Descendants that carry an ``id`` (typically empty ``<a>`` anchors that the TOC
links to) are preserved, emptied of text, so navigation keeps working.
"""

from __future__ import annotations

import os
import posixpath
import re
import tempfile
import time
import uuid
import warnings
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlsplit

import lxml.etree as etree
from bs4 import BeautifulSoup, NavigableString, Tag

from .cover import (
    CoverText,
    is_gutenberg_generated_cover,
    is_translator_generated_cover,
    render_cover,
)
from .epub_io import BLOCK_TAGS, Chapter, _marked_elements, find_opf_path
from .inline_markup import MARKER_RE, MarkerSpan, parse_markers, strip_markers
from .languages import LANGUAGES, TargetLanguage

_OPF_NS = "http://www.idpf.org/2007/opf"
_DC_NS = "http://purl.org/dc/elements/1.1/"
_NCX_MEDIA_TYPE = "application/x-dtbncx+xml"
_PG_TITLE_PREFIX = "The Project Gutenberg eBook of "
_PG_TITLE_SUFFIX = re.compile(r"\s*\|\s*Project Gutenberg\s*$", re.IGNORECASE)
# Temporary marker on id spans kept from emptied boilerplate; removed before
# writing, together with every marked span nothing links to.
_RETAINED_ATTR = "data-kn-retained-id"
# Temporary marker on a kept boilerplate element whose id mentions Gutenberg:
# the id is dropped before writing unless something links to it.
_KEPT_PG_ID_ATTR = "data-kn-kept-pg-id"

# Elements kept when "hollowing out" Project Gutenberg boilerplate: the block
# elements (``BLOCK_TAGS``) are the ``Paragraph.index`` join key and must
# survive, and these containers keep the surrounding markup valid XHTML.
# Everything else inside the boilerplate is removed; an ``id`` on a removed
# element survives as an empty retained ``<span>``.
_HOLLOW_CONTAINER_TAGS = frozenset(
    {
        "div",
        "section",
        "header",
        "footer",
        "ul",
        "ol",
        "dl",
        "dt",
        "dd",
        "table",
        "thead",
        "tbody",
        "tfoot",
        "tr",
        "blockquote",
        "figure",
        "pre",
        "hr",
    }
)
_HOLLOW_KEEP_TAGS = frozenset(BLOCK_TAGS) | _HOLLOW_CONTAINER_TAGS

# A retained ``<span>`` cannot sit directly in these; it is moved into the
# nearest kept block (a following/preceding ``li``/``td``/…) instead.
_INVALID_RETAINED_PARENTS = frozenset(
    {"ul", "ol", "dl", "table", "thead", "tbody", "tfoot", "tr"}
)
_RETAINED_TARGET_TAGS = (
    "li",
    "td",
    "th",
    "p",
    "dt",
    "dd",
    "caption",
    "blockquote",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
)

# Marks every output as unreviewed machine translation, so a copy that gets
# shared still says what it is.
def machine_translation_contributor(language_name: str) -> str:
    """The ``dc:contributor`` value for a target language name."""
    return (
        f"Unreviewed machine translation into {language_name} "
        "(Indic Book Translator)"
    )


MACHINE_TRANSLATION_CONTRIBUTOR = machine_translation_contributor("Kannada")

_FONT_CSS_TEMPLATE = """\
@font-face {{
  font-family: "{family}";
  font-style: normal;
  font-weight: 400;
  src: url("../fonts/{regular}") format("truetype");
}}

@font-face {{
  font-family: "{family}";
  font-style: normal;
  font-weight: 700;
  src: url("../fonts/{bold}") format("truetype");
}}

body, p, li, blockquote, h1, h2, h3, h4, h5, h6, td, th {{
  font-family: "{family}", sans-serif;
  line-height: 1.5em;
}}

table {{
  table-layout: auto;
}}

td, th {{
  overflow-wrap: anywhere;
  word-break: normal;
  line-height: 1.5em;
  vertical-align: top;
}}

.qa-review-flag {{
  background-color: #fff3cd;
}}
"""

# Used when the target language's font files are not bundled: keep the family
# name (a reader that has the font can still use it) with a serif fallback.
_FALLBACK_CSS_TEMPLATE = """\
body, p, li, blockquote, h1, h2, h3, h4, h5, h6, td, th {{
  font-family: "{family}", serif;
  line-height: 1.5em;
}}

table {{
  table-layout: auto;
}}

td, th {{
  overflow-wrap: anywhere;
  word-break: normal;
  line-height: 1.5em;
  vertical-align: top;
}}

.qa-review-flag {{
  background-color: #fff3cd;
}}
"""


def font_css_for(language: TargetLanguage) -> str:
    """The stylesheet embedding ``language``'s font files."""
    return _FONT_CSS_TEMPLATE.format(
        family=language.font_family,
        regular=language.font_regular,
        bold=language.font_bold,
    )


def fallback_css_for(language: TargetLanguage) -> str:
    """The stylesheet for a language whose font files are not bundled."""
    return _FALLBACK_CSS_TEMPLATE.format(family=language.font_family)


KANNADA_CSS = font_css_for(LANGUAGES["kn"])


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


def _used_marker_ids(segments: list) -> set[int]:
    """Every marker id that appears in a parsed :func:`parse_markers` tree."""
    used: set[int] = set()
    stack = list(segments)
    while stack:
        segment = stack.pop()
        if isinstance(segment, MarkerSpan):
            used.add(segment.marker_id)
            stack.extend(segment.children)
    return used


def _build_marked_nodes(soup, segments: list, marked: list) -> list:
    """Rebuild parsed segments as nodes, reusing the source tags' names/attrs."""
    nodes: list = []
    for segment in segments:
        if isinstance(segment, str):
            nodes.append(segment)
            continue
        source = marked[segment.marker_id - 1]
        element = soup.new_tag(source.name, attrs=dict(source.attrs))
        for child in _build_marked_nodes(soup, segment.children, marked):
            element.append(child)
        nodes.append(element)
    return nodes


def _replace_block(soup, element, text: str) -> None:
    """Replace a translated block's contents, preserving inline markup.

    If ``text`` carries ``⟦n⟧``/``⟦/n⟧`` markers that parse against the block's
    own marked elements, the original tags (with their attributes, including
    ``id``/``href``) are rebuilt around the translated segments and nested as in
    the source. Otherwise the markers are stripped and the plain text is used,
    exactly as before.

    ID-bearing descendants rebuilt this way keep their ``id`` on the new
    element, so they are not also added as empty preserved copies; the other
    ID-bearing descendants are preserved as empty copies so TOC targets live.
    """
    has_marker = MARKER_RE.search(text) is not None
    marked = _marked_elements(element)
    tree = (
        parse_markers(text, range(1, len(marked) + 1))
        if has_marker and marked
        else None
    )
    if tree is not None:
        rebuilt = {id(marked[n - 1]) for n in _used_marker_ids(tree)}
        preserved = [
            soup.new_tag(desc.name, attrs=dict(desc.attrs))
            for desc in element.find_all(attrs={"id": True})
            if id(desc) not in rebuilt
        ]
        element.clear()
        for anchor in preserved:
            element.append(anchor)
        for node in _build_marked_nodes(soup, tree, marked):
            element.append(node)
        return

    preserved = [
        soup.new_tag(desc.name, attrs=dict(desc.attrs))
        for desc in element.find_all(attrs={"id": True})
    ]
    element.clear()
    for anchor in preserved:
        element.append(anchor)
    element.append(strip_markers(text) if has_marker else text)


def _translate_document(
    content: bytes,
    translations_by_index: dict[int, str],
    css_href: str,
    flagged_indices: set[int] | frozenset[int] = frozenset(),
    language: TargetLanguage = LANGUAGES["kn"],
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
        _replace_block(soup, element, text)

    # QA review flags: append the class so we never clobber an existing one.
    for index in sorted(flagged_indices):
        element = _element_at(index)
        classes = element.get("class") or []
        if "qa-review-flag" not in classes:
            element["class"] = [*classes, "qa-review-flag"]

    if soup.html is not None:
        soup.html["xmlns"] = "http://www.w3.org/1999/xhtml"
        soup.html["lang"] = language.key
        soup.html["xml:lang"] = language.key

    if soup.head is not None:
        link = soup.new_tag("link", rel="stylesheet", type="text/css", href=css_href)
        soup.head.append(link)

    return _serialize_xhtml(content, soup)


def _has_gutenberg_boilerplate(tag) -> bool:
    classes = tag.get("class") or []
    return "pg-boilerplate" in classes or tag.get("id") in {"pg-header", "pg-footer"}


def _first_retained_target(tag):
    """The element a relocated retained ``<span>`` can be placed in."""
    if tag.name in _RETAINED_TARGET_TAGS:
        return tag
    return tag.find(list(_RETAINED_TARGET_TAGS))


def _relocate_retained_spans(soup: BeautifulSoup, element) -> None:
    """Move retained spans out of elements that cannot contain a span."""
    for span in element.find_all(attrs={_RETAINED_ATTR: True}):
        parent = span.parent
        if parent is None or parent.name not in _INVALID_RETAINED_PARENTS:
            continue
        target = None
        prepend = False
        for sibling in span.next_siblings:
            if isinstance(sibling, Tag):
                target = _first_retained_target(sibling)
                if target is not None:
                    prepend = True
                    break
        if target is None:
            for sibling in span.previous_siblings:
                if isinstance(sibling, Tag):
                    target = _first_retained_target(sibling)
                    if target is not None:
                        break
        if target is None:
            target = element
        span.extract()
        if prepend:
            target.insert(0, span)
        else:
            target.append(span)


def _hollow_boilerplate_element(soup: BeautifulSoup, element) -> None:
    """Empty a boilerplate element without changing its block sequence.

    Removes every text node (and comment) and every non-structural
    descendant, but keeps ``BLOCK_TAGS`` elements and their containers, so
    ``find_all(BLOCK_TAGS)`` on the output returns the same tags in the same
    order as on the input. An ``id`` on a removed element becomes an empty
    retained ``<span>``; an ``id`` on a kept element stays on it.
    """
    for node in list(element.descendants):
        if isinstance(node, NavigableString):
            node.extract()

    # Deepest-first: a removed element's children are already unwrapped (and
    # their ids already turned into spans) before the element itself is.
    for tag in reversed(element.find_all(True)):
        if tag.name in _HOLLOW_KEEP_TAGS:
            if "gutenberg" in (tag.get("id") or "").lower():
                tag[_KEPT_PG_ID_ATTR] = ""
            continue
        identifier = tag.get("id")
        if identifier:
            span = soup.new_tag("span")
            span["id"] = identifier
            span[_RETAINED_ATTR] = ""
            tag.insert_before(span)
        tag.unwrap()

    _relocate_retained_spans(soup, element)


def _strip_boilerplate_elements(soup: BeautifulSoup) -> None:
    candidates = [tag for tag in soup.find_all(True) if _has_gutenberg_boilerplate(tag)]
    for element in candidates:
        if any(_has_gutenberg_boilerplate(parent) for parent in element.parents):
            continue
        _hollow_boilerplate_element(soup, element)


def _is_top_level_toc_list(ol, top_toc_list) -> bool:
    return ol is top_toc_list


def _strip_gutenberg_navigation(soup: BeautifulSoup, *, ncx: bool) -> None:
    if ncx:
        for point in list(soup.find_all("navPoint")):
            label = point.find("navLabel")
            if label is not None and "gutenberg" in label.get_text(" ", strip=True).lower():
                point.decompose()
        return

    top_toc_list = None
    for nav in soup.find_all("nav"):
        nav_type = nav.get("epub:type") or nav.get("type") or ""
        if "toc" in (nav_type if isinstance(nav_type, str) else " ".join(nav_type)).split():
            top_toc_list = next(
                (ol for ol in nav.find_all("ol") if ol.find_parent("ol") is None),
                None,
            )
            if top_toc_list is not None:
                break

    for item in list(soup.find_all("li")):
        links = item.find_all("a")
        if not any("gutenberg" in link.get_text(" ", strip=True).lower() for link in links):
            continue
        if top_toc_list is not None and item.find_parent("ol") is top_toc_list:
            direct_items = top_toc_list.find_all("li", recursive=False)
            if len(direct_items) <= 1:
                continue
        item.decompose()

    for ol in list(soup.find_all("ol")):
        if _is_top_level_toc_list(ol, top_toc_list):
            continue
        if not ol.find("li"):
            ol.decompose()


def _clean_title_text(text: str, book_title: str) -> str:
    if text[: len(_PG_TITLE_PREFIX)].lower() == _PG_TITLE_PREFIX.lower():
        text = text[len(_PG_TITLE_PREFIX) :]
    text = _PG_TITLE_SUFFIX.sub("", text)
    if "gutenberg" in text.lower():
        text = book_title
    return text


def _strip_gutenberg_xhtml(content: bytes, *, nav: bool = False, book_title: str = "") -> bytes:
    soup = BeautifulSoup(content, "lxml")
    _strip_boilerplate_elements(soup)

    if soup.head is not None:
        for meta in list(soup.head.find_all("meta")):
            if any("gutenberg" in str(value).lower() for value in meta.attrs.values()):
                meta.decompose()
        title = soup.head.find("title")
        if title is not None:
            text = title.get_text()
            cleaned = _clean_title_text(text, book_title)
            if cleaned != text:
                title.clear()
                title.append(cleaned)

    for link in list(soup.find_all("a", href=True)):
        if "gutenberg.org" in link["href"].lower():
            link.unwrap()

    if nav:
        _strip_gutenberg_navigation(soup, ncx=False)
    return _serialize_xhtml(content, soup)


def _fragment_targets(docs: dict[str, bytes]) -> set[str]:
    """Every ``#fragment`` linked from ``href`` or NCX ``src`` in ``docs``."""
    targets: set[str] = set()
    pattern = re.compile(rb"""(?:href|src)\s*=\s*["'][^"'#]*#([^"']+)["']""")
    for data in docs.values():
        targets.update(unquote(m.decode("utf-8", "replace")) for m in pattern.findall(data))
    return targets


def _drop_unreferenced_retained_ids(content: bytes, referenced: set[str]) -> bytes:
    soup = BeautifulSoup(content, "lxml")
    for span in soup.find_all(attrs={_RETAINED_ATTR: True}):
        if span.get("id") in referenced:
            del span[_RETAINED_ATTR]
        else:
            span.decompose()
    for kept in soup.find_all(attrs={_KEPT_PG_ID_ATTR: True}):
        del kept[_KEPT_PG_ID_ATTR]
        if kept.get("id") not in referenced:
            del kept["id"]
    return _serialize_xhtml(content, soup)


def _strip_gutenberg_ncx(content: bytes, book_title: str = "") -> bytes:
    soup = BeautifulSoup(content, "xml")
    _strip_gutenberg_navigation(soup, ncx=True)
    for meta in list(soup.find_all("meta")):
        # dtb:uid must stay (it is replaced with the new package id instead).
        if meta.get("name", "").lower() != "dtb:uid" and any(
            "gutenberg" in str(value).lower() for value in meta.attrs.values()
        ):
            meta.decompose()
    for text in soup.find_all("text"):
        if "gutenberg" in text.get_text().lower():
            cleaned = _clean_title_text(text.get_text(), book_title)
            text.clear()
            text.append(cleaned)
    return soup.encode(formatter="minimal")


def _strip_gutenberg_metadata(root: etree._Element) -> str | None:
    metadata = root.find(f"{{{_OPF_NS}}}metadata")
    if metadata is None:
        return None

    unique_id = root.get("unique-identifier")
    unique_identifier = next(
        (
            element
            for element in metadata
            if element.get("id") == unique_id
            and etree.QName(element).namespace == _DC_NS
            and etree.QName(element).localname == "identifier"
        ),
        None,
    )
    replacement_uid = None
    if unique_identifier is not None:
        old_value = "".join(unique_identifier.itertext())
        if "gutenberg" in old_value.lower():
            replacement_uid = "urn:uuid:" + str(
                uuid.uuid5(uuid.NAMESPACE_URL, old_value)
            )
            unique_identifier.text = replacement_uid
            for child in list(unique_identifier):
                unique_identifier.remove(child)

    remove = set()
    for element in list(metadata):
        qname = etree.QName(element)
        if not (
            qname.namespace == _DC_NS
            or (qname.namespace == _OPF_NS and qname.localname == "meta")
        ):
            continue
        if element is unique_identifier:
            continue
        if "gutenberg" in "".join(element.itertext()).lower():
            remove.add(element)

    # Remove metadata refinements (and refinements of those refinements) when
    # their referenced metadata element is removed.
    removed_ids = {element.get("id") for element in remove if element.get("id")}
    changed = True
    while changed:
        changed = False
        for element in list(metadata):
            refines = element.get("refines", "")
            if refines.startswith("#") and refines[1:] in removed_ids and element not in remove:
                remove.add(element)
                identifier = element.get("id")
                if identifier:
                    removed_ids.add(identifier)
                changed = True

    for element in remove:
        metadata.remove(element)
    return replacement_uid


def _manifest_paths(opf_root: etree._Element, opf_dir: str) -> dict[str, tuple[str, str, str]]:
    paths = {}
    for item in opf_root.findall(f".//{{{_OPF_NS}}}manifest/{{{_OPF_NS}}}item"):
        item_id = item.get("id")
        if not item_id:
            continue
        href = unquote(urlsplit(item.get("href", "")).path)
        path = posixpath.normpath(posixpath.join(opf_dir, href))
        paths[item_id] = (
            path,
            item.get("media-type", ""),
            item.get("properties", ""),
        )
    return paths


def _find_cover_item(
    opf_root: etree._Element, opf_dir: str
) -> tuple[str, str] | None:
    """``(zip path, media type)`` of the OPF cover image, or ``None``.

    The cover is the manifest item named by ``<meta name="cover" content="id">``
    or, failing that, the one carrying ``properties="cover-image"``.
    """
    cover_id = None
    metadata = opf_root.find(f"{{{_OPF_NS}}}metadata")
    if metadata is not None:
        for meta in metadata.findall(f"{{{_OPF_NS}}}meta"):
            if (meta.get("name") or "").lower() == "cover" and meta.get("content"):
                cover_id = meta.get("content")
                break

    items = {
        item.get("id"): item
        for item in opf_root.findall(f".//{{{_OPF_NS}}}manifest/{{{_OPF_NS}}}item")
        if item.get("id")
    }
    cover_item = items.get(cover_id) if cover_id else None
    if cover_item is None:
        cover_item = next(
            (
                item
                for item in items.values()
                if "cover-image" in (item.get("properties") or "").split()
            ),
            None,
        )
    if cover_item is None:
        return None

    href = unquote(urlsplit(cover_item.get("href", "")).path)
    path = posixpath.normpath(posixpath.join(opf_dir, href))
    return path, cover_item.get("media-type", "")


def find_gutenberg_cover(epub_path: str | Path) -> str | None:
    """Zip path of a Gutenberg-generated cover still in an EPUB, else ``None``.

    A cover we rendered ourselves (see :func:`kannada_epub.cover.render_cover`)
    is not a Gutenberg cover even though it shares the name and size, so it is
    excluded via its marker chunk.
    """
    with zipfile.ZipFile(epub_path) as zf:
        opf_path = find_opf_path(zf)
        try:
            opf_root = etree.fromstring(zf.read(opf_path))
        except (KeyError, etree.XMLSyntaxError):
            return None
        cover_item = _find_cover_item(opf_root, posixpath.dirname(opf_path))
        if cover_item is None:
            return None
        path, media_type = cover_item
        try:
            data = zf.read(path)
        except KeyError:
            return None
        if is_gutenberg_generated_cover(
            path, media_type, data
        ) and not is_translator_generated_cover(data):
            return path
    return None


def find_gutenberg_mentions(epub_path: str | Path) -> list[tuple[str, str]]:
    """Return one ``(zip entry, snippet)`` for every Gutenberg text match."""
    eligible = (".xhtml", ".html", ".htm", ".opf", ".ncx")
    matches: list[tuple[str, str]] = []
    with zipfile.ZipFile(epub_path) as zf:
        for name in zf.namelist():
            if not name.lower().endswith(eligible):
                continue
            text = zf.read(name).decode("utf-8", errors="replace")
            for match in re.finditer("gutenberg", text, re.IGNORECASE):
                midpoint = (match.start() + match.end()) // 2
                start = max(0, midpoint - 40)
                end = min(len(text), start + 80)
                start = max(0, end - 80)
                matches.append((name, text[start:end]))
    return matches


def _update_opf(
    opf_bytes: bytes,
    added_items: list[tuple[str, str, str]],
    language: TargetLanguage = LANGUAGES["kn"],
) -> bytes:
    """Add manifest entries, force the package language and add the
    machine-translation contributor, keeping the rest."""
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
            lang.text = language.key
    else:
        lang = etree.SubElement(metadata, f"{{{_DC_NS}}}language")
        lang.text = language.key

    contributor_text = machine_translation_contributor(language.name)
    contributors = metadata.findall(f"{{{_DC_NS}}}contributor")
    if not any(c.text == contributor_text for c in contributors):
        contributor = etree.SubElement(metadata, f"{{{_DC_NS}}}contributor")
        contributor.text = contributor_text

    return etree.tostring(root, xml_declaration=True, encoding="utf-8", pretty_print=True)


def write_translated_epub(
    source_epub: str | Path,
    translations: dict[str, dict[int, str]],
    output_path: str | Path,
    *,
    font_dir: str | Path | None = None,
    flagged: dict[str, set[int]] | None = None,
    strip_gutenberg: bool = False,
    cover_text: CoverText | None = None,
    language: TargetLanguage = LANGUAGES["kn"],
) -> None:
    """Repackage ``source_epub`` with translated paragraphs replaced in place.

    ``translations`` maps chapter id -> {``Paragraph.index``: translated text}.
    ``flagged`` optionally maps chapter id -> the set of ``Paragraph.index``
    values to mark with the ``qa-review-flag`` CSS class (QA flagged them for
    human review).
    ``strip_gutenberg`` removes Project Gutenberg boilerplate and metadata
    references from the output while retaining fragment targets. When it is
    set and the source's cover is one Project Gutenberg generated, the cover
    is replaced in place with a cover rendered from ``cover_text`` in
    ``language`` (falling back to the OPF's cleaned English title and
    creator). If rendering fails, the original cover is kept and a warning is
    emitted.
    ``language`` sets ``lang``/``xml:lang``, ``dc:language``, the stylesheet's
    font family/files and the machine-translation contributor. When its font
    files are not in ``font_dir``, no font is embedded (and a warning is
    emitted); the stylesheet keeps the family name with a serif fallback.
    """
    source_epub = Path(source_epub)
    output_path = Path(output_path)
    font_dir = Path(font_dir) if font_dir is not None else _default_font_dir()
    flagged_by_chapter = {cid: set(indices) for cid, indices in (flagged or {}).items()}

    font_bytes: dict[str, bytes] = {}
    font_regular_path = font_dir / language.font_regular
    font_bold_path = font_dir / language.font_bold
    embed_fonts = font_regular_path.exists() and font_bold_path.exists()
    if embed_fonts:
        for filename, path in (
            (language.font_regular, font_regular_path),
            (language.font_bold, font_bold_path),
            ("OFL.txt", font_dir / "OFL.txt"),
        ):
            if not path.exists():
                raise FileNotFoundError(f"font asset not found: {path}")
            font_bytes[filename] = path.read_bytes()
    else:
        warnings.warn(
            f"No bundled font for {language.name}; readers will use their own fonts.",
            RuntimeWarning,
            stacklevel=2,
        )

    replaced_cover_path: str | None = None
    replaced_cover_bytes: bytes | None = None

    with zipfile.ZipFile(source_epub, "r") as zin:
        infos = {info.filename: info for info in zin.infolist()}
        source_data = {name: zin.read(name) for name in infos}
        opf_path = find_opf_path(zin)

        opf_dir = posixpath.dirname(opf_path)
        opf_root = etree.fromstring(source_data[opf_path])
        manifest_paths = _manifest_paths(opf_root, opf_dir)
        manifest = _manifest_items(opf_root)

        # Names of the package files we add, relative to the OPF directory.
        css_rel = "css/kannada.css"
        css_zip_path = posixpath.join(opf_dir, css_rel)
        css_text = font_css_for(language) if embed_fonts else fallback_css_for(language)

        modified_docs: dict[str, bytes] = {}
        for chapter_id in sorted(set(translations) | set(flagged_by_chapter)):
            by_index = translations.get(chapter_id, {})
            item = manifest.get(chapter_id)
            if item is None:
                raise ValueError(
                    f"chapter {chapter_id!r} is not in the OPF manifest"
                )
            doc_path = posixpath.normpath(
                posixpath.join(opf_dir, unquote(item["href"]))
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
                language=language,
            )

        replacement_uid = None
        if strip_gutenberg:
            title_element = opf_root.find(f"{{{_OPF_NS}}}metadata/{{{_DC_NS}}}title")
            book_title = (
                "".join(title_element.itertext()).strip() if title_element is not None else ""
            )
            book_title = _clean_title_text(book_title, "") or book_title
            replacement_uid = _strip_gutenberg_metadata(opf_root)

            creator_element = opf_root.find(
                f"{{{_OPF_NS}}}metadata/{{{_DC_NS}}}creator"
            )
            creator = (
                "".join(creator_element.itertext()).strip()
                if creator_element is not None
                else ""
            )
            cover_item = _find_cover_item(opf_root, opf_dir)
            if cover_item is not None:
                cover_path, cover_media_type = cover_item
                cover_data = source_data.get(cover_path)
                if (
                    cover_data is not None
                    and is_gutenberg_generated_cover(
                        cover_path, cover_media_type, cover_data
                    )
                    and not is_translator_generated_cover(cover_data)
                ):
                    try:
                        text = cover_text or CoverText(
                            title_kn=None,
                            author_kn=None,
                            title_en=book_title,
                            author_en=creator or None,
                        )
                        replaced_cover_bytes = render_cover(
                            text, font_dir, language=language
                        )
                        replaced_cover_path = cover_path
                    except Exception as exc:  # noqa: BLE001 — never fail the book
                        warnings.warn(
                            f"Keeping the Project Gutenberg cover "
                            f"{cover_path!r}: {exc}",
                            RuntimeWarning,
                            stacklevel=2,
                        )

            for doc_path, media_type, properties in manifest_paths.values():
                if doc_path not in source_data:
                    continue
                if media_type == "application/xhtml+xml":
                    source = modified_docs.get(doc_path, source_data[doc_path])
                    modified_docs[doc_path] = _strip_gutenberg_xhtml(
                        source, nav="nav" in properties.split(), book_title=book_title
                    )
                elif media_type == _NCX_MEDIA_TYPE:
                    content = _strip_gutenberg_ncx(source_data[doc_path], book_title)
                    if replacement_uid:
                        soup = BeautifulSoup(content, "xml")
                        for meta in soup.find_all("meta"):
                            if meta.get("name", "").lower() == "dtb:uid":
                                meta["content"] = replacement_uid
                        content = soup.encode(formatter="minimal")
                    modified_docs[doc_path] = content

            referenced = _fragment_targets(modified_docs)
            for doc_path, data in list(modified_docs.items()):
                if _RETAINED_ATTR.encode() in data or _KEPT_PG_ID_ATTR.encode() in data:
                    modified_docs[doc_path] = _drop_unreferenced_retained_ids(data, referenced)

        if embed_fonts:
            added_items: list[tuple[str, str, str]] = [
                ("kannada-font-regular", f"fonts/{language.font_regular}", "font/ttf"),
                ("kannada-font-bold", f"fonts/{language.font_bold}", "font/ttf"),
                ("kannada-ofl", "fonts/OFL.txt", "text/plain"),
                ("kannada-css", css_rel, "text/css"),
            ]
        else:
            added_items = [("kannada-css", css_rel, "text/css")]
        added_bytes: dict[str, bytes] = {
            posixpath.join(opf_dir, "fonts", name): data
            for name, data in font_bytes.items()
        }
        added_bytes[css_zip_path] = css_text.encode("utf-8")

        opf_input = (
            etree.tostring(opf_root, encoding="utf-8")
            if strip_gutenberg
            else source_data[opf_path]
        )
        modified_opf = _update_opf(opf_input, added_items, language)

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
                if (
                    name == replaced_cover_path
                    and replaced_cover_bytes is not None
                ):
                    # Same zip path and media type; keep the source entry's
                    # compression so the replacement is otherwise identical.
                    out_info.compress_type = info.compress_type
                    zout.writestr(out_info, replaced_cover_bytes)
                    continue
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
