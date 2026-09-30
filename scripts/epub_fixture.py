"""Build small EPUB 3 fixtures for the offline tests, with only the stdlib.

Each document is a dict with ``id``, ``href`` and ``content`` (XHTML string),
and optionally ``media_type``, ``properties``, ``linear`` (``"no"`` to mark it
non-linear) and ``in_spine`` (``False`` to list it only in the manifest).
"""

import zipfile
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

CONTAINER_XML = """<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="{opf_path}" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""

NAV_XHTML = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
<head><title>Contents</title></head>
<body><nav epub:type="toc"><ol><li><a href="{href}">Start</a></li></ol></nav></body>
</html>
"""


def build_ncx(points: list[tuple[str, str]]) -> str:
    """A minimal EPUB 2 NCX with one top-level navPoint per ``(label, src)``."""
    nav_points = "".join(
        f'<navPoint id="np{i + 1}" playOrder="{i + 1}">'
        f"<navLabel><text>{escape(label)}</text></navLabel>"
        f"<content src={quoteattr(src)}/></navPoint>"
        for i, (label, src) in enumerate(points)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">'
        "<head/><docTitle><text>Fixture</text></docTitle>"
        f"<navMap>{nav_points}</navMap></ncx>"
    )


def build_epub(
    path: Path,
    documents: list[dict],
    *,
    title: str | None = "Fixture",
    creator: str | None = "Tester",
    opf_dir: str = "EPUB",
    identifier: str = "fixture-001",
    extra_metadata: list[str] | None = None,
    nav_content: str | None = None,
    extra_spine: list[str] | None = None,
    ncx_content: str | None = None,
    include_nav: bool = True,
    version: str = "3.0",
    mimetype_content: str = "application/epub+zip",
    mimetype_compressed: bool = False,
    mimetype_first: bool = True,
    include_mimetype: bool = True,
) -> None:
    """Write an EPUB at ``path`` with ``documents`` plus a non-linear nav.

    ``extra_spine`` appends raw ``idref`` values to the spine after the
    documents, e.g. to reproduce exports that list the same document twice.
    ``ncx_content`` writes an EPUB 2 NCX (``toc.ncx``) and points the spine's
    ``toc`` attribute at it; ``include_nav=False`` omits the EPUB 3 nav
    entirely, giving an EPUB 2-style book. ``mimetype_content``,
    ``mimetype_compressed`` and ``mimetype_first`` deliberately corrupt the
    ``mimetype`` entry for the source-check tests.
    """
    manifest: list[str] = []
    spine: list[str] = []
    if include_nav:
        manifest.append(
            '<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" '
            'properties="nav"/>'
        )
        spine.append('<itemref idref="nav" linear="no"/>')
    if ncx_content is not None:
        manifest.append(
            '<item id="ncx" href="toc.ncx" '
            'media-type="application/x-dtbncx+xml"/>'
        )
    for doc in documents:
        attrs = (
            f'id={quoteattr(doc["id"])} href={quoteattr(doc["href"])} '
            f'media-type={quoteattr(doc.get("media_type", "application/xhtml+xml"))}'
        )
        if doc.get("properties"):
            attrs += f' properties={quoteattr(doc["properties"])}'
        manifest.append(f"<item {attrs}/>")
        if doc.get("in_spine", True):
            linear = f' linear="{doc["linear"]}"' if doc.get("linear") else ""
            spine.append(f'<itemref idref={quoteattr(doc["id"])}{linear}/>')
    for idref in extra_spine or []:
        spine.append(f"<itemref idref={quoteattr(idref)}/>")

    metadata = [
        f'<dc:identifier id="uid">{escape(identifier)}</dc:identifier>',
        "<dc:language>en</dc:language>",
        '<meta property="dcterms:modified">2026-01-01T00:00:00Z</meta>',
    ]
    if title is not None:
        metadata.append(f"<dc:title>{escape(title)}</dc:title>")
    if creator is not None:
        metadata.append(f"<dc:creator>{escape(creator)}</dc:creator>")
    metadata.extend(extra_metadata or [])

    spine_attrs = ' toc="ncx"' if ncx_content is not None else ""
    opf = f"""<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="{version}" unique-identifier="uid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    {"".join(metadata)}
  </metadata>
  <manifest>{"".join(manifest)}</manifest>
  <spine{spine_attrs}>{"".join(spine)}</spine>
</package>
"""
    prefix = f"{opf_dir}/" if opf_dir else ""
    first_href = documents[0]["href"] if documents else "nav.xhtml"
    mimetype_type = (
        zipfile.ZIP_DEFLATED if mimetype_compressed else zipfile.ZIP_STORED
    )
    with zipfile.ZipFile(path, "w") as zf:
        if mimetype_first and include_mimetype:
            zf.writestr("mimetype", mimetype_content, compress_type=mimetype_type)
        zf.writestr(
            "META-INF/container.xml",
            CONTAINER_XML.format(opf_path=f"{prefix}content.opf"),
            compress_type=zipfile.ZIP_DEFLATED,
        )
        zf.writestr(f"{prefix}content.opf", opf, compress_type=zipfile.ZIP_DEFLATED)
        if include_nav:
            zf.writestr(
                f"{prefix}nav.xhtml",
                nav_content if nav_content is not None else NAV_XHTML.format(href=escape(first_href)),
                compress_type=zipfile.ZIP_DEFLATED,
            )
        if ncx_content is not None:
            zf.writestr(f"{prefix}toc.ncx", ncx_content, compress_type=zipfile.ZIP_DEFLATED)
        for doc in documents:
            if doc.get("write", True):
                zf.writestr(
                    f"{prefix}{doc.get('zip_name', doc['href'])}",
                    doc["content"],
                    compress_type=zipfile.ZIP_DEFLATED,
                )
        if not mimetype_first and include_mimetype:
            zf.writestr("mimetype", mimetype_content, compress_type=mimetype_type)