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
) -> None:
    """Write an EPUB 3 at ``path`` with ``documents`` plus a non-linear nav.

    ``extra_spine`` appends raw ``idref`` values to the spine after the
    documents, e.g. to reproduce exports that list the same document twice.
    """
    manifest = [
        '<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" '
        'properties="nav"/>'
    ]
    spine = ['<itemref idref="nav" linear="no"/>']
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

    opf = f"""<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="uid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    {"".join(metadata)}
  </metadata>
  <manifest>{"".join(manifest)}</manifest>
  <spine>{"".join(spine)}</spine>
</package>
"""
    prefix = f"{opf_dir}/" if opf_dir else ""
    first_href = documents[0]["href"] if documents else "nav.xhtml"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr(
            "META-INF/container.xml",
            CONTAINER_XML.format(opf_path=f"{prefix}content.opf"),
            compress_type=zipfile.ZIP_DEFLATED,
        )
        zf.writestr(f"{prefix}content.opf", opf, compress_type=zipfile.ZIP_DEFLATED)
        zf.writestr(
            f"{prefix}nav.xhtml",
            nav_content if nav_content is not None else NAV_XHTML.format(href=escape(first_href)),
            compress_type=zipfile.ZIP_DEFLATED,
        )
        for doc in documents:
            if doc.get("write", True):
                zf.writestr(
                    f"{prefix}{doc.get('zip_name', doc['href'])}",
                    doc["content"],
                    compress_type=zipfile.ZIP_DEFLATED,
                )
