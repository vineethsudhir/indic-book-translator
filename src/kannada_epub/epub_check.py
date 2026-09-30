"""Warn when a source EPUB itself fails EPUBCheck-style checks.

The translated EPUB copies every untouched file byte-for-byte, so a defect in
the source carries over into the output. This module inspects the source for
the defects EPUBCheck reports that make users blame the translator, without
running Java or adding a dependency.

The checks are deliberately conservative: for anything ambiguous (unknown
image media types, a manifest item that is merely missing) we either skip the
item or let the existing ``epub_io`` error propagate, but we never guess that
a good file is bad. Results are stable — grouped by check, then zip path — so
callers can show and persist them reproducibly.
"""

from __future__ import annotations

import posixpath
import zipfile
from dataclasses import dataclass
from pathlib import Path

import lxml.etree as etree

from .epub_io import (
    _XML_PARSER,
    OPF_NS,
    _local_name,
    _read_opf,
    _resolve_zip_path,
)

_EPUB_MIMETYPE = "application/epub+zip"
_XHTML_MEDIA_TYPE = "application/xhtml+xml"
_NCX_MEDIA_TYPE = "application/x-dtbncx+xml"
_SVG_MEDIA_TYPE = "image/svg+xml"

# Friendly names for the media types we validate, keyed by the type's suffix.
_IMAGE_KINDS = {
    "image/png": ("PNG", ".png"),
    "image/jpeg": ("JPEG", ".jpg"),
    "image/gif": ("GIF", ".gif"),
    "image/webp": ("WebP", ".webp"),
    "image/svg+xml": ("SVG", ".svg"),
}
# Extensions that actually match each media type (JPEG accepts two spellings).
_IMAGE_EXTENSIONS = {
    "image/png": frozenset({".png"}),
    "image/jpeg": frozenset({".jpg", ".jpeg"}),
    "image/gif": frozenset({".gif"}),
    "image/webp": frozenset({".webp"}),
    "image/svg+xml": frozenset({".svg"}),
}


@dataclass(frozen=True)
class SourceProblem:
    """One defect in the source EPUB, described for a non-technical user."""

    path: str  # zip path the problem is in ("" for package-level)
    message: str  # one plain-English sentence


def _mimetype_problem(zf: zipfile.ZipFile) -> SourceProblem | None:
    """The first problem with the ``mimetype`` entry, or None if it's fine."""
    infos = zf.infolist()
    info = next((i for i in infos if i.filename == "mimetype"), None)
    if info is None:
        return SourceProblem("mimetype", "is missing from the EPUB")
    if infos and infos[0].filename != "mimetype":
        return SourceProblem("mimetype", "is not the first file in the EPUB")
    if info.compress_type != zipfile.ZIP_STORED:
        return SourceProblem(
            "mimetype", "is compressed; it must be stored without compression"
        )
    try:
        content = zf.read("mimetype")
    except KeyError:  # unreachable, but never raise on a defect we check for
        return SourceProblem("mimetype", "is missing from the EPUB")
    if content != _EPUB_MIMETYPE.encode("ascii"):
        return SourceProblem(
            "mimetype", "doesn't contain 'application/epub+zip'"
        )
    return None


def _png_ok(data: bytes) -> bool:
    return data.startswith(b"\x89PNG\r\n\x1a\n")


def _jpeg_ok(data: bytes) -> bool:
    return data.startswith(b"\xff\xd8\xff")


def _gif_ok(data: bytes) -> bool:
    return data.startswith((b"GIF87a", b"GIF89a"))


def _webp_ok(data: bytes) -> bool:
    return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP"


def _svg_root(data: bytes):
    """The parsed SVG root element, or None if it isn't valid SVG XML."""
    try:
        root = etree.fromstring(data, _XML_PARSER)
    except etree.XMLSyntaxError:
        return None
    return root if _local_name(root.tag) == "svg" else None


def _signature_ok(media_type: str, data: bytes) -> bool:
    if media_type == "image/png":
        return _png_ok(data)
    if media_type == "image/jpeg":
        return _jpeg_ok(data)
    if media_type == "image/gif":
        return _gif_ok(data)
    if media_type == "image/webp":
        return _webp_ok(data)
    if media_type == _SVG_MEDIA_TYPE:
        return _svg_root(data) is not None
    raise AssertionError(f"no signature rule for {media_type!r}")


def _duplicate_ids(root) -> list[tuple[str, int]]:
    """``(id, count)`` for every id used more than once, in first-appearance order."""
    counts: dict[str, int] = {}
    order: list[str] = []
    for element in root.iter():
        value = element.get("id")
        if value is None:
            continue
        if value not in counts:
            order.append(value)
        counts[value] = counts.get(value, 0) + 1
    return [(value, counts[value]) for value in order if counts[value] > 1]


def check_source_epub(path: str | Path) -> list[SourceProblem]:
    """Defects in a source EPUB, in a stable order (by check, then zip path).

    Never raises for a defect it checks for; a defect becomes a
    :class:`SourceProblem`. If the zip or package document can't be read at
    all, the existing ``epub_io`` error propagates unchanged.
    """
    problems: list[tuple[int, str, str]] = []

    with zipfile.ZipFile(path) as zf:
        opf_path, opf = _read_opf(zf)  # ValueError/BadZipFile propagate
        opf_dir = posixpath.dirname(opf_path)
        names = set(zf.namelist())

        # 1. mimetype -----------------------------------------------------
        mimetype_problem = _mimetype_problem(zf)
        if mimetype_problem is not None:
            problems.append((1, mimetype_problem.path, mimetype_problem.message))

        # 2..5. manifest items --------------------------------------------
        items = []
        seen_paths: set[str] = set()
        for item in opf.iterfind(f"{{{OPF_NS}}}manifest/{{{OPF_NS}}}item"):
            href = item.get("href") or ""
            if not href:
                continue
            resolved = _resolve_zip_path(opf_dir, href)
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            items.append((resolved, item.get("media-type") or ""))

        # 2. missing files -------------------------------------------------
        for resolved, _media_type in items:
            if resolved not in names:
                problems.append((
                    2,
                    resolved,
                    "is listed in the manifest but missing from the EPUB",
                ))

        # 3. image type ----------------------------------------------------
        image_problem_paths: set[str] = set()
        for resolved, media_type in items:
            if not media_type.startswith("image/"):
                continue
            if media_type not in _IMAGE_KINDS:
                continue  # unknown image media type: skip
            if resolved not in names:
                continue  # already reported as missing
            try:
                data = zf.read(resolved)
            except KeyError:
                continue
            if not _signature_ok(media_type, data):
                kind = _IMAGE_KINDS[media_type][0]
                problems.append((
                    3,
                    resolved,
                    f"is damaged or isn't really a {kind} image",
                ))
                image_problem_paths.add(resolved)
                continue
            extensions = _IMAGE_EXTENSIONS[media_type]
            extension = posixpath.splitext(resolved)[1].lower()
            if extension not in extensions:
                expected = _IMAGE_KINDS[media_type][1]
                problems.append((
                    3,
                    resolved,
                    f"has no {expected} file extension, so some readers and "
                    "EPUBCheck treat it as damaged",
                ))
                image_problem_paths.add(resolved)

        # 4. well-formed XML; 5. duplicate ids -----------------------------
        for resolved, media_type in items:
            if media_type not in (
                _XHTML_MEDIA_TYPE,
                _NCX_MEDIA_TYPE,
                _SVG_MEDIA_TYPE,
            ):
                continue
            if resolved not in names:
                continue
            # An SVG already reported by check 3 would otherwise get a second,
            # redundant "not well-formed XML" problem.
            if resolved in image_problem_paths:
                continue
            try:
                data = zf.read(resolved)
            except KeyError:
                continue
            try:
                root = etree.fromstring(data, _XML_PARSER)
            except etree.XMLSyntaxError:
                if media_type == _XHTML_MEDIA_TYPE:
                    message = "is not well-formed XHTML"
                elif media_type == _SVG_MEDIA_TYPE:
                    message = "is not well-formed SVG"
                else:
                    message = "is not well-formed XML"
                problems.append((4, resolved, message))
                continue  # not checked further
            if media_type in (_XHTML_MEDIA_TYPE, _NCX_MEDIA_TYPE):
                duplicates = _duplicate_ids(root)
                if duplicates:
                    first_id, first_count = duplicates[0]
                    message = f"uses the id '{first_id}' {first_count} times"
                    extra = sum(count - 1 for _value, count in duplicates[1:])
                    if extra:
                        message += f" and {extra} more"
                    problems.append((5, resolved, message))

    problems.sort(key=lambda problem: (problem[0], problem[1]))
    return [SourceProblem(path, message) for _check, path, message in problems]
