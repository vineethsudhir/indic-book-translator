"""Inline-markup markers shared by the reader, the writer and the pipeline.

FR-1.3 preserves bold/italic/link markup inside a translated paragraph. The
reader (:mod:`kannada_epub.epub_io`) walks a block's marked descendants and
inserts ``⟦n⟧`` before and ``⟦/n⟧`` after each one. The translation engine and
the consistency editor carry those markers through to the Kannada text, and the
writer (:mod:`kannada_epub.epub_writer`) parses them back into elements.

This module holds the marker grammar and the two pure helpers both sides use, so
the reader and the writer can only ever agree on what a marker is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ``⟦n⟧`` opens marker ``n``; ``⟦/n⟧`` closes it. ``n`` is 1-based and matches
# the position of the element in ``epub_io._marked_elements``.
MARKER_RE = re.compile(r"⟦/?(\d+)⟧")


@dataclass
class MarkerSpan:
    """One parsed ``⟦n⟧ … ⟦/n⟧`` span and its children.

    ``children`` mixes ``str`` text and nested :class:`MarkerSpan` objects, so
    the tree mirrors the source nesting.
    """

    marker_id: int
    children: list


def strip_markers(text: str) -> str:
    """Remove every ``⟦n⟧``/``⟦/n⟧`` marker and collapse the whitespace left.

    On a paragraph with no markers this is still ``" ".join(text.split())``.
    """
    return " ".join(MARKER_RE.sub("", text).split())


def parse_markers(text: str, allowed_ids) -> list | None:
    """Parse ``text`` into a list of ``str``/``MarkerSpan`` segments.

    Returns ``None`` when the markers are malformed: an id outside
    ``allowed_ids``, an id opened twice, a close with no matching open, spans
    that cross, or spans left unclosed at the end. Ids that were present in the
    source but are absent from ``text`` are allowed — that span is simply lost.

    ``allowed_ids`` is the set of 1-based marker numbers the source block
    produced (see ``epub_io._marked_elements``).
    """
    allowed = set(allowed_ids)
    root: list = []
    stack: list[tuple[int, list]] = []
    used: set[int] = set()
    position = 0

    for match in MARKER_RE.finditer(text):
        chunk = text[position : match.start()]
        if chunk:
            (stack[-1][1] if stack else root).append(chunk)
        position = match.end()

        marker_id = int(match.group(1))
        if marker_id not in allowed:
            return None
        if match.group(0).startswith("⟦/"):
            if not stack or stack[-1][0] != marker_id:
                return None
            _, children = stack.pop()
            span = MarkerSpan(marker_id, children)
            (stack[-1][1] if stack else root).append(span)
        else:
            if marker_id in used:
                return None
            used.add(marker_id)
            stack.append((marker_id, []))

    tail = text[position:]
    if tail:
        (stack[-1][1] if stack else root).append(tail)
    if stack:
        return None
    return root
