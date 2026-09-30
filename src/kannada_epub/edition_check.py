"""Point out edition metadata that may carry its own copyright.

Public-domain books are often only available as later reprints. The reprint's
front matter (a preface, introduction or notes) can be copyrighted even when
the original text is not, and the reprint year often lives inside the first
chapter rather than in ``dc:date``. This module collects the signals a user can
check before translating:

- the edition's ``dc:date`` (as written);
- front-matter headings found in the opening chapters or as chapter titles;
- four-digit years, in short imprint-like lines, recent enough that a US copyright may still apply, and
- explicit copyright notices.

It is advice, never a block: callers decide what to do with
:attr:`EditionNotes.needs_review`. An edition check must never stop a run or
an upload, so callers wrap :func:`edition_notes` and carry on when it raises.
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .epub_io import DC_NS, OPF_NS, Chapter, _read_opf

# Front-matter headings, normalised (case-folded, trailing punctuation and
# curly apostrophes removed) to the title-case form we report. A few spellings
# map to one kind so a book that uses both still reports it once.
_FRONT_MATTER = {
    "preface": "Preface",
    "foreword": "Foreword",
    "introduction": "Introduction",
    "introductory note": "Introductory Note",
    "prefatory note": "Prefatory Note",
    "editor's note": "Editor's Note",
    "editor's preface": "Editor's Preface",
    "translator's note": "Translator's Note",
    "translator's preface": "Translator's Preface",
    "publisher's note": "Publisher's Note",
    "note on the text": "Note on the Text",
    "notes": "Notes",
    "bibliographical note": "Bibliographical Note",
    "acknowledgements": "Acknowledgements",
    "acknowledgments": "Acknowledgements",
}

# A paragraph counts as a front-matter heading only when it is short; a whole
# preface paragraph also begins with the word "Preface" but is body text.
_FRONT_MATTER_MAX_CHARS = 60

# Years to look for in the text: 1500-1999 and 2000-2099. This deliberately
# excludes "2150"-style typos and years outside the plausible print range.
_YEAR_RE = re.compile(r"\b(1[5-9]\d\d|20\d\d)\b")

# Copyright notice shapes: the word, the symbol, an explicit "(c) 1922", or
# the reserved-rights phrase. Matched case-insensitively.
_COPYRIGHT_RE = re.compile(
    r"copyright|©|\(c\)\s*\d{4}|all rights reserved", re.IGNORECASE
)

# Years only count in short, imprint-like lines ("CALCUTTA. 1935",
# "Copyright, 1950"). In running prose a four-digit number is too often a
# count ("a drama of 2000 lines"), which flagged The Golden Threshold.
_IMPRINT_MAX_CHARS = 120

# How many opening chapters (and paragraphs within each) are scanned. Front
# matter almost always sits at the very front; a chapter *titled* like front
# matter is scanned in full wherever it appears.
_SCAN_CHAPTERS = 3
_SCAN_PARAGRAPHS = 60

# How many findings we keep (one per paragraph for years).
_MAX_FINDINGS = 5

# Trailing punctuation ignored when matching a heading ("Preface." -> "Preface").
_TRAILING_PUNCTUATION = ".:;,·…!?\"'\u2018\u2019\u201c\u201d"


@dataclass(frozen=True)
class EditionNotes:
    """What an edition's date and opening matter suggest about its rights.

    ``date`` is the first ``dc:date`` of the package document exactly as
    written. ``recent_years`` and ``copyright_notices`` are
    ``(chapter label, …)`` tuples so a user can find the evidence in the book.
    """

    date: str | None
    cutoff_year: int
    front_matter: list[str]
    recent_years: list[tuple[str, int, str]]
    copyright_notices: list[tuple[str, str]]
    # True if the latest dated copyright notice is on or after the cutoff, or
    # only undated notices exist. A notice dated before the cutoff
    # ("Copyright, 1927", with "All rights reserved" on the next line) has
    # expired in the US, so it is reported but doesn't by itself call for
    # review.
    recent_copyright_notice: bool = False

    @property
    def needs_review(self) -> bool:
        """True when the edition may include text still under copyright."""
        year = _edition_year(self.date)
        return (
            (year is not None and year >= self.cutoff_year)
            or bool(self.recent_years)
            or self.recent_copyright_notice
        )


def _copyright_cutoff(today: date) -> int:
    """The first year whose works may still be under US copyright.

    A work published in year ``Y`` enters the public domain 95 years later, so
    anything from ``today.year - 95`` on is still in copyright. This is the
    rule Wikisource states as "published before January 1, 1931" in 2026; it
    moves with the calendar, so it is never hard-coded.
    """
    return today.year - 95


def _edition_year(value: str | None) -> int | None:
    """The publication year of a ``dc:date``, or None if it is not one.

    Only a bare four-digit year is treated as the edition's publication year.
    E-text tools such as Project Gutenberg put their generation timestamp in
    ``dc:date`` (``1999-03-01``); reading that as a publication year would flag
    every public-domain e-text, so a date carrying a month or day is ignored
    for the copyright check. The raw value is still shown to the user.
    """
    if value is None:
        return None
    text = value.strip()
    if len(text) == 4 and text.isdigit():
        return int(text)
    return None


def _first_date(path: str | Path) -> str | None:
    """The first ``dc:date`` in the package document, as written."""
    with zipfile.ZipFile(path) as zf:
        _, opf = _read_opf(zf)
    metadata = opf.find(f"{{{OPF_NS}}}metadata")
    if metadata is None:
        return None
    element = metadata.find(f"{{{DC_NS}}}date")
    text = (element.text or "").strip() if element is not None else ""
    return text or None


def _normalise_heading(text: str) -> str:
    """Case-fold a heading and drop its trailing punctuation and quotes."""
    collapsed = " ".join(text.split()).strip()
    collapsed = collapsed.replace("\u2019", "'").replace("\u02bc", "'")
    return collapsed.rstrip(_TRAILING_PUNCTUATION).strip().casefold()


def _front_matter_kind(text: str) -> str | None:
    """The reported front-matter kind for a heading, or None."""
    return _FRONT_MATTER.get(_normalise_heading(text))


def _snippet(text: str) -> str:
    """Whitespace-collapsed paragraph text, cut at 80 characters with "…"."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= 80:
        return collapsed
    return collapsed[:79] + "…"


def edition_notes(
    path: str | Path,
    chapters: list[Chapter],
    *,
    today: date | None = None,
) -> EditionNotes:
    """Collect the edition's copyright signals from its metadata and text.

    ``chapters`` are the chapters exactly as
    :func:`kannada_epub.epub_io.load_epub_chapters` returned them (so any
    Gutenberg/Wikisource boilerplate is already gone). The first three
    chapters and their first sixty paragraphs are scanned, plus every
    paragraph of a chapter whose title reads like front matter.
    """
    today = today or date.today()
    cutoff = _copyright_cutoff(today)

    front_matter: list[str] = []
    recent_years: list[tuple[str, int, str]] = []
    copyright_notices: list[tuple[str, str]] = []
    notice_years: list[int] = []
    undated_notice = False

    for index, chapter in enumerate(chapters):
        title = chapter.title
        title_kind = _front_matter_kind(title) if title else None
        scan_all = title_kind is not None
        if not scan_all and index >= _SCAN_CHAPTERS:
            continue
        if title_kind and title_kind not in front_matter:
            front_matter.append(title_kind)

        label = chapter.title or chapter.id
        paragraphs = (
            chapter.paragraphs if scan_all else chapter.paragraphs[:_SCAN_PARAGRAPHS]
        )
        for paragraph in paragraphs:
            text = paragraph.text
            if len(text) <= _FRONT_MATTER_MAX_CHARS:
                kind = _front_matter_kind(text)
                if kind and kind not in front_matter:
                    front_matter.append(kind)
            if len(recent_years) < _MAX_FINDINGS and len(text) <= _IMPRINT_MAX_CHARS:
                for match in _YEAR_RE.finditer(text):
                    year = int(match.group(1))
                    if cutoff <= year <= today.year:
                        recent_years.append((label, year, _snippet(text)))
                        break
            if _COPYRIGHT_RE.search(text):
                if len(copyright_notices) < _MAX_FINDINGS:
                    copyright_notices.append((label, _snippet(text)))
                years = [int(y) for y in _YEAR_RE.findall(text) if int(y) <= today.year]
                notice_years.extend(years)
                undated_notice = undated_notice or not years

    recent_copyright_notice = (
        max(notice_years) >= cutoff if notice_years else undated_notice
    )
    return EditionNotes(
        date=_first_date(path),
        cutoff_year=cutoff,
        front_matter=front_matter,
        recent_years=recent_years,
        copyright_notices=copyright_notices,
        recent_copyright_notice=recent_copyright_notice,
    )


def edition_payload(notes: EditionNotes) -> dict:
    """The JSON shape used by the API response and the run manifest."""
    return {
        "date": notes.date,
        "cutoff_year": notes.cutoff_year,
        "front_matter": list(notes.front_matter),
        "recent_years": [
            {"chapter": label, "year": year, "snippet": snippet}
            for label, year, snippet in notes.recent_years
        ],
        "copyright_notices": [
            {"chapter": label, "snippet": snippet}
            for label, snippet in notes.copyright_notices
        ],
        "needs_review": notes.needs_review,
    }
