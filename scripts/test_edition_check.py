"""Tests for ``kannada_epub.edition_check`` (front-matter copyright risks).

Everything runs offline against small EPUB fixtures and the two real books in
the checkout. ``rajmohan.epub`` is git-ignored, so its case is skipped (with a
note) when the file is absent, e.g. in CI.

Run: .venv/bin/python scripts/test_edition_check.py
"""

import sys
import tempfile
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from epub_fixture import build_epub  # noqa: E402

from kannada_epub.edition_check import edition_notes  # noqa: E402
from kannada_epub.epub_io import load_epub_chapters  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TODAY = date(2026, 9, 30)
SHERLOCK = ROOT / "data" / "sherlock_holmes.epub"
RAJMOHAN = ROOT / ".omc" / "showcase" / "books" / "rajmohan.epub"


def _doc(title: str, paragraphs: list[str]) -> str:
    body = "".join(f"<p>{text}</p>" for text in paragraphs)
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<html xmlns="http://www.w3.org/1999/xhtml">'
        f"<head><title>{title}</title></head>"
        f"<body><h1>{title}</h1>{body}</body></html>"
    )


def _book(path: Path, chapters: list[tuple[str, str, list[str]]], date_value: str) -> None:
    documents = [
        {"id": chapter_id, "href": f"{chapter_id}.xhtml", "content": _doc(title, paragraphs)}
        for chapter_id, title, paragraphs in chapters
    ]
    build_epub(path, documents, extra_metadata=[f"<dc:date>{date_value}</dc:date>"])


def _notes(path: Path, **kwargs):
    kwargs.setdefault("today", TODAY)
    return edition_notes(path, load_epub_chapters(path), **kwargs)


def test_cutoff() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cutoff.epub"
        _book(path, [("ch1", "Chapter One", ["Hello."])], "1864")
        assert _notes(path).cutoff_year == 1931
        assert _notes(path, today=date(2027, 1, 1)).cutoff_year == 1932


def test_reprint_front_matter() -> None:
    """A Rajmohan-shaped book: 1864 date, 1935 imprint and preface in text."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "rajmohan.epub"
        _book(
            path,
            [(
                "c0",
                "Rajmohan's Wife",
                [
                    "RAJMOHAN'S WIFE A NOVEL",
                    "ByBankim Chandra Chatterjee",
                    "R. CHATTERJEE CALCUTTA. 1935",
                    "Published by K. N. Chatterjee, 120-2, Upper Circular Road",
                    "PREFACE",
                    "Strangely enough, Bengal's first great novelist made his debut here.",
                    "Brajendra Nath Banerji",
                ],
            )],
            "1864",
        )
        notes = _notes(path)
        assert notes.date == "1864", notes.date
        assert notes.needs_review
        assert notes.front_matter == ["Preface"], notes.front_matter
        assert len(notes.recent_years) == 1, notes.recent_years
        label, year, snippet = notes.recent_years[0]
        assert label == "Rajmohan's Wife", label
        assert year == 1935, year
        assert "1935" in snippet, snippet


def test_recent_date_alone_needs_review() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "recent.epub"
        _book(path, [("ch1", "Chapter One", ["Hello there."])], "1950")
        notes = _notes(path)
        assert notes.needs_review
        assert notes.recent_years == [] and notes.copyright_notices == []


def test_copyright_notice_before_cutoff() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "notice.epub"
        _book(path, [("ch1", "Chapter One", ["Copyright 1922 by Someone."])], "1864")
        notes = _notes(path)
        # Reported, but a notice dated before the cutoff has expired in the US.
        assert not notes.needs_review, notes
        assert len(notes.copyright_notices) == 1, notes.copyright_notices
        label, snippet = notes.copyright_notices[0]
        assert label == "Chapter One", label
        assert snippet.startswith("Copyright 1922"), snippet

    for text in ("Copyright 1950 by Someone.", "All rights reserved."):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "notice.epub"
            _book(path, [("ch1", "Chapter One", [text])], "1864")
            notes = _notes(path)
            assert notes.needs_review, (text, notes)

    # Gay-Neck's shape: an old dated notice with an undated line after it.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "notice.epub"
        _book(
            path,
            [("ch1", "Chapter One", ["Copyright, 1927 By E. P. Dutton", "All rights reserved"])],
            "1927",
        )
        notes = _notes(path)
        assert len(notes.copyright_notices) == 2, notes.copyright_notices
        assert not notes.needs_review, notes


def test_years_in_prose_ignored() -> None:
    """A long paragraph's four-digit count isn't an imprint year."""
    prose = (
        "At thirteen I wrote a drama of 2000 lines, and a long poem, in a fit "
        "of pique, which I have since destroyed along with much else besides."
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "prose.epub"
        _book(path, [("ch1", "Introduction", [prose])], "1905")
        notes = _notes(path)
        assert notes.recent_years == [], notes.recent_years
        assert not notes.needs_review, notes


def test_old_book_is_clean() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "old.epub"
        _book(
            path,
            [("ch1", "Chapter One", ["Written in 1890.", "Reprinted in 1900."])],
            "1890",
        )
        notes = _notes(path)
        assert not notes.needs_review, notes
        assert notes.recent_years == [] and notes.copyright_notices == []


def test_future_years_ignored() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "future.epub"
        _book(
            path,
            [("ch1", "Chapter One", ["The year 2150 is a typo.", "Coming in 2027."])],
            "1864",
        )
        notes = _notes(path)
        assert not notes.needs_review, notes
        assert notes.recent_years == [], notes.recent_years


def test_front_matter_chapter_after_first_three() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "titled.epub"
        _book(
            path,
            [
                ("c0", "Chapter One", ["Alpha."]),
                ("c1", "Chapter Two", ["Beta."]),
                ("c2", "Chapter Three", ["Gamma."]),
                ("c3", "Chapter Four", ["Delta."]),
                ("c4", "Introduction", ["First published in 1940.", "More text."]),
            ],
            "1864",
        )
        notes = _notes(path)
        assert "Introduction" in notes.front_matter, notes.front_matter
        assert any(year == 1940 for _label, year, _snippet in notes.recent_years), (
            notes.recent_years
        )


def test_real_books() -> None:
    assert SHERLOCK.is_file(), f"missing test book: {SHERLOCK}"
    sherlock = _notes(SHERLOCK)
    assert not sherlock.needs_review, sherlock

    if RAJMOHAN.is_file():
        rajmohan = _notes(RAJMOHAN)
        assert rajmohan.needs_review, rajmohan
        assert any(year == 1935 for _label, year, _snippet in rajmohan.recent_years), (
            rajmohan.recent_years
        )
    else:
        print("test_edition_check: rajmohan.epub absent; skipping the real-book case")


def main() -> None:
    test_cutoff()
    test_reprint_front_matter()
    test_recent_date_alone_needs_review()
    test_copyright_notice_before_cutoff()
    test_years_in_prose_ignored()
    test_old_book_is_clean()
    test_future_years_ignored()
    test_front_matter_chapter_after_first_three()
    test_real_books()
    print("test_edition_check: all assertions passed")


if __name__ == "__main__":
    main()
