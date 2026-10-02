"""Check the EPUB writer against a corpus of real books, without models.

For each EPUB given (files or folders of .epub files), this writes a
translated copy whose "translation" is every paragraph with its letters
replaced by a script letter of the target language (inline-markup markers
kept when --markup is on), then reports:

- what the book contains (EPUB version, tables, lists, footnote links,
  poetry-like blocks), so the corpus can be checked for coverage;
- problems ``check_source_epub`` finds in the source;
- EPUBCheck errors in the output that the source doesn't already have.

    .venv/bin/python scripts/check_corpus.py data/sherlock_holmes.epub .omc/pg

Outputs go to a temporary folder unless --out is given. The EPUBCheck
columns need Java and the EPUBCheck jar, found the way the app finds it
(``KANNADA_EPUBCHECK_JAR``, else the app data folder); without them those
columns say "n/a". The same book given twice (same name and size) is
checked once. Exits non-zero if any book adds EPUBCheck errors or fails to write.
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
import zipfile
from pathlib import Path

from bs4 import BeautifulSoup
from lxml import etree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kannada_epub.epub_check import check_source_epub  # noqa: E402
from kannada_epub.epub_io import load_epub_chapters  # noqa: E402
from kannada_epub.epub_writer import write_translated_epub  # noqa: E402
from kannada_epub.epubcheck_runner import (  # noqa: E402
    find_epubcheck,
    new_errors,
    run_epubcheck,
)
from kannada_epub.inline_markup import MARKER_RE  # noqa: E402
from kannada_epub.languages import LANGUAGES  # noqa: E402

_LETTER_RE = re.compile(r"[A-Za-z]")
_POETRY_CLASS_RE = re.compile(r"poem|verse|stanza|line", re.IGNORECASE)


def _fake(text: str, letter: str) -> str:
    """Replace letters with ``letter``, leaving inline-markup markers intact."""
    parts = []
    last = 0
    for match in MARKER_RE.finditer(text):
        parts.append(_LETTER_RE.sub(letter, text[last:match.start()]))
        parts.append(match.group(0))
        last = match.end()
    parts.append(_LETTER_RE.sub(letter, text[last:]))
    return "".join(parts)


def _features(path: Path) -> dict[str, object]:
    """Count the structures #16 asks the corpus to cover."""
    counts = {"version": "?", "tables": 0, "lists": 0, "footnotes": 0, "poetry": 0}
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        container = etree.fromstring(zf.read("META-INF/container.xml"))
        rootfile = container.find(".//{*}rootfile").get("full-path")
        opf = etree.fromstring(zf.read(rootfile))
        counts["version"] = opf.get("version", "?")
        for name in names:
            if not name.lower().endswith((".xhtml", ".html", ".htm")):
                continue
            soup = BeautifulSoup(zf.read(name), "lxml")
            counts["tables"] += len(soup.find_all("table"))
            counts["lists"] += len(soup.find_all(["ul", "ol"]))
            counts["footnotes"] += sum(
                1
                for a in soup.find_all("a", href=True)
                if "#" in a["href"]
                and (
                    "noteref" in (a.get("epub:type") or "")
                    or a.find_parent("sup") is not None
                    or a.find("sup") is not None
                    or re.fullmatch(r"\[?\*?\d{1,3}\]?", a.get_text(strip=True) or "")
                )
            )
            counts["poetry"] += sum(
                1
                for block in soup.find_all(["p", "div"])
                if _POETRY_CLASS_RE.search(" ".join(block.get("class") or []))
                or len(block.find_all("br", recursive=False)) >= 2
            )
    return counts


def check_book(path: Path, out_dir: Path, *, language_key: str, markup: bool, epubcheck: bool) -> dict:
    language = LANGUAGES[language_key]
    # The script's letter "a" sits five code points into each block (ಅ, அ, అ, അ, अ).
    letter = chr(int(re.search(r"\\u([0-9A-Fa-f]{4})", language.script_re).group(1), 16) + 5)
    row: dict[str, object] = {"book": path.name}
    row.update(_features(path))
    row["source_problems"] = len(check_source_epub(path))
    chapters = load_epub_chapters(path, exclude_ids=[])
    translations = {
        chapter.id: {
            p.index: _fake((p.marked_text or p.text) if markup else p.text, letter)
            for p in chapter.paragraphs
        }
        for chapter in chapters
    }
    row["chapters"] = len(chapters)
    row["paragraphs"] = sum(len(c.paragraphs) for c in chapters)
    row["marked"] = sum(1 for c in chapters for p in c.paragraphs if p.marked_text)
    output = out_dir / f"{path.stem}.{language.key}.epub"
    try:
        write_translated_epub(path, translations, output, strip_gutenberg=True, language=language)
    except Exception as exc:  # report and keep going through the corpus
        row["write"] = f"FAILED: {type(exc).__name__}: {exc}"
        return row
    row["write"] = "ok"
    if epubcheck:
        source_result = run_epubcheck(path)
        output_result = run_epubcheck(output)
        row["source_errors"] = source_result.errors + source_result.fatals
        row["new_errors"] = new_errors(output_result, source_result)
        row["new_messages"] = [
            m for m in output_result.messages if m not in set(source_result.messages)
        ][:5]
    return row


def _epubs(arguments: list[str]) -> list[Path]:
    paths: list[Path] = []
    for argument in arguments:
        path = Path(argument)
        if path.is_dir():
            paths.extend(sorted(p for p in path.glob("*.epub") if not p.stem.endswith("-old")))
        elif path.is_file():
            paths.append(path)
        else:
            print(f"skipping {argument}: not found", file=sys.stderr)
    seen: set[tuple[str, int]] = set()
    unique = []
    for path in paths:
        key = (path.name, path.stat().st_size)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("epubs", nargs="+", help="EPUB files or folders of EPUBs")
    parser.add_argument("--language", default="kn", choices=sorted(LANGUAGES))
    parser.add_argument("--markup", action="store_true", help="keep inline-markup markers")
    parser.add_argument("--out", help="folder for the written EPUBs (default: temporary)")
    args = parser.parse_args(argv)

    epubcheck = find_epubcheck() is not None
    if not epubcheck:
        print("EPUBCheck not found; skipping the EPUBCheck columns.", file=sys.stderr)
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(args.out) if args.out else Path(tmp)
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = [
            check_book(path, out_dir, language_key=args.language, markup=args.markup, epubcheck=epubcheck)
            for path in _epubs(args.epubs)
        ]

    header = ["book", "version", "chapters", "paragraphs", "marked", "tables", "lists",
              "footnotes", "poetry", "source_problems", "source_errors", "new_errors", "write"]
    print("\t".join(header))
    failed = False
    for row in rows:
        print("\t".join(str(row.get(key, "n/a")) for key in header))
        if row["write"] != "ok" or row.get("new_errors", 0):
            failed = True
        for message in row.get("new_messages", []) if row.get("new_errors") else []:
            print(f"    {message}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
