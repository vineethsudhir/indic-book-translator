"""Import plain text, OCR text or cleaned HTML into a translatable EPUB 3.

Public-domain books often exist only as an archive.org OCR text dump or as an
HTML transcription (e.g. a Lehigh Scalar page). This module turns either into
ordered :class:`ImportedChapter` objects and can package them as a minimal,
valid EPUB 3 with :func:`build_epub`.

Everything is stdlib + ``bs4``/``lxml``; there is no network access and no
per-book configuration. The rules are deliberately general so the same code
handles every source. When something is ambiguous the importer records a
plain-English note in :attr:`ImportResult.warnings` for the user to review in
the preview rather than guessing.
"""

from __future__ import annotations

import os
import re
import tempfile
import time
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

from bs4 import BeautifulSoup

from .epub_io import BLOCK_TAGS

# Characters that end a sentence well enough that a following paragraph is a
# genuinely new paragraph, not the tail of one split by a page break.
_END_PUNCT = '.!?"\'”’:;)—'

_SYSTEM_WORDS_PATH = "/usr/share/dict/words"
_system_words_cache: set[str] | None = None

_PAGE_NUMBER = re.compile(r"^[\divxlIl]{1,4}$")
_ROMAN_ONLY = re.compile(r"^[IVXL]+\.?$")
_CHAPTER_MARKER = re.compile(
    r"(?i)^((?:CHAPTER|CHAP\.?)\s+(?:[IVXL]+|\d+)\.?)(?:\s+\S.*)?$"
)
_BOOK_MARKER = re.compile(r"(?i)^((?:BOOK|PART)\s+\d+\.?)(?:\s+\S.*)?$")
_CONTENTS_HEADING = re.compile(r"(?i)(?:table\s+of\s+contents|contents)")
_CONTENTS_ENTRY = re.compile(
    r"^(?P<title>.{2,}?)\s+(?P<page>[0-9ivxlIVXL]{1,5})\.?$"
)
_QUOTE_JUNK = re.compile(r"''|'\^|\^'|\*'|'<|'\*")
_WRAP_PUNCT = tuple('.!?"\'”’):;')


@dataclass
class ImportedChapter:
    """One chapter of imported text, ready to be translated."""

    title: str
    paragraphs: list[str]


@dataclass
class ImportResult:
    """The output of importing a source document.

    ``warnings`` are plain-English notes for the preview (dropped running
    headers, headings that were never found, suspicious chapter openings).
    ``detected_headings`` lists the headings the importer found for itself
    (e.g. from a contents list), so the UI can offer them for editing.
    """

    chapters: list[ImportedChapter]
    warnings: list[str]
    detected_headings: list[str]


# --- small text helpers ----------------------------------------------------


def _system_words() -> set[str]:
    """Words from the system dictionary, if one exists (not on Windows)."""
    global _system_words_cache
    if _system_words_cache is None:
        try:
            with open(_SYSTEM_WORDS_PATH, encoding="utf-8", errors="ignore") as handle:
                _system_words_cache = {
                    line.strip().lower() for line in handle if line.strip()
                }
        except OSError:
            _system_words_cache = set()
    return _system_words_cache


def _normalise_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")


def _normalise_line(line: str) -> str:
    return " ".join(line.replace("\xa0", " ").split())


def _norm_letters(text: str) -> str:
    """Letters-only, uppercase form used to compare headings and paragraphs."""
    return re.sub(r"[^A-Z]", "", text.upper())


def _book_vocabulary(text: str) -> set[str]:
    """Every word the book itself spells out unhyphenated."""
    return {word.lower() for word in re.findall(r"[A-Za-z]+", text)}


def _dehyphenate(text: str, known: set[str]) -> str:
    """Join a line-break hyphen ("re- cently") when the joined word is known."""

    def fix(match: re.Match[str]) -> str:
        joined = match.group(1) + match.group(2)
        lowered = joined.lower()
        if lowered in known or lowered.rstrip("s") in known:
            return joined
        return f"{match.group(1)}-{match.group(2)}"

    return re.sub(r"\b([A-Za-z]+)- +([a-z]+)\b", fix, text)


def _keep_paragraph(paragraph: str) -> bool:
    return len(paragraph) > 1 and not re.fullmatch(r"[\W\d]+", paragraph)


def _is_front_garbage(paragraph: str) -> bool:
    """True for the OCR junk lines that sit before a book's real start."""
    chars = [c for c in paragraph if not c.isspace()]
    if not chars:
        return True
    letters = sum(c.isalpha() for c in chars)
    if letters == 0:
        return True
    if letters / len(chars) < 0.5:
        return True
    return not chars[0].isalnum() and len(chars) < 8


def _blank_line_blocks(lines: list[str]) -> list[list[str]]:
    blocks: list[list[str]] = []
    buffer: list[str] = []
    for line in lines:
        if line:
            buffer.append(line)
        elif buffer:
            blocks.append(buffer)
            buffer = []
    if buffer:
        blocks.append(buffer)
    return blocks


# --- text extraction -------------------------------------------------------


def _plain_paragraphs(text: str) -> list[str]:
    """Paragraphs from ordinary text: blank-line separated, else one per line."""
    normalised = _normalise_text(text)
    if re.search(r"\n[^\S\n]*\n", normalised):
        blocks = re.split(r"\n[^\S\n]*\n+", normalised)
    else:
        blocks = normalised.split("\n")
    paragraphs = [" ".join(block.split()) for block in blocks]
    return [p for p in paragraphs if p]


def _is_header_candidate(line: str) -> bool:
    """A short, mostly-uppercase line plausibly used as a running head."""
    if not line or len(line) > 60:
        return False
    letters = re.sub(r"[^A-Za-z]", "", line)
    if len(letters) < 3:
        return False
    return sum(c.isupper() for c in letters) >= 0.9 * len(letters)


def _fix_ocr(text: str) -> str:
    text = _QUOTE_JUNK.sub('"', text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r'"\s+"', '" "', text)
    return " ".join(text.split()).strip()


def _ocr_paragraphs(text: str) -> tuple[list[str], list[str]]:
    """Paragraphs from archive.org OCR text, plus warnings."""
    normalised = _normalise_text(text)
    lines = [_normalise_line(line) for line in normalised.split("\n")]

    # Drop page-number-only lines ("12", "xiv", "l2").
    content = [line for line in lines if not (line and _PAGE_NUMBER.match(line))]

    # Automatic running page headers: a short mostly-uppercase line whose
    # letters-only form repeats at least three times in the book.
    counts: dict[str, int] = {}
    for line in content:
        if _is_header_candidate(line):
            key = _norm_letters(line)
            counts[key] = counts.get(key, 0) + 1
    running = {key for key, count in counts.items() if count >= 3}

    # Drop running-head lines, but keep each key's first occurrence: a book
    # prints the running title on the chapter's own opening page too, and
    # dropping that line would lose the chapter heading (e.g. "WHAT HAS INDIA
    # CONTRIBUTED" / "TO HUMAN WELFARE?").
    kept: list[str] = []
    dropped = 0
    seen_running: set[str] = set()
    for line in content:
        if line and _is_header_candidate(line):
            key = _norm_letters(line)
            if key in running:
                if key in seen_running:
                    dropped += 1
                    continue
                seen_running.add(key)
        kept.append(line)

    if re.search(r"\n[^\S\n]*\n", normalised):
        blocks = _blank_line_blocks(kept)
    else:
        blocks = [[line] for line in kept if line]

    paragraphs = [_fix_ocr(" ".join(block)) for block in blocks]

    # Rejoin a paragraph that a page break split in two: the first half does
    # not end a sentence and the second starts lowercase or with a comma/semi.
    merged: list[str] = []
    for paragraph in paragraphs:
        if not paragraph:
            continue
        if (
            merged
            and not merged[-1].endswith(tuple(_END_PUNCT))
            and (paragraph[:1].islower() or paragraph[:1] in ",;")
        ):
            merged[-1] = merged[-1] + " " + paragraph
        else:
            merged.append(paragraph)

    result = [p for p in merged if _keep_paragraph(p)]
    while result and _is_front_garbage(result[0]):
        result.pop(0)

    warnings: list[str] = []
    if dropped:
        warnings.append(
            f"Dropped {dropped} running page header line(s) repeated at least "
            "3 times."
        )
    return result, warnings


def _find_main_container(soup: BeautifulSoup):
    """The element holding the book: Scalar's ``sioc:content`` span, else densest."""
    scope = soup.body or soup
    special = scope.find(attrs={"property": "sioc:content"})
    if special is not None and " ".join(special.get_text().split()):
        return special

    def text_length(element) -> int:
        return len(" ".join(element.get_text().split()))

    best = scope
    best_length = text_length(scope)
    best_depth = len(list(scope.parents))
    for element in scope.find_all(True):
        length = text_length(element)
        depth = len(list(element.parents))
        if length > best_length or (length == best_length and depth > best_depth):
            best, best_length, best_depth = element, length, depth
    return best


def _is_hard_wrapped(lines: list[str]) -> bool:
    """Heuristic: most medium-length lines end without punctuation."""
    considered = [line for line in lines if 50 <= len(line) <= 90]
    if len(considered) < 3:
        return False
    no_end = sum(1 for line in considered if not line.endswith(_WRAP_PUNCT))
    return no_end >= 0.6 * len(considered)


def _html_paragraphs(container) -> list[str]:
    if container.find(BLOCK_TAGS) is None:
        for br in container.find_all("br"):
            br.replace_with("\n")
        raw = container.get_text().replace("\xa0", " ")
        blocks: list[list[str]] = []
        all_lines: list[str] = []
        for chunk in re.split(r"\n[^\S\n]*\n+", raw):
            lines = [line.strip() for line in chunk.split("\n") if line.strip()]
            if lines:
                blocks.append(lines)
                all_lines.extend(lines)
        if _is_hard_wrapped(all_lines):
            paragraphs = [" ".join(lines) for lines in blocks]
        else:
            paragraphs = [line for lines in blocks for line in lines]
        return [" ".join(p.split()) for p in paragraphs if " ".join(p.split())]

    paragraphs = []
    for tag in container.find_all(BLOCK_TAGS):
        if tag.find(BLOCK_TAGS) is not None:
            continue
        text = " ".join(tag.get_text().split())
        if text:
            paragraphs.append(text)
    return paragraphs


# --- chapters --------------------------------------------------------------


def _detect_contents(
    paragraphs: list[str],
) -> tuple[list[str], int | None]:
    """Find a front-matter contents list; return ``(titles, index_after_list)``."""
    for index, paragraph in enumerate(paragraphs):
        if not _CONTENTS_HEADING.fullmatch(paragraph.strip()):
            continue
        cursor = index + 1
        while cursor < len(paragraphs) and paragraphs[cursor].strip().upper() in (
            "PAGE",
            "PAGE.",
            "",
        ):
            cursor += 1
        titles: list[str] = []
        while cursor < len(paragraphs):
            match = _CONTENTS_ENTRY.match(paragraphs[cursor].strip())
            if not match:
                break
            title = match.group("title").strip(" .·•\u2026\t")
            if title:
                titles.append(title)
            cursor += 1
        if len(titles) >= 3:
            return titles, cursor
    return [], None


def _pattern_heading(paragraph: str) -> str | None:
    p = paragraph.strip()
    if not p:
        return None
    match = _CHAPTER_MARKER.match(p)
    if match:
        return match.group(1).strip()
    match = _BOOK_MARKER.match(p)
    if match:
        return match.group(1).strip()
    if _ROMAN_ONLY.match(p):
        return p.rstrip(".")
    return None


def _pattern_headings(paragraphs: list[str]) -> list[str]:
    headings = []
    for paragraph in paragraphs:
        heading = _pattern_heading(paragraph)
        if heading:
            headings.append(heading)
    return headings


def _heading_key(heading: str) -> str:
    """Letters-only prefix used to match a heading against body paragraphs."""
    return _norm_letters(heading)[:24]


def _heading_remainder(paragraph: str, heading: str) -> str:
    """Text after a heading run into its chapter's first paragraph."""
    if len(paragraph) <= 130:
        return ""
    words = re.findall(r"[A-Za-z0-9]+", heading)
    if not words:
        return ""
    pattern = r"\s*".join(re.escape(word) for word in words)
    match = re.match(pattern, paragraph, re.IGNORECASE)
    if not match:
        return ""
    return paragraph[match.end():].lstrip(" .,;:!?\"'”’—-\t").strip()


def _split_chapters(
    paragraphs: list[str], headings: list[str], start: int
) -> tuple[list[tuple[str, list[str]]], list[str]]:
    """Slice ``paragraphs`` at the headings, in order, starting at ``start``."""
    total = len(paragraphs)
    positions: list[tuple[int, str]] = []
    missing: list[str] = []
    cursor = start
    for heading in headings:
        key = _heading_key(heading)
        if not key:
            missing.append(heading)
            continue
        roman_only = bool(_ROMAN_ONLY.match(heading.strip()))
        found = total
        while cursor < total:
            candidate = _norm_letters(paragraphs[cursor])
            matched = candidate == key if roman_only else candidate.startswith(key)
            if matched:
                found = cursor
                break
            cursor += 1
        if found < total:
            positions.append((found, heading))
            cursor = found + 1
        else:
            missing.append(heading)

    built: list[tuple[str, list[str]]] = []
    first = positions[0][0] if positions else total
    front = paragraphs[:first]
    if front:
        built.append(("Front matter", front))
    for order, (position, heading) in enumerate(positions):
        end = positions[order + 1][0] if order + 1 < len(positions) else total
        body = list(paragraphs[position + 1 : end])
        remainder = _heading_remainder(paragraphs[position], heading)
        if remainder:
            body.insert(0, remainder)
        built.append((heading, body))
    return built, missing


def _suspicious_openings(
    chapters: list[tuple[str, list[str]]], known: set[str]
) -> list[str]:
    """Warn about chapter openings that look like misread drop caps."""
    warnings = []
    for title, paragraphs in chapters:
        if title == "Front matter" or not paragraphs:
            continue
        tokens = paragraphs[0].split()
        if not tokens:
            continue
        token = re.sub(r"[^A-Za-z0-9]", "", tokens[0])
        if len(token) <= 1 or token.lower() in known:
            continue
        if re.fullmatch(r"[A-Z0-9]+", token):
            snippet = " ".join(tokens[:3]) + " …"
            warnings.append(
                f'Chapter "{title}" starts with "{snippet}" — check the first '
                "word."
            )
    return warnings


def _assemble(
    paragraphs: list[str],
    headings: list[str] | None,
    system_words: set[str],
    base_warnings: list[str],
) -> ImportResult:
    warnings = list(base_warnings)
    contents_titles, contents_end = _detect_contents(paragraphs)
    if headings:
        used = list(headings)
    elif contents_titles:
        used = list(contents_titles)
    else:
        used = _pattern_headings(paragraphs)

    detected = list(contents_titles)
    if not detected and not headings:
        detected = list(used)

    start = contents_end if contents_end is not None else 0
    built, missing = _split_chapters(paragraphs, used, start)
    for heading in missing:
        warnings.append(f'Heading "{heading}" was not found in the text.')

    # A drop-cap token like "GEZHE" occurs once by definition, so "known" for
    # the opening check means a word used more than once in the book (or in
    # the system dictionary), not merely present in the vocabulary.
    counts: dict[str, int] = {}
    for paragraph in paragraphs:
        for word in re.findall(r"[A-Za-z]+", paragraph):
            counts[word.lower()] = counts.get(word.lower(), 0) + 1
    repeated = {word for word, count in counts.items() if count >= 2}
    warnings.extend(_suspicious_openings(built, repeated | system_words))

    chapters = [
        ImportedChapter(title=title, paragraphs=paragraphs)
        for title, paragraphs in built
    ]
    return ImportResult(chapters=chapters, warnings=warnings, detected_headings=detected)


# --- public API ------------------------------------------------------------


def import_text(
    text: str, *, ocr: bool = False, headings: list[str] | None = None
) -> ImportResult:
    """Import plain text (``ocr=False``) or archive.org OCR text (``ocr=True``)."""
    system_words = _system_words()
    if ocr:
        paragraphs, warnings = _ocr_paragraphs(text)
        known = _book_vocabulary(text) | system_words
        paragraphs = [_dehyphenate(p, known) for p in paragraphs]
        paragraphs = [p for p in paragraphs if _keep_paragraph(p)]
    else:
        paragraphs = _plain_paragraphs(text)
        warnings = []
    return _assemble(paragraphs, headings, system_words, warnings)


def import_html(html: str, *, headings: list[str] | None = None) -> ImportResult:
    """Import an HTML transcription (e.g. a Lehigh Scalar page)."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "header", "footer", "form", "noscript"]):
        tag.decompose()
    container = _find_main_container(soup)
    raw_paragraphs = _html_paragraphs(container)

    system_words = _system_words()
    known = _book_vocabulary(" ".join(raw_paragraphs)) | system_words
    paragraphs = []
    for paragraph in raw_paragraphs:
        paragraph = " ".join(paragraph.replace("\xa0", " ").split())
        if not paragraph:
            continue
        paragraph = _dehyphenate(paragraph, known)
        paragraphs.append(paragraph)
    return _assemble(paragraphs, headings, system_words, [])


# --- EPUB output -----------------------------------------------------------


def build_epub(
    path,
    *,
    title: str,
    author: str | None,
    date: str | None,
    source: str | None,
    chapters: list[ImportedChapter],
    language: str = "en",
) -> None:
    """Write ``chapters`` as a minimal, valid EPUB 3 to ``path``."""
    output = Path(path)
    uid = f"urn:uuid:{uuid.uuid5(uuid.NAMESPACE_URL, source or title)}"
    modified = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    docs = []
    for number, chapter in enumerate(chapters, 1):
        heading = chapter.title or ""
        body = f"<h2>{escape(heading)}</h2>\n" if heading else ""
        body += "\n".join(f"<p>{escape(p)}</p>" for p in chapter.paragraphs)
        docs.append((f"c{number:02d}", f"c{number:02d}.xhtml", heading or title, body))

    metadata = [
        f'<dc:identifier id="uid">{escape(uid)}</dc:identifier>',
        f"<dc:title>{escape(title)}</dc:title>",
    ]
    if author:
        metadata.append(f"<dc:creator>{escape(author)}</dc:creator>")
    metadata.append(f"<dc:language>{escape(language)}</dc:language>")
    if date:
        metadata.append(f"<dc:date>{escape(date)}</dc:date>")
    if source:
        metadata.append(f"<dc:source>{escape(source)}</dc:source>")
    metadata.append(f'<meta property="dcterms:modified">{modified}</meta>')

    manifest = [
        '<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" '
        'properties="nav"/>'
    ]
    manifest += [
        f'<item id="{doc_id}" href="{href}" media-type="application/xhtml+xml"/>'
        for doc_id, href, _title, _body in docs
    ]
    spine = [f'<itemref idref="{doc_id}"/>' for doc_id, _href, _title, _body in docs]
    opf = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<package xmlns="http://www.idpf.org/2007/opf" version="3.0" '
        f'unique-identifier="uid" xml:lang="{escape(language)}">\n'
        '  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">\n    '
        + "\n    ".join(metadata)
        + "\n  </metadata>\n"
        '  <manifest>' + "".join(manifest) + "</manifest>\n"
        '  <spine>' + "".join(spine) + "</spine>\n"
        "</package>"
    )

    nav_items = "".join(
        f'<li><a href="{href}">{escape(doc_title)}</a></li>'
        for _doc_id, href, doc_title, _body in docs
    )
    nav = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE html>\n'
        '<html xmlns="http://www.w3.org/1999/xhtml" '
        'xmlns:epub="http://www.idpf.org/2007/ops" xml:lang="en" lang="en">\n'
        f"<head><title>{escape(title)}</title></head>\n"
        '<body><nav epub:type="toc"><h1>Contents</h1>'
        f"<ol>{nav_items}</ol></nav></body></html>"
    )
    container_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<container version="1.0" '
        'xmlns="urn:oasis:names:tc:opendocument:xmlns:container">\n'
        '<rootfiles><rootfile full-path="OEBPS/content.opf" '
        'media-type="application/oebps-package+xml"/></rootfiles>\n'
        "</container>"
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        dir=str(output.parent), prefix=output.name + ".", suffix=".tmp"
    )
    os.close(fd)
    try:
        with zipfile.ZipFile(temp_name, "w") as archive:
            archive.writestr(
                "mimetype",
                "application/epub+zip",
                compress_type=zipfile.ZIP_STORED,
            )
            archive.writestr(
                "META-INF/container.xml",
                container_xml,
                compress_type=zipfile.ZIP_DEFLATED,
            )
            archive.writestr(
                "OEBPS/content.opf", opf, compress_type=zipfile.ZIP_DEFLATED
            )
            archive.writestr("OEBPS/nav.xhtml", nav, compress_type=zipfile.ZIP_DEFLATED)
            for _doc_id, href, doc_title, body in docs:
                document = (
                    '<?xml version="1.0" encoding="UTF-8"?>\n'
                    '<!DOCTYPE html>\n'
                    '<html xmlns="http://www.w3.org/1999/xhtml" '
                    'xml:lang="en" lang="en">\n'
                    f"<head><title>{escape(doc_title)}</title></head>\n"
                    f"<body>\n{body}\n</body></html>"
                )
                archive.writestr(
                    f"OEBPS/{href}", document, compress_type=zipfile.ZIP_DEFLATED
                )
        os.chmod(temp_name, 0o644)
        os.replace(temp_name, output)
    except BaseException:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
        raise
