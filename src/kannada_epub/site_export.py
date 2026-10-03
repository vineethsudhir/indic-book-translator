"""Export translated books as a self-contained static reading website.

Plain HTML + CSS only: no JavaScript, no server and no remote resources, so
a translated book can be published by copying one folder. The pairing of
English and translation mirrors the Library reader exactly
(``source_english`` zipped with the marker-stripped ``edited_kannada``, in
batch order); a book with any count mismatch is skipped, never misaligned.
"""

from __future__ import annotations

import html
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

from .epub_writer import _clean_title_text, _default_font_dir, find_gutenberg_mentions
from .inline_markup import strip_markers
from .languages import TargetLanguage, get_language
from .library import book_metadata, book_outputs, source_epub, source_titles

_NOTICE = (
    "Machine translation into {language} of the public-domain English text. "
    "It has not been checked by a human translator and may contain errors."
)

_BASE_CSS = """\
:root {
  color-scheme: light dark;
}
* {
  box-sizing: border-box;
}
body {
  margin: 0 auto;
  max-width: 960px;
  padding: 28px 22px 64px;
  background: #fbfaf7;
  color: #1b1b19;
  font-family: Georgia, "Times New Roman", serif;
  line-height: 1.65;
}
a {
  color: #17614a;
}
a:hover {
  color: #0e4433;
}
h1 {
  font-size: 1.7rem;
  line-height: 1.25;
  margin: 0 0 0.4em;
}
p {
  margin: 0 0 0.8em;
}
.meta,
.credit,
.notice,
.en {
  color: #5a574f;
}
.notice {
  font-size: 0.9rem;
}
.preview {
  color: #8a5a00;
  font-weight: 600;
}
.books {
  list-style: none;
  padding: 0;
  display: grid;
  gap: 14px;
}
.books li {
  border: 1px solid #e2ded4;
  border-radius: 10px;
  padding: 14px 16px;
  background: #ffffff;
}
.books .book-title {
  font-size: 1.15rem;
  font-weight: 700;
}
.chapters {
  padding-left: 1.2em;
}
.chapters li {
  margin: 0.35em 0;
}
.row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 22px;
  padding: 16px 0;
  border-bottom: 1px solid #e6e2d8;
}
.row:last-child {
  border-bottom: 0;
}
.translated {
  font-size: 1.12rem;
}
.nav {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  margin: 18px 0;
  font-size: 0.95rem;
}
.nav .contents {
  text-align: center;
}
.epub-link {
  display: inline-block;
  margin-top: 8px;
}
footer {
  margin-top: 40px;
  border-top: 1px solid #e6e2d8;
  padding-top: 16px;
}
@media (max-width: 700px) {
  .row {
    grid-template-columns: 1fr;
    gap: 6px;
  }
}
@media (prefers-color-scheme: dark) {
  body {
    background: #15171b;
    color: #e9e7e1;
  }
  a {
    color: #7fd0b0;
  }
  a:hover {
    color: #a5e4cb;
  }
  .books li {
    background: #1d2026;
    border-color: #2c313a;
  }
  .row,
  footer {
    border-color: #2c313a;
  }
  .meta,
  .credit,
  .notice,
  .en {
    color: #a7a49c;
  }
  .preview {
    color: #e0b25f;
  }
}
"""


@dataclass
class SiteExportResult:
    """What one export wrote: where, how much, and what was skipped."""

    path: Path
    books: int
    chapters: int
    skipped: list[tuple[str, str]] = field(default_factory=list)
    warnings: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class _Chapter:
    id: str
    title: str
    index: int
    paragraphs: list[tuple[str, str]]


@dataclass
class _Book:
    folder: Path
    slug: str
    title: str
    author: str
    source_name: str
    language: TargetLanguage
    lang_key: str
    outputs: dict
    chapters: list[_Chapter]
    epub: Path | None = None


def _esc(value: object) -> str:
    """Escape every piece of user text or metadata before it reaches HTML."""
    return html.escape(str(value), quote=True)


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "book"


def _unique_slug(base: str, used: set[str]) -> str:
    slug = base
    suffix = 2
    while slug in used:
        slug = f"{base}-{suffix}"
        suffix += 1
    used.add(slug)
    return slug


def _preview_label(outputs: dict) -> str:
    return (
        f"Preview: {outputs['paragraphs_translated']} of "
        f"{outputs['paragraphs_total']} paragraphs translated"
    )


def _chapter_pairs(data: dict) -> list[tuple[str, str]]:
    """English/translation pairs for one chapter, in the reader's order.

    ``zip(..., strict=True)`` is deliberate: a batch whose English and
    translation lists disagree raises ``ValueError`` (the book is skipped),
    never shifts a Kannada paragraph under the wrong English one.
    """
    pairs: list[tuple[str, str]] = []
    for batch in data.get("batches") or []:
        english = batch.get("source_english") or []
        translated = batch.get("edited_kannada") or []
        for source, target in zip(english, translated, strict=True):
            pairs.append((str(source), strip_markers(str(target))))
    return pairs


def _page(*, title: str, css_href: str, body: str) -> str:
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{_esc(title)}</title>\n"
        f'<link rel="stylesheet" href="{_esc(css_href)}">\n'
        "</head>\n"
        "<body>\n"
        f"{body}\n"
        "</body>\n"
        "</html>\n"
    )


def _mentions_gutenberg(epub: Path) -> bool:
    try:
        return bool(find_gutenberg_mentions(epub))
    except Exception:  # unreadable EPUB: don't publish what can't be checked
        return True


def _collect_books(
    book_folders: list[Path],
    skipped: list[tuple[str, str]],
    warnings: list[tuple[str, str]],
) -> tuple[list[_Book], list[TargetLanguage]]:
    books: list[_Book] = []
    used_languages: list[TargetLanguage] = []
    used_slugs: set[str] = set()

    for folder in book_folders:
        folder = Path(folder)
        name = folder.name
        manifest_path = folder / "manifest.json"
        if not manifest_path.is_file():
            skipped.append((name, "no manifest.json"))
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            skipped.append((name, "unreadable manifest.json"))
            continue
        if not isinstance(manifest, dict):
            skipped.append((name, "unreadable manifest.json"))
            continue

        raw_chapters = manifest.get("chapters")
        chapters_dir = folder / "chapters"
        if (
            not isinstance(raw_chapters, list)
            or not raw_chapters
            or not chapters_dir.is_dir()
        ):
            skipped.append((name, "no chapters"))
            continue

        lang_key = str(manifest.get("target_language") or "kn")
        try:
            language = get_language(lang_key)
        except ValueError:
            language = get_language("kn")

        outputs = book_outputs(folder, manifest)
        title, author = book_metadata(folder, manifest)
        # Source titles often read "The Project Gutenberg eBook of …"; the
        # trademark must not appear in a redistributed translation.
        title = _clean_title_text(title, folder.name)
        author = _clean_title_text(author, "Unknown author")
        source = source_epub(folder, manifest)
        source_name = (
            source.name
            if source is not None
            else (Path(str(manifest.get("epub", ""))).name or folder.name)
        )
        titles = source_titles(folder, manifest)

        chapters: list[_Chapter] = []
        mismatched = False
        for entry in raw_chapters:
            chapter_id = entry.get("id") if isinstance(entry, dict) else None
            if (
                not isinstance(chapter_id, str)
                or not chapter_id
                or chapter_id in {".", ".."}
                or "/" in chapter_id
                or "\\" in chapter_id
            ):
                continue
            chapter_path = chapters_dir / f"{chapter_id}.json"
            if not chapter_path.is_file():
                continue
            try:
                chapter_data = json.loads(chapter_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(chapter_data, dict):
                continue
            try:
                pairs = _chapter_pairs(chapter_data)
            except ValueError:
                mismatched = True
                break
            chapter_title = (
                titles.get(chapter_id)
                or (entry.get("title") if isinstance(entry, dict) else None)
                or chapter_id
            )
            chapters.append(
                _Chapter(
                    id=chapter_id,
                    title=_clean_title_text(str(chapter_title), title),
                    index=len(chapters) + 1,
                    paragraphs=pairs,
                )
            )

        if mismatched:
            # Never publish a book whose saved English and translation
            # paragraph counts disagree: pairing them would misalign text.
            skipped.append((name, "English and translated paragraph counts differ"))
            continue
        if not chapters:
            skipped.append((name, "no chapters"))
            continue

        epub: Path | None = folder / outputs["epub"]
        if not epub.is_file():
            epub = None
        elif _mentions_gutenberg(epub):
            # Translations made before boilerplate removal still carry the
            # trademark; publishing them would redistribute it.
            warnings.append((
                name,
                "EPUB not included: it still contains Project Gutenberg text. "
                "Translate the book again with boilerplate removal on.",
            ))
            epub = None

        slug = _unique_slug(_slugify(folder.name), used_slugs)
        books.append(
            _Book(
                folder=folder,
                slug=slug,
                title=title,
                author=author,
                source_name=source_name,
                language=language,
                lang_key=lang_key,
                outputs=outputs,
                chapters=chapters,
                epub=epub,
            )
        )
        if all(existing.key != language.key for existing in used_languages):
            used_languages.append(language)

    return books, used_languages


def _font_face(language: TargetLanguage) -> str:
    return (
        "@font-face {\n"
        f'  font-family: "{language.font_family}";\n'
        "  font-style: normal;\n"
        "  font-weight: 400;\n"
        f'  src: url("fonts/{language.font_regular}") format("truetype");\n'
        "}\n"
        "@font-face {\n"
        f'  font-family: "{language.font_family}";\n'
        "  font-style: normal;\n"
        "  font-weight: 700;\n"
        f'  src: url("fonts/{language.font_bold}") format("truetype");\n'
        "}"
    )


def _write_style(site: Path, languages: list[TargetLanguage], font_dir: Path) -> None:
    rules: list[str] = []
    copied: dict[str, Path] = {}
    for language in languages:
        regular = font_dir / language.font_regular
        bold = font_dir / language.font_bold
        if regular.is_file() and bold.is_file():
            rules.append(_font_face(language))
            copied[language.font_regular] = regular
            copied[language.font_bold] = bold
            ofl = font_dir / "OFL.txt"
            if ofl.is_file():
                copied["OFL.txt"] = ofl
        rules.append(
            f'[lang="{language.key}"] {{ font-family: "{language.font_family}", serif; }}'
        )

    if copied:
        fonts_dir = site / "fonts"
        fonts_dir.mkdir(parents=True, exist_ok=True)
        for filename, source in copied.items():
            shutil.copyfile(source, fonts_dir / filename)

    css = _BASE_CSS + "\n" + "\n".join(rules) + "\n"
    (site / "style.css").write_text(css, encoding="utf-8")


def _chapter_count_text(count: int) -> str:
    return f"{count} chapter" + ("" if count == 1 else "s")


def _write_index(site: Path, books: list[_Book]) -> None:
    items: list[str] = []
    for book in books:
        preview = (
            f'<p class="preview">{_esc(_preview_label(book.outputs))}</p>'
            if book.outputs["preview"]
            else ""
        )
        items.append(
            f'<li><a class="book-title" href="{_esc(book.slug)}/index.html">'
            f"{_esc(book.title)}</a>"
            f'<p class="meta">{_esc(book.author)} · '
            f"{_chapter_count_text(len(book.chapters))}</p>"
            f"{preview}</li>"
        )
    body = (
        "<header>\n<h1>Translated books</h1>\n"
        "<p class=\"meta\">Machine-translated English texts, shown with the "
        "original.</p>\n</header>\n"
        f'<ul class="books">{"".join(items)}</ul>'
    )
    (site / "index.html").write_text(
        _page(title="Translated books", css_href="style.css", body=body),
        encoding="utf-8",
    )


def _write_book_page(site: Path, book: _Book) -> None:
    book_dir = site / book.slug
    book_dir.mkdir()
    notice = _esc(_NOTICE.format(language=book.language.name))
    preview = (
        f'<p class="preview">{_esc(_preview_label(book.outputs))}</p>'
        if book.outputs["preview"]
        else ""
    )
    credit = _esc(
        f"{book.title} by {book.author}. English source: {book.source_name}."
    )
    epub_link = ""
    if book.epub is not None:
        epub_link = (
            f'<p><a class="epub-link" href="{_esc(quote(book.epub.name))}">Download EPUB</a></p>'
        )
    chapter_items = "".join(
        f'<li><a href="{chapter.index}.html">{_esc(chapter.title)}</a></li>'
        for chapter in book.chapters
    )
    body = (
        "<header>\n"
        f"<h1>{_esc(book.title)}</h1>\n"
        f'<p class="meta">{_esc(book.author)}</p>\n'
        f'<p class="credit">{credit}</p>\n'
        f"{preview}\n"
        f'<p class="notice">{notice}</p>\n'
        f"{epub_link}\n"
        "</header>\n"
        f'<ol class="chapters">{chapter_items}</ol>\n'
        f'<footer><p class="notice">{notice}</p></footer>'
    )
    (book_dir / "index.html").write_text(
        _page(title=book.title, css_href="../style.css", body=body),
        encoding="utf-8",
    )


def _copy_epub(site: Path, book: _Book) -> None:
    if book.epub is not None:
        shutil.copyfile(book.epub, site / book.slug / book.epub.name)


def _write_chapter_pages(site: Path, book: _Book) -> int:
    book_dir = site / book.slug
    notice = _esc(_NOTICE.format(language=book.language.name))
    total = len(book.chapters)
    for chapter in book.chapters:
        rows = "".join(
            f'<div class="row"><div class="en" lang="en">{_esc(source)}</div>'
            f'<div class="translated" lang="{_esc(book.lang_key)}">'
            f"{_esc(target)}</div></div>"
            for source, target in chapter.paragraphs
        )
        previous = (
            f'<a class="prev" href="{chapter.index - 1}.html">← Previous</a>'
            if chapter.index > 1
            else '<span class="prev"></span>'
        )
        following = (
            f'<a class="next" href="{chapter.index + 1}.html">Next →</a>'
            if chapter.index < total
            else '<span class="next"></span>'
        )
        navigation = (
            f'<nav class="nav">{previous}'
            '<a class="contents" href="index.html">Contents</a>'
            f"{following}</nav>"
        )
        body = (
            f"{navigation}\n"
            f"<h1>{_esc(chapter.title)}</h1>\n"
            f'<div class="paragraphs">{rows}</div>\n'
            f"{navigation}\n"
            f'<footer><p class="notice">{notice}</p></footer>'
        )
        (book_dir / f"{chapter.index}.html").write_text(
            _page(
                title=f"{chapter.title} — {book.title}",
                css_href="../style.css",
                body=body,
            ),
            encoding="utf-8",
        )
    return total


def export_site(
    book_folders: list[Path],
    dest: Path,
    *,
    font_dir: Path | None = None,
) -> SiteExportResult:
    """Write a static website for ``book_folders`` at ``dest``.

    ``dest`` must not already exist; the site is built in a sibling ``.tmp``
    folder and renamed into place, so a failure leaves no half-written site.
    Folders without a readable manifest or chapters are skipped, not fatal.
    """
    dest = Path(dest)
    if dest.exists():
        raise FileExistsError(f"{dest} already exists")
    font_dir = Path(font_dir) if font_dir is not None else _default_font_dir()

    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_name(dest.name + ".tmp")
    if temp.exists():
        shutil.rmtree(temp)

    skipped: list[tuple[str, str]] = []
    warnings: list[tuple[str, str]] = []
    try:
        books, languages = _collect_books(book_folders, skipped, warnings)
        temp.mkdir()
        _write_style(temp, languages, font_dir)
        _write_index(temp, books)
        chapter_count = 0
        for book in books:
            _write_book_page(temp, book)
            _copy_epub(temp, book)
            chapter_count += _write_chapter_pages(temp, book)
        temp.rename(dest)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise

    return SiteExportResult(
        path=dest,
        books=len(books),
        chapters=chapter_count,
        skipped=skipped,
        warnings=warnings,
    )
