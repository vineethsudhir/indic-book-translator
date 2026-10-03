"""Tests for the static reading-site export (``kannada_epub.site_export``).

Fast and offline: it builds fake output folders (a source EPUB, a
``manifest.json`` and ``chapters/*.json``) in a temp directory and checks the
written HTML/CSS, never a model or the network.

Run: .venv/bin/python scripts/test_site_export.py
"""

from __future__ import annotations

import html
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(tempfile.mkdtemp(prefix="kannada-site-test-"))
# Must be set before the app package reads its paths.
os.environ["KANNADA_APP_DATA_DIR"] = str(DATA_DIR)

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from epub_fixture import build_epub  # noqa: E402

from kannada_epub.app.paths import books_dir  # noqa: E402
from kannada_epub.site_export import SiteExportResult, export_site  # noqa: E402

SOURCE_DOC = (
    '<?xml version="1.0" encoding="utf-8"?>'
    '<html xmlns="http://www.w3.org/1999/xhtml">'
    "<head><title>Source</title></head>"
    "<body><h1>Source</h1><p>Some public-domain text.</p></body></html>"
)

_HREF_RE = re.compile(r'(?:href|src)="([^"]+)"')
_URL_RE = re.compile(r'url\(\s*["\']?([^"\')]+)["\']?\s*\)')


def assert_raises(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type as exc:
        return exc
    raise AssertionError(f"expected {exc_type.__name__}, but no exception was raised")


def write_book(
    folder: Path,
    *,
    title: str,
    author: str,
    chapters: list[dict],
    lang: str = "kn",
    stem: str | None = None,
    preview: bool | None = None,
    paragraphs_total: int | None = None,
    target_language: str | None = None,
    corrupt_pairs: bool = False,
    epub_text: str = "Translated text.",
) -> Path:
    """Create an output folder that looks like a finished translation run.

    ``chapters`` is a list of ``{"id", "title", "pairs": [(en, kn), ...]}``.
    A source EPUB is written into ``books_dir()`` and referenced (absolutely,
    like a real manifest) so the title/author are read from its metadata.
    """
    folder.mkdir(parents=True, exist_ok=True)
    stem = stem or folder.name
    source = books_dir() / f"{stem}.epub"
    build_epub(
        source,
        [{"id": "src", "href": "src.xhtml", "content": SOURCE_DOC}],
        title=title,
        creator=author,
    )

    chapter_meta = []
    translated = 0
    for chapter in chapters:
        pairs = list(chapter["pairs"])
        translated += len(pairs)
        chapter_meta.append(
            {
                "id": chapter["id"],
                "title": chapter["title"],
                "paragraphs": len(pairs),
                "total_paragraphs": len(pairs),
            }
        )
        english = [en for en, _kn in pairs]
        kannada = [kn for _en, kn in pairs]
        if corrupt_pairs:
            # One short list must keep the book out of the site (never misalign).
            kannada = kannada[:-1]
        chapter_data = {
            "batches": [
                {
                    "paragraph_start": 0,
                    "source_english": english,
                    "edited_kannada": kannada,
                    "edited_emotions": [None] * len(english),
                }
            ]
        }
        chapters_dir = folder / "chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)
        (chapters_dir / f"{chapter['id']}.json").write_text(
            json.dumps(chapter_data, ensure_ascii=False), encoding="utf-8"
        )

    total = translated if paragraphs_total is None else paragraphs_total
    if preview is None:
        preview = translated < total
    epub_name = f"{stem}.{lang}.epub"
    build_epub(
        folder / epub_name,
        [{"id": "out", "href": "out.xhtml",
          "content": SOURCE_DOC.replace("Some public-domain text.", epub_text)}],
        title=title,
        creator=author,
    )
    manifest = {
        "epub": str(source),
        "preview": preview,
        "paragraphs_translated": translated,
        "paragraphs_total": total,
        "chapters": chapter_meta,
    }
    if target_language is not None:
        manifest["target_language"] = target_language
    (folder / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    return folder


def text_files(dest: Path) -> list[Path]:
    return [
        path
        for path in dest.rglob("*")
        if path.is_file() and path.suffix.lower() in {".html", ".css"}
    ]


def all_files(dest: Path) -> list[Path]:
    return [path for path in dest.rglob("*") if path.is_file()]


# ---------------------------------------------------------------------------
# layout and counts
# ---------------------------------------------------------------------------
def test_layout_and_counts() -> None:
    root = Path(tempfile.mkdtemp(dir=DATA_DIR))
    kannada = write_book(
        root / "sherlock",
        title="Sherlock Holmes",
        author="Arthur Conan Doyle",
        chapters=[
            {"id": "c1", "title": "A Scandal", "pairs": [("One.", "ಒಂದು.")]},
            {"id": "c2", "title": "The Red-Headed League", "pairs": [("Two.", "ಎರಡು.")]},
        ],
    )
    tamil = write_book(
        root / "silappathikaram",
        title="Tamil Book",
        author="Ilango",
        lang="ta",
        target_language="ta",
        chapters=[{"id": "t1", "title": "First", "pairs": [("A.", "அ."), ("B.", "ஆ.")]}],
    )
    dest = root / "site"

    result = export_site([kannada, tamil], dest)

    assert isinstance(result, SiteExportResult)
    assert result.path == dest
    assert result.books == 2, result
    assert result.chapters == 3, result
    assert result.skipped == [], result.skipped

    expected = [
        "index.html",
        "style.css",
        "sherlock/index.html",
        "sherlock/1.html",
        "sherlock/2.html",
        "sherlock/sherlock.kn.epub",
        "silappathikaram/index.html",
        "silappathikaram/1.html",
        "silappathikaram/silappathikaram.ta.epub",
    ]
    for relative in expected:
        assert (dest / relative).is_file(), relative

    # Chapter files are numbered, never named after chapter ids.
    assert not (dest / "sherlock" / "c1.html").exists()
    assert not (dest / "silappathikaram" / "t1.html").exists()

    index = (dest / "index.html").read_text(encoding="utf-8")
    assert "Sherlock Holmes" in index and "Tamil Book" in index
    assert "3 chapters" not in index  # 2 + 1, shown per book


def test_slug_dedup() -> None:
    root = Path(tempfile.mkdtemp(dir=DATA_DIR))
    folders = []
    for name in ("Alpha Book", "alpha-book", "alpha_book"):
        folders.append(
            write_book(
                root / name,
                title=f"Book {name}",
                author="A",
                chapters=[{"id": "c1", "title": "One", "pairs": [("x", "y")]}],
            )
        )
    dest = root / "site"
    export_site(folders, dest)
    assert (dest / "alpha-book" / "index.html").is_file()
    assert (dest / "alpha-book-2" / "index.html").is_file()
    assert (dest / "alpha-book-3" / "index.html").is_file()


# ---------------------------------------------------------------------------
# pairing, order, markers, lang
# ---------------------------------------------------------------------------
def test_pairs_order_and_markers() -> None:
    root = Path(tempfile.mkdtemp(dir=DATA_DIR))
    book = write_book(
        root / "marked",
        title="Marked",
        author="A",
        chapters=[
            {
                "id": "c1",
                "title": 'Ch & <One> "quoted"',
                "pairs": [
                    ("First English.", "ಮೊದಲ ⟦1⟧ಭಾಗ⟦/1⟧"),
                    ("Second English.", "ಎರಡನೇ ಭಾಗ"),
                ],
            }
        ],
    )
    dest = root / "site"
    export_site([book], dest)
    page = (dest / "marked" / "1.html").read_text(encoding="utf-8")

    assert '<html lang="en">' in page
    assert 'lang="kn"' in page
    assert "⟦" not in page and "⟧" not in page

    positions = [
        page.index("First English."),
        page.index("ಮೊದಲ ಭಾಗ"),
        page.index("Second English."),
        page.index("ಎರಡನೇ ಭಾಗ"),
    ]
    assert positions == sorted(positions), positions

    # Chapter titles are escaped too, and the raw form never appears.
    assert "Ch &amp; &lt;One&gt;" in page
    assert "Ch & <One>" not in page


def test_default_language_is_kannada() -> None:
    root = Path(tempfile.mkdtemp(dir=DATA_DIR))
    book = write_book(
        root / "old",
        title="Old Style",
        author="A",
        chapters=[{"id": "c1", "title": "One", "pairs": [("Hello.", "ನಮಸ್ಕಾರ")]}],
    )
    # A manifest written before target_language existed: there is no key, so
    # ``_collect_books`` must default the lang attribute to "kn".
    manifest_path = root / "old" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "target_language" not in manifest
    dest = root / "site"
    export_site([book], dest)
    page = (dest / "old" / "1.html").read_text(encoding="utf-8")
    assert 'lang="kn"' in page


# ---------------------------------------------------------------------------
# escaping
# ---------------------------------------------------------------------------
def test_escaping() -> None:
    root = Path(tempfile.mkdtemp(dir=DATA_DIR))
    book = write_book(
        root / "escape",
        title='Tom & "Jerry" <Tales>',
        author='A & B <C>',
        chapters=[
            {
                "id": "c1",
                "title": 'Bad <b>Title</b> & "quotes"',
                "pairs": [
                    ("<script>alert('en')</script>", "<script>alert('kn')</script>"),
                ],
            }
        ],
    )
    dest = root / "site"
    export_site([book], dest)

    index = (dest / "index.html").read_text(encoding="utf-8")
    book_page = (dest / "escape" / "index.html").read_text(encoding="utf-8")
    chapter = (dest / "escape" / "1.html").read_text(encoding="utf-8")

    for page in (index, book_page, chapter):
        assert "<script" not in page.lower()
        assert "&lt;script&gt;" in page or "<script" not in page

    assert "&amp;" in book_page and "&quot;" in book_page and "&lt;Tales&gt;" in book_page
    assert "<Tales>" not in index and "<Tales>" not in book_page
    assert "&lt;script&gt;alert" in chapter
    assert "<script>alert" not in chapter


# ---------------------------------------------------------------------------
# trademark: no "Gutenberg" (case-insensitive) anywhere
# ---------------------------------------------------------------------------
def test_no_gutenberg() -> None:
    root = Path(tempfile.mkdtemp(dir=DATA_DIR))
    book = write_book(
        root / "plain",
        # Real Gutenberg sources put the trademark in titles like these.
        title="The Project Gutenberg eBook of A Plain Book",
        author="Nobody",
        chapters=[
            {"id": "c1", "title": "The Project Gutenberg eBook of A Plain Book",
             "pairs": [("Public domain.", "ಸಾರ್ವಜನಿಕ")]},
            {"id": "c2", "title": "Two | Project Gutenberg", "pairs": [("b", "ಬ")]},
        ],
    )
    dest = root / "site"
    export_site([book], dest)
    book_page = (dest / "plain" / "index.html").read_text(encoding="utf-8")
    assert "<h1>A Plain Book</h1>" in book_page, book_page
    assert '<a href="2.html">Two</a>' in book_page, book_page
    for path in all_files(dest):
        content = path.read_bytes().lower()
        assert b"gutenberg" not in content, path


def test_epub_with_gutenberg_text_is_left_out() -> None:
    root = Path(tempfile.mkdtemp(dir=DATA_DIR))
    book = write_book(
        root / "old",
        title="Old Run",
        author="A",
        chapters=[{"id": "c1", "title": "One", "pairs": [("a", "ಅ")]}],
        epub_text="End of the Project Gutenberg EBook",
    )
    dest = root / "site"
    result = export_site([book], dest)
    assert result.books == 1, result
    assert [name for name, _ in result.warnings] == ["old"], result
    assert not list((dest / "old").glob("*.epub"))
    assert "Download EPUB" not in (dest / "old" / "index.html").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# preview label
# ---------------------------------------------------------------------------
def test_preview_label() -> None:
    root = Path(tempfile.mkdtemp(dir=DATA_DIR))
    preview = write_book(
        root / "partial",
        title="Partial",
        author="A",
        chapters=[{"id": "c1", "title": "One", "pairs": [("Only.", "ಮಾತ್ರ")]}],
        preview=True,
        paragraphs_total=5,
    )
    whole = write_book(
        root / "whole",
        title="Whole",
        author="A",
        chapters=[{"id": "c1", "title": "One", "pairs": [("Done.", "ಮುಗಿದಿದೆ")]}],
    )
    dest = root / "site"
    export_site([preview, whole], dest)

    label = "Preview: 1 of 5 paragraphs translated"
    assert label in (dest / "index.html").read_text(encoding="utf-8")
    assert label in (dest / "partial" / "index.html").read_text(encoding="utf-8")
    assert "Preview:" not in (dest / "whole" / "index.html").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# fonts
# ---------------------------------------------------------------------------
def test_fonts() -> None:
    root = Path(tempfile.mkdtemp(dir=DATA_DIR))
    books = [
        write_book(
            root / "kn-one",
            title="Kannada One",
            author="A",
            chapters=[{"id": "c1", "title": "One", "pairs": [("a", "ಬ")]}],
        ),
        write_book(
            root / "kn-two",
            title="Kannada Two",
            author="A",
            chapters=[{"id": "c1", "title": "One", "pairs": [("c", "ಡ")]}],
        ),
        write_book(
            root / "tamil",
            title="Tamil",
            author="A",
            lang="ta",
            target_language="ta",
            chapters=[{"id": "c1", "title": "One", "pairs": [("e", "எ")]}],
        ),
    ]
    dest = root / "site"
    export_site(books, dest)  # default font_dir = epub_writer._default_font_dir()

    font_files = sorted(path.name for path in (dest / "fonts").iterdir())
    assert font_files == [
        "NotoSansKannada-Bold.ttf",
        "NotoSansKannada-Regular.ttf",
        "OFL.txt",
    ], font_files

    css = (dest / "style.css").read_text(encoding="utf-8")
    # Regular + bold for Kannada only, and only once even for two books.
    assert css.count("@font-face") == 2, css
    assert "NotoSansKannada-Regular.ttf" in css
    assert "NotoSansKannada-Bold.ttf" in css
    assert "NotoSansTamil" not in css
    # Tamil still gets a family with a serif fallback.
    assert '[lang="ta"] { font-family: "Noto Sans Tamil", serif; }' in css
    assert '[lang="kn"] { font-family: "Noto Sans Kannada", serif; }' in css


def test_missing_fonts_means_no_font_face() -> None:
    root = Path(tempfile.mkdtemp(dir=DATA_DIR))
    empty_fonts = root / "no-fonts"
    empty_fonts.mkdir()
    book = write_book(
        root / "tamil",
        title="Tamil",
        author="A",
        lang="ta",
        target_language="ta",
        chapters=[{"id": "c1", "title": "One", "pairs": [("a", "அ")]}],
    )
    dest = root / "site"
    export_site([book], dest, font_dir=empty_fonts)
    css = (dest / "style.css").read_text(encoding="utf-8")
    assert "@font-face" not in css
    assert not (dest / "fonts").exists()
    assert '[lang="ta"] { font-family: "Noto Sans Tamil", serif; }' in css


# ---------------------------------------------------------------------------
# failures and skips
# ---------------------------------------------------------------------------
def test_dest_exists_raises() -> None:
    root = Path(tempfile.mkdtemp(dir=DATA_DIR))
    book = write_book(
        root / "book",
        title="Book",
        author="A",
        chapters=[{"id": "c1", "title": "One", "pairs": [("a", "b")]}],
    )
    dest = root / "site"
    dest.mkdir()
    assert_raises(FileExistsError, export_site, [book], dest)


def test_skips_folders_without_manifest_or_chapters() -> None:
    root = Path(tempfile.mkdtemp(dir=DATA_DIR))
    good = write_book(
        root / "good",
        title="Good",
        author="A",
        chapters=[{"id": "c1", "title": "One", "pairs": [("a", "b")]}],
    )
    no_manifest = root / "no-manifest"
    no_manifest.mkdir()
    (no_manifest / "chapters").mkdir()

    no_chapters = root / "no-chapters"
    no_chapters.mkdir()
    (no_chapters / "manifest.json").write_text(
        json.dumps({"chapters": []}), encoding="utf-8"
    )

    dest = root / "site"
    result = export_site([good, no_manifest, no_chapters], dest)

    assert result.books == 1, result
    assert result.chapters == 1, result
    reasons = dict(result.skipped)
    assert "no-manifest" in reasons and "manifest" in reasons["no-manifest"]
    assert "no-chapters" in reasons and "chapters" in reasons["no-chapters"]
    assert (dest / "good" / "index.html").is_file()
    assert not (dest / "no-manifest").exists()
    assert not (dest / "no-chapters").exists()


def test_misaligned_book_is_skipped() -> None:
    root = Path(tempfile.mkdtemp(dir=DATA_DIR))
    broken = write_book(
        root / "broken",
        title="Broken",
        author="A",
        chapters=[{"id": "c1", "title": "One", "pairs": [("a", "b"), ("c", "d")]}],
        corrupt_pairs=True,
    )
    good = write_book(
        root / "good",
        title="Good",
        author="B",
        chapters=[{"id": "c1", "title": "One", "pairs": [("e", "ಎ")]}],
    )
    dest = root / "site"
    result = export_site([broken, good], dest)
    assert result.books == 1, result
    assert result.skipped == [("broken", "English and translated paragraph counts differ")], result
    assert not (dest / "broken").exists()
    assert (dest / "good" / "1.html").is_file()
    assert not dest.with_name(dest.name + ".tmp").exists()


# ---------------------------------------------------------------------------
# offline and link integrity
# ---------------------------------------------------------------------------
def test_no_scripts_or_remote_urls() -> None:
    root = Path(tempfile.mkdtemp(dir=DATA_DIR))
    book = write_book(
        root / "book",
        title="Book",
        author="A",
        chapters=[{"id": "c1", "title": "One", "pairs": [("a", "b")]}],
    )
    dest = root / "site"
    export_site([book], dest)

    for path in text_files(dest):
        content = path.read_text(encoding="utf-8")
        assert "<script" not in content.lower(), path
        assert "http://" not in content and "https://" not in content, path


def test_every_relative_link_exists() -> None:
    root = Path(tempfile.mkdtemp(dir=DATA_DIR))
    first = write_book(
        root / "first",
        title="First",
        author="A",
        chapters=[
            {"id": "c1", "title": "One", "pairs": [("a", "ಬ")]},
            {"id": "c2", "title": "Two", "pairs": [("c", "ಡ")]},
        ],
    )
    second = write_book(
        root / "second",
        title="Second",
        author="B",
        chapters=[{"id": "c1", "title": "One", "pairs": [("e", "ಎ")]}],
        # A file name with characters that must be URL-encoded in an href.
        stem="Tales #1 & more 100%",
    )
    dest = root / "site"
    export_site([first, second], dest)
    # "#" would cut the link short, so the name must be percent-encoded.
    assert 'href="Tales%20%231%20%26%20more%20100%25.kn.epub"' in (
        dest / "second" / "index.html"
    ).read_text(encoding="utf-8")

    html_files = [path for path in dest.rglob("*.html")]
    assert html_files
    for page in html_files:
        content = page.read_text(encoding="utf-8")
        for link in _HREF_RE.findall(content):
            if link.startswith(("#", "mailto:", "data:")):
                continue
            target = (page.parent / unquote(html.unescape(link))).resolve()
            assert target.is_file(), f"{page.relative_to(dest)} -> {link}"

    for css in dest.rglob("*.css"):
        content = css.read_text(encoding="utf-8")
        for url in _URL_RE.findall(content):
            if url.startswith(("data:", "#")):
                continue
            target = (css.parent / url).resolve()
            assert target.is_file(), f"{css.relative_to(dest)} -> {url}"


def main() -> None:
    try:
        test_layout_and_counts()
        test_slug_dedup()
        test_pairs_order_and_markers()
        test_default_language_is_kannada()
        test_escaping()
        test_no_gutenberg()
        test_epub_with_gutenberg_text_is_left_out()
        test_preview_label()
        test_fonts()
        test_missing_fonts_means_no_font_face()
        test_dest_exists_raises()
        test_skips_folders_without_manifest_or_chapters()
        test_misaligned_book_is_skipped()
        test_no_scripts_or_remote_urls()
        test_every_relative_link_exists()
    finally:
        import shutil

        shutil.rmtree(DATA_DIR, ignore_errors=True)

    print("test_site_export: all assertions passed")


if __name__ == "__main__":
    main()
