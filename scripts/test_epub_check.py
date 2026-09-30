"""Tests for ``kannada_epub.epub_check`` (source-EPUB defect detection).

Everything runs offline with stdlib + lxml and small fixtures: no models, no
network, no Java/EPUBCheck. ``data/sherlock_holmes.epub`` is checked as a real
clean book.

Run: .venv/bin/python scripts/test_epub_check.py
"""

import base64
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from epub_fixture import build_epub  # noqa: E402

from kannada_epub.epub_check import check_source_epub  # noqa: E402

SHERLOCK = ROOT / "data" / "sherlock_holmes.epub"

# A genuinely valid 1x1 PNG: signature, IHDR/IDAT/IEND chunks and zlib data.
PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mNk"
    "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)

CHAPTER = (
    '<?xml version="1.0" encoding="utf-8"?>'
    '<html xmlns="http://www.w3.org/1999/xhtml">'
    "<head><title>One</title></head>"
    "<body><h1>One</h1><p>Hello there.</p></body></html>"
)


def _chapter(doc_id: str = "ch1") -> dict:
    return {"id": doc_id, "href": f"{doc_id}.xhtml", "content": CHAPTER}


def _ncx(duplicate_id: bool) -> str:
    second = "np1" if duplicate_id else "np2"
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">'
        "<head/><docTitle><text>Fixture</text></docTitle><navMap>"
        '<navPoint id="np1" playOrder="1"><navLabel><text>One</text></navLabel>'
        '<content src="ch1.xhtml"/></navPoint>'
        f'<navPoint id="{second}" playOrder="2"><navLabel><text>Two</text></navLabel>'
        '<content src="ch1.xhtml"/></navPoint>'
        "</navMap></ncx>"
    )


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="epub-check-test-"))
    try:
        # --- a clean fixture reports nothing -----------------------------
        clean = tmp / "clean.epub"
        build_epub(clean, [_chapter()])
        assert check_source_epub(clean) == [], check_source_epub(clean)

        # --- valid PNG at an href with no extension ----------------------
        no_extension = tmp / "no_extension.epub"
        build_epub(no_extension, [
            _chapter(),
            {
                "id": "img",
                "href": "images/pic",
                "content": PNG_1PX,
                "media_type": "image/png",
            },
        ])
        problems = check_source_epub(no_extension)
        assert len(problems) == 1, problems
        assert problems[0].path == "EPUB/images/pic", problems[0]
        assert "has no .png file extension" in problems[0].message, problems[0]

        # --- bytes that aren't a PNG declared as image/png ----------------
        damaged = tmp / "damaged.epub"
        build_epub(damaged, [
            _chapter(),
            {
                "id": "img",
                "href": "images/pic.png",
                "content": b"this is not a png",
                "media_type": "image/png",
            },
        ])
        problems = check_source_epub(damaged)
        assert len(problems) == 1, problems
        assert problems[0].path == "EPUB/images/pic.png", problems[0]
        assert "is damaged or isn't really a PNG image" in problems[0].message, problems[0]

        # --- a manifest item whose file is missing ------------------------
        missing = tmp / "missing.epub"
        build_epub(missing, [
            _chapter(),
            {
                "id": "ch2",
                "href": "ch2.xhtml",
                "content": CHAPTER,
                "write": False,
            },
        ])
        problems = check_source_epub(missing)
        assert len(problems) == 1, problems
        assert problems[0].path == "EPUB/ch2.xhtml", problems[0]
        assert "missing from the EPUB" in problems[0].message, problems[0]

        # --- a non-well-formed XHTML document -----------------------------
        broken = tmp / "broken.epub"
        build_epub(broken, [{
            "id": "ch1",
            "href": "ch1.xhtml",
            "content": "<html><body><p>unclosed</body></html>",
        }])
        problems = check_source_epub(broken)
        assert len(problems) == 1, problems
        assert problems[0].path == "EPUB/ch1.xhtml", problems[0]
        assert "is not well-formed XHTML" in problems[0].message, problems[0]

        # --- duplicate ids in an XHTML document ---------------------------
        duplicate_xhtml = tmp / "duplicate_xhtml.epub"
        build_epub(duplicate_xhtml, [{
            "id": "ch1",
            "href": "ch1.xhtml",
            "content": (
                '<?xml version="1.0" encoding="utf-8"?>'
                '<html xmlns="http://www.w3.org/1999/xhtml"><body>'
                '<p id="x">a</p><p id="x">b</p><p id="x">c</p>'
                '<p id="y">d</p><p id="y">e</p>'
                "</body></html>"
            ),
        }])
        problems = check_source_epub(duplicate_xhtml)
        assert len(problems) == 1, problems
        assert problems[0].path == "EPUB/ch1.xhtml", problems[0]
        assert problems[0].message == "uses the id 'x' 3 times and 1 more", problems[0]

        # --- duplicate ids in an NCX (navPoint ids) -----------------------
        duplicate_ncx = tmp / "duplicate_ncx.epub"
        build_epub(duplicate_ncx, [_chapter()], ncx_content=_ncx(duplicate_id=True))
        problems = check_source_epub(duplicate_ncx)
        assert len(problems) == 1, problems
        assert problems[0].path == "EPUB/toc.ncx", problems[0]
        assert problems[0].message == "uses the id 'np1' 2 times", problems[0]

        # --- mimetype not first -------------------------------------------
        not_first = tmp / "mimetype_not_first.epub"
        build_epub(not_first, [_chapter()], mimetype_first=False)
        problems = check_source_epub(not_first)
        assert len(problems) == 1, problems
        assert problems[0].path == "mimetype", problems[0]
        assert "is not the first file" in problems[0].message, problems[0]

        # --- mimetype compressed ------------------------------------------
        compressed = tmp / "mimetype_compressed.epub"
        build_epub(compressed, [_chapter()], mimetype_compressed=True)
        problems = check_source_epub(compressed)
        assert len(problems) == 1, problems
        assert problems[0].path == "mimetype", problems[0]
        assert "is compressed" in problems[0].message, problems[0]

        # --- mimetype wrong content ---------------------------------------
        wrong = tmp / "mimetype_wrong.epub"
        build_epub(wrong, [_chapter()], mimetype_content="text/plain")
        problems = check_source_epub(wrong)
        assert len(problems) == 1, problems
        assert problems[0].path == "mimetype", problems[0]
        assert problems[0].message == "doesn't contain 'application/epub+zip'", problems[0]

        # --- mimetype missing ---------------------------------------------
        absent = tmp / "mimetype_absent.epub"
        build_epub(absent, [_chapter()], include_mimetype=False)
        problems = check_source_epub(absent)
        assert len(problems) == 1, problems
        assert problems[0].path == "mimetype", problems[0]
        assert "is missing from the EPUB" in problems[0].message, problems[0]

        # --- a real clean book --------------------------------------------
        assert check_source_epub(SHERLOCK) == [], check_source_epub(SHERLOCK)
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)

    print("test_epub_check: all assertions passed")


if __name__ == "__main__":
    main()
