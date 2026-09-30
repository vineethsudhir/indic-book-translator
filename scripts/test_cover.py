"""Smoke test for cover detection and rendering (GitHub issue #12).

Fast, offline, no models: detection reads only the PNG header, rendering uses
Pillow from the repo venv, and the writer test builds a fixture EPUB with
``scripts/epub_fixture.py``.

Run: .venv/bin/python scripts/test_cover.py
"""

import io
import shutil
import subprocess
import sys
import tempfile
import warnings
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from epub_fixture import build_epub

from kannada_epub.cover import (
    CoverText,
    is_gutenberg_generated_cover,
    is_translator_generated_cover,
    render_cover,
)
from kannada_epub.epub_writer import find_gutenberg_cover, write_translated_epub

ROOT = Path(__file__).resolve().parent.parent
FONTS = ROOT / "assets" / "fonts"
OPF_NS = "http://www.idpf.org/2007/opf"
CONTAINER_NS = "urn:oasis:names:tc:opendocument:xmlns:container"

CHAPTER = (
    '<?xml version="1.0" encoding="utf-8"?>'
    '<html xmlns="http://www.w3.org/1999/xhtml">'
    "<head><title>One</title></head>"
    "<body><h1>One</h1><p>Hello there.</p><p>Second.</p></body></html>"
)


def _png(width: int, height: int, color=(10, 20, 30)) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _jpeg(width: int, height: int) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (10, 20, 30)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _opf_path(zf: zipfile.ZipFile) -> str:
    import lxml.etree as ET

    root = ET.fromstring(zf.read("META-INF/container.xml"))
    return root.find(f".//{{{CONTAINER_NS}}}rootfile").get("full-path")


def _build_cover_fixture(path: Path, cover_name: str, cover_bytes: bytes,
                         media_type: str) -> None:
    build_epub(
        path,
        [
            {"id": "ch1", "href": "ch1.xhtml", "content": CHAPTER},
            {
                "id": "cover-img",
                "href": cover_name,
                "zip_name": cover_name,
                "content": cover_bytes,
                "media_type": media_type,
                "properties": "cover-image",
                "in_spine": False,
            },
        ],
        title="Fixture",
        creator="Tester",
    )


def _check_detection() -> None:
    png = _png(1600, 2400)
    assert is_gutenberg_generated_cover("OEBPS/123_456-cover.png", "image/png", png) is True
    assert is_gutenberg_generated_cover("123_456-cover.png", "image/png", png) is True
    # Wrong name.
    assert is_gutenberg_generated_cover("123_cover.png", "image/png", png) is False
    # Right name and size, but not a PNG.
    assert (
        is_gutenberg_generated_cover("123_456-cover.png", "image/jpeg", _jpeg(1600, 2400))
        is False
    )
    # Right name and type, wrong size.
    assert (
        is_gutenberg_generated_cover("123_456-cover.png", "image/png", _png(800, 1200))
        is False
    )
    # Not an image at all.
    assert is_gutenberg_generated_cover("123_456-cover.png", "image/png", b"not a png") is False
    # A real render is recognised as ours, not as a Gutenberg cover.
    rendered = render_cover(
        CoverText("ಶೀರ್ಷಿಕೆ", "ಲೇಖಕ", "Title", "Author"), FONTS
    )
    assert is_gutenberg_generated_cover("123_456-cover.png", "image/png", rendered) is True
    assert is_translator_generated_cover(rendered) is True
    assert is_translator_generated_cover(png) is False


def _check_render() -> None:
    from PIL import Image, features

    data = render_cover(
        CoverText("ಸುವರ್ಣ ದ್ವಾರ", "ಸರೋಜಿನಿ ನಾಯ್ಡು", "The Golden Threshold", "Sarojini Naidu"),
        FONTS,
    )
    image = Image.open(io.BytesIO(data))
    assert image.format == "PNG", image.format
    assert image.size == (1600, 2400), image.size
    colors = image.convert("RGB").getcolors(maxcolors=2_000_000)
    assert colors is not None and len(colors) > 1, "cover is a single colour"
    assert is_translator_generated_cover(data) is True

    # With raqm unavailable, no Kannada is drawn but a valid cover is produced.
    original = features.check
    features.check = lambda name, *a, **k: False if name == "raqm" else original(name, *a, **k)
    try:
        no_raqm = render_cover(
            CoverText("ಸುವರ್ಣ ದ್ವಾರ", "ಸರೋಜಿನಿ ನಾಯ್ಡು", "The Golden Threshold", None),
            FONTS,
        )
    finally:
        features.check = original
    image2 = Image.open(io.BytesIO(no_raqm))
    assert image2.format == "PNG" and image2.size == (1600, 2400)
    colors2 = image2.convert("RGB").getcolors(maxcolors=2_000_000)
    assert colors2 is not None and len(colors2) > 1

    # A missing Kannada title also renders a valid (English) cover.
    english = render_cover(CoverText(None, None, "English Only", "Someone"), FONTS)
    image3 = Image.open(io.BytesIO(english))
    assert image3.format == "PNG" and image3.size == (1600, 2400)


def _entry_compress_types(zf: zipfile.ZipFile) -> dict[str, int]:
    return {info.filename: info.compress_type for info in zf.infolist()}


def _check_writer(tmp: Path) -> None:
    cover_name = "111_222-cover.png"
    cover_png = _png(1600, 2400)
    source = tmp / "generated.epub"
    _build_cover_fixture(source, cover_name, cover_png, "image/png")

    with zipfile.ZipFile(source) as zf:
        opf_path = _opf_path(zf)
        cover_zip = opf_path.rsplit("/", 1)[0] + "/" + cover_name
        assert cover_zip in zf.namelist(), zf.namelist()
        src_infos = {info.filename: info for info in zf.infolist()}
        src_data = {name: zf.read(name) for name in zf.namelist()}

    # find_gutenberg_cover sees the source's generated cover.
    assert find_gutenberg_cover(source) == cover_zip

    # --- strip_gutenberg=True replaces the generated cover ----------------
    out = tmp / "generated.kn.epub"
    write_translated_epub(source, {}, out, strip_gutenberg=True)
    assert find_gutenberg_cover(out) is None

    with zipfile.ZipFile(out) as zf:
        out_infos = {info.filename: info for info in zf.infolist()}
        assert cover_zip in out_infos, out_infos.keys()
        assert zf.read(cover_zip) != cover_png, "cover was not replaced"
        replaced = zf.read(cover_zip)
        assert is_translator_generated_cover(replaced)
        from PIL import Image

        assert Image.open(io.BytesIO(replaced)).size == (1600, 2400)
        assert out_infos[cover_zip].compress_type == src_infos[cover_zip].compress_type

        # Everything strip mode does not rewrite is byte-identical.
        rewritten = {opf_path, cover_zip}
        for name in src_infos:
            if name.endswith((".xhtml", ".html", ".htm", ".ncx")):
                rewritten.add(name)
        for name in src_infos:
            if name in rewritten:
                continue
            assert src_data[name] == zf.read(name), f"entry changed: {name}"

    # --- strip_gutenberg=False keeps the cover byte-identical -------------
    plain = tmp / "generated.plain.epub"
    write_translated_epub(source, {}, plain, strip_gutenberg=False)
    with zipfile.ZipFile(plain) as zf:
        assert zf.read(cover_zip) == cover_png
    # The generated cover is still detected when not stripped.
    assert find_gutenberg_cover(plain) == cover_zip

    # --- a scanned (JPEG) cover is kept even when stripping ---------------
    jpg_name = "111_cover.jpg"
    jpg = _jpeg(1200, 1800)
    scanned = tmp / "scanned.epub"
    _build_cover_fixture(scanned, jpg_name, jpg, "image/jpeg")
    with zipfile.ZipFile(scanned) as zf:
        scanned_opf = _opf_path(zf)
        jpg_zip = scanned_opf.rsplit("/", 1)[0] + "/" + jpg_name
    assert find_gutenberg_cover(scanned) is None
    scanned_out = tmp / "scanned.kn.epub"
    write_translated_epub(scanned, {}, scanned_out, strip_gutenberg=True)
    with zipfile.ZipFile(scanned_out) as zf:
        assert zf.read(jpg_zip) == jpg, "scanned cover was modified"

    # --- no Pillow: the write succeeds and the cover is unchanged ---------
    no_pil_source = tmp / "no_pil.epub"
    _build_cover_fixture(no_pil_source, cover_name, cover_png, "image/png")
    with zipfile.ZipFile(no_pil_source) as zf:
        no_pil_opf = _opf_path(zf)
        no_pil_cover = no_pil_opf.rsplit("/", 1)[0] + "/" + cover_name
    no_pil_out = tmp / "no_pil.kn.epub"
    saved_pil = sys.modules.get("PIL")
    sys.modules["PIL"] = None  # type: ignore[assignment]
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            write_translated_epub(no_pil_source, {}, no_pil_out, strip_gutenberg=True)
        assert any("cover" in str(item.message).lower() for item in caught), caught
    finally:
        if saved_pil is None:
            del sys.modules["PIL"]
        else:
            sys.modules["PIL"] = saved_pil
    with zipfile.ZipFile(no_pil_out) as zf:
        assert zf.read(no_pil_cover) == cover_png, "cover changed without Pillow"
    assert find_gutenberg_cover(no_pil_out) == no_pil_cover


def _check_lazy_pillow_import() -> None:
    src = str(ROOT / "src")
    code = (
        "import sys\n"
        f"sys.path.insert(0, {src!r})\n"
        "import kannada_epub.epub_writer\n"
        "import kannada_epub.pipeline\n"
        "assert 'PIL' not in sys.modules, 'PIL imported eagerly'\n"
        "print('lazy-pillow-ok')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert proc.returncode == 0, (
        f"subprocess failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    assert "lazy-pillow-ok" in proc.stdout


def main() -> None:
    _check_detection()
    _check_render()
    tmp = Path(tempfile.mkdtemp())
    try:
        _check_writer(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    _check_lazy_pillow_import()
    print("test_cover: all assertions passed")


if __name__ == "__main__":
    main()
