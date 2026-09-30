"""Detect and render Project Gutenberg's generated covers.

Project Gutenberg generates a plain cover for many books: a 1600x2400 PNG
named ``<digits>_<ebook number>-cover.png``. It carries the English title,
author and a large "Project Gutenberg" label, so it should not survive a
translated, de-branded output. Other books instead include a scan of the
original printed cover, which is public domain and must be kept.

Detection is deliberately cheap: no Pillow, just the IHDR size from the PNG
header, so it works everywhere the writer works. Rendering is the only part
that needs Pillow, and it is imported lazily inside :func:`render_cover`.
Rendered covers carry a small ``tEXt`` marker chunk so
:func:`is_translator_generated_cover` can tell our own cover apart from a real
Gutenberg one at the same path (both are 1600x2400 PNGs).
"""

from __future__ import annotations

import io
import posixpath
import re
import struct
from dataclasses import dataclass
from pathlib import Path

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_GENERATED_COVER_RE = re.compile(r"^\d+_\d+-cover\.png$", re.IGNORECASE)
_COVER_WIDTH = 1600
_COVER_HEIGHT = 2400

# A tEXt chunk written by render_cover; a real Gutenberg cover never has it.
_MARKER_KEY = b"Software"
TRANSLATOR_COVER_MARKER = b"kannada-book-translator-generated-cover"

# Calm, print-like palette.
_BACKGROUND = (24, 34, 54)
_BORDER = (198, 166, 100)
_TITLE_COLOR = (247, 245, 238)
_AUTHOR_COLOR = (226, 220, 205)
_ENGLISH_COLOR = (232, 228, 216)
_FOOTER_COLOR = (172, 168, 158)
_FOOTER_EN = "Machine translation"


@dataclass(frozen=True)
class CoverText:
    """Text for a rendered cover; Kannada entries are optional."""

    title_kn: str | None
    author_kn: str | None
    title_en: str
    author_en: str | None


def _png_size(data: bytes) -> tuple[int, int] | None:
    """The ``(width, height)`` from a PNG's IHDR chunk, or ``None``."""
    if len(data) < 24 or not data.startswith(_PNG_MAGIC):
        return None
    if data[12:16] != b"IHDR":
        return None
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def is_gutenberg_generated_cover(zip_name: str, media_type: str, data: bytes) -> bool:
    """True for the PNGs Project Gutenberg's cover generator produces.

    Requires an ``image/png`` item whose base name is
    ``<digits>_<ebook number>-cover.png`` and whose header is 1600x2400.
    A scanned original cover (usually a JPEG) or any other image is ``False``.
    """
    if media_type != "image/png":
        return False
    if not _GENERATED_COVER_RE.match(posixpath.basename(zip_name)):
        return False
    return _png_size(data) == (_COVER_WIDTH, _COVER_HEIGHT)


def is_translator_generated_cover(data: bytes) -> bool:
    """True if ``data`` is a PNG written by :func:`render_cover`.

    Rendered covers keep the same 1600x2400 size and file name as the
    Gutenberg cover they replace, so a marker chunk is the only way to tell
    them apart without OCR.
    """
    if not data.startswith(_PNG_MAGIC):
        return False
    pos = 8
    while pos + 8 <= len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        chunk_type = data[pos + 4 : pos + 8]
        chunk = data[pos + 8 : pos + 8 + length]
        if chunk_type == b"tEXt":
            payload = chunk.split(b"\x00", 1)
            if payload and payload[0] == _MARKER_KEY:
                text = payload[1] if len(payload) > 1 else b""
                if text == TRANSLATOR_COVER_MARKER:
                    return True
        if chunk_type in (b"IDAT", b"IEND"):
            break
        pos += 12 + length
    return False


# ---------------------------------------------------------------------------
# Rendering (Pillow imported lazily; see AGENTS.md)
# ---------------------------------------------------------------------------
def _font_has_latin(font) -> bool:
    """Heuristic: a font lacks distinct Latin glyphs when ``A`` == ``B`` ink."""
    try:
        return bytes(font.getmask("A")) != bytes(font.getmask("B"))
    except Exception:  # noqa: BLE001 — a probe must never break rendering
        return False


def _measure_width(draw, text: str, font) -> float:
    return draw.textlength(text, font=font)


def _wrap_lines(draw, text: str, font, max_width: float) -> list[str]:
    """Greedy word wrap by measured width; never cuts a word."""
    words = text.split()
    if not words:
        return []
    lines = [words[0]]
    for word in words[1:]:
        candidate = f"{lines[-1]} {word}"
        if _measure_width(draw, candidate, font) <= max_width:
            lines[-1] = candidate
        else:
            lines.append(word)
    return lines


def _fit_block(
    draw,
    text: str,
    make_font,
    max_width: float,
    max_height: float,
    start_size: int,
    min_size: int,
):
    """Largest font size whose wrapped text fits ``max_width`` x ``max_height``.

    Returns ``(font, lines)``; ``lines`` is empty for empty text. If even
    ``min_size`` overflows, the ``min_size`` layout is returned anyway.
    """
    if not text:
        return make_font(min_size), []
    size = start_size
    fallback = None
    while size >= min_size:
        font = make_font(size)
        lines = _wrap_lines(draw, text, font, max_width)
        ascent, descent = font.getmetrics()
        line_height = ascent + descent
        gap = max(4, int(size * 0.2))
        widths = [_measure_width(draw, line, font) for line in lines]
        total_height = len(lines) * line_height + (len(lines) - 1) * gap
        if max(widths) <= max_width and total_height <= max_height:
            return font, lines
        fallback = (font, lines)
        size -= 2
    return fallback if fallback is not None else (make_font(min_size), [])


def _draw_block(
    draw,
    text: str,
    make_font,
    box: tuple[float, float, float, float],
    start_size: int,
    min_size: int,
    fill: tuple[int, int, int],
) -> None:
    """Draw wrapped, centred text inside ``box`` (left, top, right, bottom)."""
    if not text:
        return
    left, top, right, bottom = box
    font, lines = _fit_block(
        draw, text, make_font, right - left, bottom - top, start_size, min_size
    )
    if not lines:
        return
    ascent, descent = font.getmetrics()
    size = getattr(font, "size", start_size)
    line_height = ascent + descent
    gap = max(4, int(size * 0.2))
    total_height = len(lines) * line_height + (len(lines) - 1) * gap
    center_x = (left + right) / 2
    y = top + (bottom - top - total_height) / 2 + line_height / 2
    for line in lines:
        draw.text((center_x, y), line, font=font, fill=fill, anchor="mm")
        y += line_height + gap


def _draw_segments(
    draw,
    segments: list[str],
    font_factory,
    box: tuple[float, float, float, float],
    start_size: int,
    min_size: int,
    fill: tuple[int, int, int],
) -> None:
    """Draw a single centred line made of runs that use different fonts.

    ``font_factory(index, size)`` returns the font for segment ``index``; the
    size shrinks until every run fits on one line. Used for the footer, whose
    Kannada and Latin parts need different fonts.
    """
    if not segments:
        return
    left, top, right, bottom = box
    max_width = right - left
    size = start_size
    fonts = []
    while size >= min_size:
        fonts = [font_factory(i, size) for i in range(len(segments))]
        total = sum(
            draw.textlength(s, font=f)
            for s, f in zip(segments, fonts, strict=True)
        )
        if total <= max_width:
            break
        size -= 2
    x = (left + right) / 2 - sum(
        draw.textlength(s, font=f) for s, f in zip(segments, fonts, strict=True)
    ) / 2
    center_y = (top + bottom) / 2
    for segment, font in zip(segments, fonts, strict=True):
        draw.text((x, center_y), segment, font=font, fill=fill, anchor="lm")
        x += draw.textlength(segment, font=font)


def render_cover(text: CoverText, font_dir: Path) -> bytes:
    """Render a 1600x2400 RGB PNG cover; returns the PNG bytes.

    Kannada is only drawn when Raqm shaping is available (otherwise it would
    render as broken, unjoined glyphs); without it, the English title and
    author are drawn large instead. Pillow is imported here, not at module
    import, so every other code path works without it.
    """
    from PIL import Image, ImageDraw, ImageFont, PngImagePlugin, features

    font_dir = Path(font_dir)
    bold_path = font_dir / "NotoSansKannada-Bold.ttf"
    regular_path = font_dir / "NotoSansKannada-Regular.ttf"
    for path in (bold_path, regular_path):
        if not path.exists():
            raise FileNotFoundError(f"cover font asset not found: {path}")

    raqm = bool(features.check("raqm"))
    layout_engine = ImageFont.Layout.RAQM if raqm else ImageFont.Layout.BASIC

    image = Image.new("RGB", (_COVER_WIDTH, _COVER_HEIGHT), _BACKGROUND)
    draw = ImageDraw.Draw(image)
    inset = 70
    draw.rectangle(
        [inset, inset, _COVER_WIDTH - inset - 1, _COVER_HEIGHT - inset - 1],
        outline=_BORDER,
        width=6,
    )

    def kannada_font(size: int, *, bold: bool = False):
        path = bold_path if bold else regular_path
        if raqm:
            return ImageFont.truetype(str(path), size, layout_engine=layout_engine)
        return ImageFont.truetype(str(path), size)

    latin_support: bool | None = None

    def english_font(size: int):
        nonlocal latin_support
        if latin_support is None:
            probe = ImageFont.truetype(str(regular_path), max(size, 24))
            latin_support = _font_has_latin(probe)
        if latin_support:
            return ImageFont.truetype(str(regular_path), size)
        return ImageFont.load_default(size=size)

    kannada_title = (text.title_kn or "").strip()
    kannada_author = (text.author_kn or "").strip()

    if raqm and kannada_title:
        _draw_block(
            draw, kannada_title, lambda size: kannada_font(size, bold=True),
            (160, 360, 1440, 1330), 210, 64, _TITLE_COLOR,
        )
        if kannada_author:
            _draw_block(
                draw, kannada_author, lambda size: kannada_font(size),
                (260, 1360, 1340, 1650), 104, 44, _AUTHOR_COLOR,
            )
        if text.title_en:
            _draw_block(
                draw, text.title_en, english_font,
                (200, 1880, 1400, 2010), 60, 30, _ENGLISH_COLOR,
            )
        if text.author_en:
            _draw_block(
                draw, text.author_en, english_font,
                (260, 2020, 1340, 2130), 50, 26, _ENGLISH_COLOR,
            )
    else:
        # No Kannada shaping (or no Kannada title): English large instead.
        if text.title_en:
            _draw_block(
                draw, text.title_en, english_font,
                (160, 470, 1440, 1600), 200, 60, _TITLE_COLOR,
            )
        if text.author_en:
            _draw_block(
                draw, text.author_en, english_font,
                (220, 1690, 1380, 2010), 120, 48, _AUTHOR_COLOR,
            )

    if raqm:
        _draw_segments(
            draw,
            ["ಯಂತ್ರ ಅನುವಾದ", " · Machine translation"],
            lambda index, size: (kannada_font(size) if index == 0 else english_font(size)),
            (180, 2180, 1420, 2300),
            46,
            26,
            _FOOTER_COLOR,
        )
    else:
        _draw_block(
            draw, _FOOTER_EN, english_font, (180, 2180, 1420, 2300), 46, 26, _FOOTER_COLOR
        )

    png_info = PngImagePlugin.PngInfo()
    png_info.add_text(_MARKER_KEY.decode("ascii"), TRANSLATOR_COVER_MARKER.decode("ascii"))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", pnginfo=png_info)
    return buffer.getvalue()
