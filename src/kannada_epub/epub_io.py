import warnings
from dataclasses import dataclass
from pathlib import Path

import ebooklib
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from ebooklib import epub

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

BLOCK_TAGS = ["p", "li", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6"]


@dataclass
class Paragraph:
    index: int
    text: str


@dataclass
class Chapter:
    id: str
    title: str | None
    paragraphs: list[Paragraph]

    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.paragraphs)


def load_epub_chapters(
    path: str | Path,
    min_paragraph_chars: int = 1,
    exclude_ids: list[str] | set[str] | None = None,
) -> list[Chapter]:
    """Extract reading-order chapters as plain paragraph text.

    Deliberately minimal: no DOM/attribute preservation and no table
    extraction — just enough structure (chapter -> ordered paragraphs) to
    translate and review. Reassembly back into an .epub now lives in
    `kannada_epub.epub_writer`, which re-reads the source with this same
    parse (and same block enumeration) so a `Paragraph.index` maps back to
    its element. Full AST-preserving extraction (FR-1.3) and table handling
    (FR-3.2) remain separate, heavier pieces of work for later.

    A block tag that itself contains a block tag (e.g.
    ``<blockquote><p>…</p></blockquote>``) is skipped, so its text is not
    emitted twice; ``enumerate`` still advances over the skipped tag, keeping
    the index of every other paragraph stable for the writer.

    Reading order comes from the spine (not manifest iteration order), since
    the spine is what the EPUB spec actually guarantees reflects intended
    reading order. Spine items marked non-linear (EPUB3 nav/TOC documents,
    typically linear="no") are skipped — they are navigation aids, not
    content. Additional spine IDs (e.g. Project Gutenberg boilerplate
    chapters like "pg-header"/"pg-footer") can be skipped via `exclude_ids`.
    """
    book = epub.read_epub(str(path), options={"ignore_ncx": True})
    chapters: list[Chapter] = []
    excluded = set(exclude_ids or [])

    for idref, linear in book.spine:
        # ebooklib surfaces linear as 'yes'/'no' strings (EPUB spec values);
        # non-linear spine items are nav/TOC aids, not translatable content.
        if linear in (False, 0, "no", "No", "NO"):
            continue
        if idref in excluded:
            continue
        item = book.get_item_with_id(idref)
        if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue

        soup = BeautifulSoup(item.get_content(), "lxml")
        title_tag = soup.find(["h1", "h2", "title"])
        title = title_tag.get_text(strip=True) if title_tag else None

        paragraphs = []
        for i, tag in enumerate(soup.find_all(BLOCK_TAGS)):
            # A block nested inside another block would otherwise be extracted
            # twice (once as the parent's text, once as its own). Skip it, but
            # keep enumerate() advancing so all other indices are unchanged.
            if tag.find(BLOCK_TAGS) is not None:
                continue
            text = " ".join(tag.get_text().split())
            if len(text) >= min_paragraph_chars:
                paragraphs.append(Paragraph(i, text))

        if paragraphs:
            chapters.append(Chapter(id=item.get_id(), title=title, paragraphs=paragraphs))

    return chapters