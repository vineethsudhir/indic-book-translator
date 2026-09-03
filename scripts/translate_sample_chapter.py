"""Prototype end-to-end run: load a real chapter from a real EPUB, translate
it with the CTranslate2 IndicTrans2 engine, and write English/Kannada pairs
to a file for manual quality review.

Deliberately does NOT run the glossary/consistency-editor layer yet — the
point of this pass is to judge raw IndicTrans2 output quality first, before
spending effort on continuity polish.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kannada_epub.epub_io import load_epub_chapters
from kannada_epub.translation import IndicTrans2Engine

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "en-indic-1b-ct2" / "en-indic-1b-ct2" / "ctranslate2_model"
EPUB_PATH = ROOT / "data" / "sherlock_holmes.epub"
OUTPUT_PATH = ROOT / "data" / "scandal_in_bohemia_sample.txt"
CHAPTER_ID = "item4"  # "A Scandal in Bohemia"
NUM_PARAGRAPHS = 20

if __name__ == "__main__":
    chapters = load_epub_chapters(EPUB_PATH)
    chapter = next(c for c in chapters if c.id == CHAPTER_ID)
    print(f"Chapter: {chapter.title} ({len(chapter.paragraphs)} paragraphs total)")

    sample = chapter.paragraphs[:NUM_PARAGRAPHS]
    english_texts = [p.text for p in sample]

    engine = IndicTrans2Engine(
        ct2_model_dir=MODEL_DIR,
        spm_src_path=MODEL_DIR / "vocab" / "model.SRC",
        spm_tgt_path=MODEL_DIR / "vocab" / "model.TGT",
        device="cpu",
        compute_type="int8",
    )

    print(f"Translating {len(english_texts)} paragraphs...")
    kannada_texts = engine.translate_paragraphs(english_texts, src_lang="eng_Latn", tgt_lang="kan_Knda")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(f"{chapter.title}\n{'=' * len(chapter.title)}\n\n")
        for i, (en, kn) in enumerate(zip(english_texts, kannada_texts), 1):
            f.write(f"[{i}] EN: {en}\n")
            f.write(f"[{i}] KN: {kn}\n\n")

    print(f"Wrote {len(sample)} paragraph pairs to {OUTPUT_PATH}")