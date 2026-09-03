"""Prove the chapter-boundary continuity reset: process a few small batches
spanning a real story boundary in the Sherlock Holmes EPUB (A Scandal in
Bohemia -> The Red-Headed League) and show that rolling context carries
forward WITHIN a chapter but resets to empty at the next chapter."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kannada_epub.book_translator import BookTranslator
from kannada_epub.config import load_provider_config
from kannada_epub.consistency_editor import ConsistencyEditor
from kannada_epub.epub_io import load_epub_chapters
from kannada_epub.glossary import GlossaryStore
from kannada_epub.providers.factory import build_provider
from kannada_epub.translation import IndicTrans2Engine

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "en-indic-1b-ct2" / "en-indic-1b-ct2" / "ctranslate2_model"
EPUB_PATH = ROOT / "data" / "sherlock_holmes.epub"
CONFIG_PATH = ROOT / "config" / "consistency_editor.example.yaml"

if __name__ == "__main__":
    all_chapters = load_epub_chapters(EPUB_PATH)
    by_id = {c.id: c for c in all_chapters}

    # Trim to a handful of paragraphs each, purely to keep this a fast,
    # inspectable proof of the reset mechanism rather than a full run.
    scandal = by_id["item4"]
    scandal.paragraphs = scandal.paragraphs[:5]
    red_headed = by_id["item5"]
    red_headed.paragraphs = red_headed.paragraphs[:3]

    engine = IndicTrans2Engine(
        ct2_model_dir=MODEL_DIR,
        spm_src_path=MODEL_DIR / "vocab" / "model.SRC",
        spm_tgt_path=MODEL_DIR / "vocab" / "model.TGT",
        device="cpu",
        compute_type="int8",
    )
    glossary_store = GlossaryStore(ROOT / "data" / "project.glossary.db")
    editor_config = load_provider_config(CONFIG_PATH)
    editor = ConsistencyEditor(build_provider(editor_config))

    book_translator = BookTranslator(
        translation_engine=engine,
        glossary_store=glossary_store,
        consistency_editor=editor,
        batch_size=2,
        context_tail_paragraphs=2,
    )

    batches = book_translator.translate_chapters([scandal, red_headed])

    print(f"{'chapter':10s} {'paragraphs':12s} {'emotions':30s} prior_context_used")
    print("-" * 110)
    for b in batches:
        ctx = b.prior_context_used or "(empty — reset)"
        emotions = ",".join(b.edited_emotions)
        print(
            f"{b.chapter_id:10s} [{b.paragraph_start}:{b.paragraph_end}]".ljust(23),
            emotions[:28].ljust(30),
            ctx[:50],
        )