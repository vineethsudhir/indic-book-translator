"""Build a narrated Kannada audiobook clip for the first 20 paragraphs of "A
Scandal in Bohemia" (the same span already reviewed in the proofing PDF).

Runs the full pipeline end to end: IndicTrans2 draft translation ->
consistency-edit + emotion tagging (BookTranslator) -> Indic Parler-TTS
narration in the user's preferred voice (Chetan), concatenated into one wav.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kannada_epub.audiobook_builder import build_audiobook
from kannada_epub.book_translator import BookTranslator
from kannada_epub.config import load_provider_config
from kannada_epub.consistency_editor import ConsistencyEditor
from kannada_epub.epub_io import load_epub_chapters
from kannada_epub.glossary import GlossaryStore
from kannada_epub.providers.factory import build_provider
from kannada_epub.translation import IndicTrans2Engine
from kannada_epub.tts import IndicParlerTTSEngine

ROOT = Path(__file__).resolve().parent.parent
CT2_MODEL_DIR = ROOT / "models" / "en-indic-1b-ct2" / "en-indic-1b-ct2" / "ctranslate2_model"
TTS_MODEL_DIR = ROOT / "models" / "indic-parler-tts"
EPUB_PATH = ROOT / "data" / "sherlock_holmes.epub"
CONFIG_PATH = ROOT / "config" / "consistency_editor.example.yaml"
OUTPUT_PATH = ROOT / "data" / "scandal_in_bohemia_audiobook.wav"

VOICE = "Chetan"
SAMPLE_PARAGRAPH_COUNT = 20

if __name__ == "__main__":
    all_chapters = load_epub_chapters(EPUB_PATH)
    by_id = {c.id: c for c in all_chapters}
    scandal = by_id["item4"]
    scandal.paragraphs = scandal.paragraphs[:SAMPLE_PARAGRAPH_COUNT]
    print(f"Chapter: {scandal.title!r} — {len(scandal.paragraphs)} paragraphs")

    print("Loading IndicTrans2 engine...")
    translation_engine = IndicTrans2Engine(
        ct2_model_dir=CT2_MODEL_DIR,
        spm_src_path=CT2_MODEL_DIR / "vocab" / "model.SRC",
        spm_tgt_path=CT2_MODEL_DIR / "vocab" / "model.TGT",
        device="cpu",
        compute_type="int8",
    )
    glossary_store = GlossaryStore(ROOT / "data" / "project.glossary.db")
    editor_config = load_provider_config(CONFIG_PATH)
    print(f"Consistency editor provider: {editor_config.provider}  model: {editor_config.model}")
    consistency_editor = ConsistencyEditor(build_provider(editor_config))

    book_translator = BookTranslator(
        translation_engine=translation_engine,
        glossary_store=glossary_store,
        consistency_editor=consistency_editor,
        batch_size=20,
    )

    print("Translating + consistency-editing + emotion-tagging...")
    batches = book_translator.translate_chapters([scandal])
    total_paragraphs = sum(len(b.edited_kannada) for b in batches)
    print(f"Got {total_paragraphs} edited, emotion-tagged paragraphs.")

    print(f"Loading Indic Parler-TTS ({VOICE})...")
    tts_engine = IndicParlerTTSEngine(model_dir=TTS_MODEL_DIR)
    print(f"Device: {tts_engine.device}  Sampling rate: {tts_engine.sampling_rate}")

    print("Synthesizing audiobook...")
    output_path = build_audiobook(
        batches=batches,
        voice=VOICE,
        tts_engine=tts_engine,
        output_path=OUTPUT_PATH,
    )

    import soundfile as sf

    info = sf.info(str(output_path))
    print(f"\nWrote {output_path} — {info.duration:.1f}s at {info.samplerate} Hz")