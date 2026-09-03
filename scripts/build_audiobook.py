"""Sample-span audiobook demo: translate + consistency-edit + emotion-tag the
first chapters of "A Scandal in Bohemia", then narrate to one WAV.

Translation and TTS engines come from config/book.yaml, so this works fully
local (IndicTrans2 + Parler-TTS) or fully cloud (e.g. Sarvam) with no code
changes — see the `translation:` and `tts:` sections in
config/book.example.yaml.

Runs the full pipeline end to end: draft translation -> consistency-edit +
emotion tagging (BookTranslator) -> narration in the configured voice,
concatenated into one wav.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:
    from dotenv import load_dotenv

    ROOT = Path(__file__).resolve().parent.parent
    load_dotenv(ROOT / ".env")
except ImportError:
    ROOT = Path(__file__).resolve().parent.parent

from kannada_epub.audiobook_builder import build_audiobook
from kannada_epub.book_translator import BookTranslator
from kannada_epub.config import load_book_config, load_provider_config
from kannada_epub.consistency_editor import ConsistencyEditor
from kannada_epub.epub_io import load_epub_chapters
from kannada_epub.glossary import GlossaryStore
from kannada_epub.providers.factory import build_provider
from kannada_epub.translation import build_translation_provider
from kannada_epub.tts import build_tts_provider

CHAPTER_ID = "item4"  # "A Scandal in Bohemia"
SAMPLE_PARAGRAPH_COUNT = 20


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else ROOT / p


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a sample-span narrated Kannada audiobook.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--voice", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.config:
        config_path = ROOT / args.config
    elif (ROOT / "config" / "book.yaml").exists():
        config_path = ROOT / "config" / "book.yaml"
    else:
        config_path = ROOT / "config" / "book.example.yaml"
    cfg = load_book_config(config_path)

    all_chapters = load_epub_chapters(_resolve(cfg.epub_path), exclude_ids=cfg.exclude_ids)
    by_id = {c.id: c for c in all_chapters}
    scandal = by_id[CHAPTER_ID]
    scandal.paragraphs = scandal.paragraphs[:SAMPLE_PARAGRAPH_COUNT]
    print(f"Chapter: {scandal.title!r} — {len(scandal.paragraphs)} paragraphs")

    t = cfg.translation
    print(f"Translation: {t.provider} ({t.model})")
    translation_engine = build_translation_provider(
        t, ct2_model_dir=str(_resolve(cfg.ct2_model_dir))
    )
    glossary_store = GlossaryStore(_resolve(cfg.glossary_db))
    editor_config = load_provider_config(_resolve(cfg.provider_config))
    print(f"Consistency editor: {editor_config.provider}  model: {editor_config.model}")
    consistency_editor = ConsistencyEditor(build_provider(editor_config))

    book_translator = BookTranslator(
        translation_engine=translation_engine,
        glossary_store=glossary_store,
        consistency_editor=consistency_editor,
        batch_size=cfg.batch_size,
    )

    print("Translating + consistency-editing + emotion-tagging...")
    batches = book_translator.translate_chapters([scandal])
    total_paragraphs = sum(len(b.edited_kannada) for b in batches)
    print(f"Got {total_paragraphs} edited, emotion-tagged paragraphs.")

    tts_cfg = cfg.tts
    voice = args.voice or tts_cfg.voice
    print(f"TTS: {tts_cfg.provider}  voice: {voice}")
    tts_engine = build_tts_provider(
        tts_cfg, local_model_dir=str(ROOT / "models" / "indic-parler-tts")
    )
    print(f"Sampling rate: {tts_engine.sampling_rate}")

    print("Synthesizing audiobook...")
    output_path = build_audiobook(
        batches=batches,
        voice=voice,
        tts_engine=tts_engine,
        output_path=_resolve(args.output or "data/scandal_in_bohemia_audiobook.wav"),
    )

    import soundfile as sf

    info = sf.info(str(output_path))
    print(f"\nWrote {output_path} — {info.duration:.1f}s at {info.samplerate} Hz")
