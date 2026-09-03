"""Full-book run: EPUB -> IndicTrans2 draft -> consistency edit ->
per-chapter EN/KN JSON outputs.

Usage:
  python scripts/translate_book.py --config config/book.yaml
  python scripts/translate_book.py --config config/book.yaml --limit-chapters item4 --max-paragraphs 20 --batch-size 5

Resume unit is the chapter: completed chapters (all batch checkpoint files
present) are loaded from disk instead of re-translated, so a crash re-runs
at most the in-flight chapter. Rolling context resets at every chapter
boundary (see BookTranslator), which makes chapter-level resume exact.
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kannada_epub.book_translator import BookTranslator, TranslatedBatch
from kannada_epub.config import load_book_config, load_provider_config
from kannada_epub.consistency_editor import ConsistencyEditor
from kannada_epub.epub_io import load_epub_chapters
from kannada_epub.glossary import GlossaryStore
from kannada_epub.providers.factory import build_provider
from kannada_epub.translation import build_translation_provider

ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else ROOT / p


def _batch_key(chapter_id: str, start: int) -> str:
    return f"{chapter_id}_{start:04d}.json"


def _expected_starts(n_paragraphs: int, batch_size: int) -> list[int]:
    return list(range(0, n_paragraphs, batch_size))


def _load_checkpoint(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _checkpoint_to_batch(data: dict) -> TranslatedBatch:
    return TranslatedBatch(**data)


def _write_checkpoint(checkpoints_dir: Path, batch: TranslatedBatch) -> None:
    (checkpoints_dir / _batch_key(batch.chapter_id, batch.paragraph_start)).write_text(
        json.dumps(asdict(batch), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate a whole EPUB book to Kannada.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--epub", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit-chapters", nargs="*", default=None)
    parser.add_argument("--max-paragraphs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    if args.config:
        config_path = ROOT / args.config
    elif (ROOT / "config" / "book.yaml").exists():
        config_path = ROOT / "config" / "book.yaml"
    else:
        config_path = ROOT / "config" / "book.example.yaml"
    cfg = load_book_config(config_path)
    epub_path = _resolve(args.epub or cfg.epub_path)
    output_dir = _resolve(args.output_dir or cfg.output_dir)
    batch_size = args.batch_size or cfg.batch_size
    limit_chapters = args.limit_chapters or cfg.limit_chapters
    max_paragraphs = args.max_paragraphs if args.max_paragraphs is not None else cfg.max_paragraphs_per_chapter

    chapters_dir = output_dir / "chapters"
    checkpoints_dir = output_dir / "checkpoints"
    chapters_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    chapters = load_epub_chapters(epub_path, exclude_ids=cfg.exclude_ids)
    if limit_chapters:
        wanted = set(limit_chapters)
        chapters = [c for c in chapters if c.id in wanted]
    if max_paragraphs is not None:
        for c in chapters:
            c.paragraphs = c.paragraphs[:max_paragraphs]
    print(f"Chapters to process: {[(c.id, len(c.paragraphs)) for c in chapters]}")

    t = cfg.translation
    if t.provider == "indictrans2_local":
        print(f"Translation: local IndicTrans2 ({cfg.ct2_model_dir})")
    else:
        print(f"Translation: cloud {t.provider} ({t.model} @ {t.base_url})")
    engine = build_translation_provider(t, ct2_model_dir=str(_resolve(cfg.ct2_model_dir)))
    glossary_store = GlossaryStore(_resolve(cfg.glossary_db))
    provider_cfg = load_provider_config(_resolve(cfg.provider_config))
    print(f"Consistency editor: {provider_cfg.provider} {provider_cfg.model}")
    editor = ConsistencyEditor(build_provider(provider_cfg))
    translator = BookTranslator(
        translation_engine=engine,
        glossary_store=glossary_store,
        consistency_editor=editor,
        batch_size=batch_size,
        register=cfg.tone_register,
    )

    manifest = {"epub": str(epub_path), "chapters": [], "skipped": []}
    for chapter in chapters:
        starts = _expected_starts(len(chapter.paragraphs), batch_size)
        cached = [checkpoints_dir / _batch_key(chapter.id, s) for s in starts]
        if cached and all(p.exists() for p in cached):
            batches = [_checkpoint_to_batch(_load_checkpoint(p)) for p in cached]
            print(f"[{chapter.id}] resume: {len(batches)} batches from checkpoints")
            manifest["skipped"].append(chapter.id)
        else:
            print(f"[{chapter.id}] translating {len(chapter.paragraphs)} paragraphs...")
            batches = translator.translate_chapters([chapter])
            for b in batches:
                _write_checkpoint(checkpoints_dir, b)

        chapter_out = {
            "chapter_id": chapter.id,
            "chapter_title": chapter.title,
            "batches": [asdict(b) for b in batches],
        }
        (chapters_dir / f"{chapter.id}.json").write_text(
            json.dumps(chapter_out, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        manifest["chapters"].append(
            {"id": chapter.id, "title": chapter.title, "paragraphs": len(chapter.paragraphs)}
        )

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Done. Outputs in {output_dir}")


if __name__ == "__main__":
    main()
