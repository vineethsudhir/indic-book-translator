"""Build a translated Kannada EPUB from the per-chapter JSON outputs.

Reads the same book config as ``scripts/translate_book.py``, re-runs the
loader (no ``max_paragraphs`` truncation — positions that were never
translated are simply absent), reads every ``<output_dir>/chapters/*.json``,
and writes ``<output_dir>/<epub stem>.kn.epub`` (override with ``--out``).

Usage:
  python scripts/build_epub.py --config config/book.yaml
  python scripts/build_epub.py --config config/book.yaml --out data/book.kn.epub
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kannada_epub.config import load_book_config
from kannada_epub.epub_io import load_epub_chapters
from kannada_epub.epub_writer import translations_from_batches, write_translated_epub

ROOT = Path(__file__).resolve().parent.parent


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else ROOT / p


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a Kannada EPUB from translated chapter JSON outputs."
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--epub", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--out", default=None)
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

    chapters = load_epub_chapters(epub_path, exclude_ids=cfg.exclude_ids)

    chapters_dir = output_dir / "chapters"
    batches: list[dict] = []
    for chapter_file in sorted(chapters_dir.glob("*.json")):
        data = json.loads(chapter_file.read_text(encoding="utf-8"))
        batches.extend(data.get("batches", []))

    translations = translations_from_batches(chapters, batches)

    total = 0
    for chapter in chapters:
        replaced = len(translations.get(chapter.id, {}))
        total += replaced
        if replaced:
            print(f"[{chapter.id}] replaced {replaced} paragraphs")
    print(f"Total paragraphs replaced: {total}")

    if args.out:
        out_path = _resolve(args.out)
    else:
        out_path = output_dir / f"{epub_path.stem}.kn.epub"

    write_translated_epub(epub_path, translations, out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
