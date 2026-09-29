"""Full-book run: EPUB -> draft translation -> consistency edit -> QA -> EPUB.

Thin CLI over :mod:`kannada_epub.pipeline`: it resolves the config and
book-relative paths, loads ``.env``, then builds the configured engines and
calls :func:`run_book`. All pipeline behavior (checkpoints/resume, QA retry,
EPUB writing, optional audiobook) lives in the importable module so a packaged
GUI can call it without shelling out to ``python scripts/...``.

Usage:
  python scripts/translate_book.py --config config/book.yaml
  python scripts/translate_book.py --config config/book.yaml --limit-chapters item4 --max-paragraphs 20 --batch-size 5

Resume unit is the chapter: completed chapters (all batch checkpoint files
present) are loaded from disk instead of re-translated, so a crash re-runs
at most the in-flight chapter.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kannada_epub.config import load_book_config
from kannada_epub.pipeline import RunOptions, build_components, run_book

ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else ROOT / p


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate a whole EPUB book to Kannada.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--epub", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit-chapters", nargs="*", default=None)
    parser.add_argument("--max-paragraphs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--no-epub", action="store_true", help="skip writing the translated EPUB"
    )
    parser.add_argument(
        "--audiobook", action="store_true", help="also synthesize a Kannada audiobook (.wav)"
    )
    args = parser.parse_args()

    if args.config:
        config_path = ROOT / args.config
    elif (ROOT / "config" / "book.yaml").exists():
        config_path = ROOT / "config" / "book.yaml"
    else:
        config_path = ROOT / "config" / "book.example.yaml"
    cfg = load_book_config(config_path)

    if args.epub:
        cfg.epub_path = args.epub
    if args.output_dir:
        cfg.output_dir = args.output_dir

    components = build_components(cfg, _resolve, need_tts=args.audiobook)
    options = RunOptions(
        limit_chapters=args.limit_chapters,
        max_paragraphs=args.max_paragraphs,
        batch_size=args.batch_size,
        write_epub=not args.no_epub,
        build_audiobook=args.audiobook,
    )
    run_book(cfg, resolve_path=_resolve, options=options, components=components)


if __name__ == "__main__":
    main()
