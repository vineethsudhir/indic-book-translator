"""Shared reading-side helpers for the Library and the static site export.

These were the server's private helpers for turning an output folder plus its
``manifest.json`` into a title, an author and the translated output file names.
The site exporter needs exactly the same answers, so they live here and both
import them (rather than each carrying a copy that could drift).
"""

from __future__ import annotations

from pathlib import Path

from .app.paths import books_dir
from .app.settings import load_settings
from .epub_io import load_epub_chapters, read_epub_metadata
from .pipeline import output_file_names


def metadata(path: Path) -> tuple[str, str]:
    """The EPUB's ``(title, author)``, with fallbacks for missing metadata."""
    title, author = read_epub_metadata(path)
    return title or path.stem, author or "Unknown author"


def source_epub(folder: Path, manifest: dict) -> Path | None:
    """Find a source EPUB from the manifest, then the app's uploaded books."""
    source_name = Path(str(manifest.get("epub", ""))).name
    candidates = []
    if manifest.get("epub"):
        candidates.append(Path(manifest["epub"]))
    if source_name:
        candidates.append(books_dir() / source_name)
    candidates.append(books_dir() / f"{folder.name}.epub")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def source_titles(folder: Path, manifest: dict) -> dict[str, str]:
    """Map chapter ids to the source EPUB's (corrected) titles, if readable."""
    source = source_epub(folder, manifest)
    if source is None:
        return {}
    try:
        return {
            chapter.id: chapter.title or chapter.id
            for chapter in load_epub_chapters(source, exclude_ids=[])
        }
    except Exception:
        return {}


def book_outputs(folder: Path, manifest: dict) -> dict:
    """Output file names and translation coverage for one book folder.

    Manifests written before previews were labelled lack "preview" and
    "paragraphs_total"; for those the total is counted from the source EPUB,
    so an old preview isn't presented as a finished book.
    """
    stem = Path(manifest.get("epub", folder.name)).stem
    chapters = manifest.get("chapters", [])
    translated = int(manifest.get("paragraphs_translated", sum(c.get("paragraphs", 0) for c in chapters)))
    total = manifest.get("paragraphs_total")
    if total is None:
        total = translated
        source = source_epub(folder, manifest)
        if source is not None:
            try:
                wanted = {c.get("id") for c in chapters}
                # Older manifests have no "preview"/"paragraphs_total", but a
                # run that skipped sections records the exclude_ids it ran
                # with. Prefer those over today's global setting, so a skipped
                # chapter still present in the source isn't counted as work
                # the finished book never intended to do.
                recorded = manifest.get("exclude_ids")
                exclude_ids = recorded if isinstance(recorded, list) else load_settings().exclude_ids
                source_chapters = load_epub_chapters(source, exclude_ids=exclude_ids)
                total = sum(len(c.paragraphs) for c in source_chapters if c.id in wanted) or translated
            except Exception:
                pass
    preview = bool(manifest.get("preview", translated < total))
    target_language = str(manifest.get("target_language") or "kn")
    epub_name, audio_name = output_file_names(stem, preview, target_language)
    # Older runs always wrote .kn.* names; fall back to them when present.
    if not (folder / epub_name).is_file() and (folder / f"{stem}.kn.epub").is_file():
        epub_name = f"{stem}.kn.epub"
    if not (folder / audio_name).is_file() and (folder / f"{stem}.kn.wav").is_file():
        audio_name = f"{stem}.kn.wav"
    return {
        "epub": epub_name,
        "audiobook": audio_name,
        "preview": preview,
        "paragraphs_translated": translated,
        "paragraphs_total": total,
    }


def book_metadata(folder: Path, manifest: dict) -> tuple[str, str]:
    """The book's ``(title, author)`` from its source EPUB, else the folder."""
    source = source_epub(folder, manifest)
    if source is not None:
        try:
            return metadata(source)
        except Exception:
            pass
    return folder.name, "Unknown author"
