"""Importable full-book pipeline: EPUB -> translation -> consistency edit ->
QA -> translated EPUB (+ optional audiobook).

This is the library form of what used to live entirely in
``scripts/translate_book.py``. It is safe to import and run in a cloud-only
install: every local-ML import (IndicTrans2, Parler-TTS, local embeddings) is
lazy, so `torch`/`ctranslate2`/... are only needed when a local provider is
actually selected.

The GUI and tests inject their own engines through :class:`PipelineComponents`;
the CLI builds the real ones via :func:`build_components`. All status output
goes through the `progress` callback, never `print`, and the module never reads
``.env`` or any file outside the paths named in the config — callers own the
environment.
"""

from __future__ import annotations

import json
import posixpath
import threading
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from .audiobook_builder import build_audiobook
from .book_translator import BookTranslator, TranslatedBatch
from .config import BookConfig, load_provider_config
from .consistency_editor import ConsistencyEditor
from .cover import CoverText
from .edition_check import edition_notes, edition_payload
from .epub_check import check_source_epub
from .epub_io import Chapter, load_epub_chapters, read_epub_metadata
from .epub_writer import (
    _clean_title_text,
    find_gutenberg_cover,
    find_gutenberg_mentions,
    translations_from_batches,
    write_translated_epub,
)
from .glossary import GlossaryStore
from .providers.factory import build_provider
from .qa import (
    FLAGGED_FOR_REVIEW,
    RETRY,
    QAResult,
    build_qa,
    evaluate,
    write_qa_report,
)
from .qa.base import BackTranslator, Embedder
from .translation import TranslationProvider, build_translation_provider
from .tts import TTSProvider, build_tts_provider


@dataclass
class RunOptions:
    limit_chapters: list[str] | None = None
    max_paragraphs: int | None = None
    batch_size: int | None = None  # None = cfg.batch_size
    write_epub: bool = True
    build_audiobook: bool = False


@dataclass
class PipelineComponents:
    """Engines the pipeline runs with.

    Letting callers inject these keeps the pipeline testable (and the GUI
    able to reuse already-built engines) without touching config-driven
    construction in :func:`build_components`.
    """

    translation_engine: TranslationProvider
    editor: ConsistencyEditor
    glossary_store: GlossaryStore
    qa: tuple[BackTranslator, Embedder] | None = None
    tts: TTSProvider | None = None


@dataclass
class RunResult:
    output_dir: Path
    chapters: list[str]
    epub_path: Path | None
    qa_report_path: Path | None
    audiobook_path: Path | None
    cancelled: bool


# ---------------------------------------------------------------------------
# Checkpoint helpers (chapter-level resume)
# ---------------------------------------------------------------------------
def _batch_key(chapter_id: str, start: int) -> str:
    return f"{chapter_id}_{start:04d}.json"


def _load_checkpoint(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _checkpoint_to_batch(data: dict) -> TranslatedBatch:
    return TranslatedBatch(**data)


def _matching_checkpoints(
    checkpoints_dir: Path, chapter: Chapter
) -> list[TranslatedBatch] | None:
    """The chapter's saved batches, only if they cover exactly the paragraphs
    being translated now; otherwise None (re-translate the chapter).

    The batches are read as a chain of saved checkpoints: start at position 0,
    load the checkpoint whose name encodes that start, require its span to
    begin there and its source text to match the chapter's paragraphs, then
    continue from its end until the whole chapter is covered. A checkpoint's
    file name only encodes its start position, so a batch saved by a preview
    run (e.g. 2 paragraphs) has the same name as the first batch of a full
    run; matching the span and source text keeps a resume from splicing
    preview output into a full book.

    Because the chain is read from the checkpoints themselves, a resume works
    regardless of the `batch_size` the checkpoints were saved with — retrying
    a chapter with a smaller batch size still reuses finished work.
    """
    n = len(chapter.paragraphs)
    batches: list[TranslatedBatch] = []
    pos = 0
    while pos < n:
        path = checkpoints_dir / _batch_key(chapter.id, pos)
        if not path.exists():
            return None
        batch = _checkpoint_to_batch(_load_checkpoint(path))
        if (
            batch.paragraph_start != pos
            or batch.paragraph_end <= pos
            or batch.paragraph_end > n
        ):
            return None
        expected_source = [p.text for p in chapter.paragraphs[pos : batch.paragraph_end]]
        if (
            batch.source_english != expected_source
            or len(batch.edited_kannada) != batch.paragraph_end - pos
        ):
            return None
        batches.append(batch)
        pos = batch.paragraph_end
    return batches or None


def _write_checkpoint(checkpoints_dir: Path, batch: TranslatedBatch) -> None:
    (checkpoints_dir / _batch_key(batch.chapter_id, batch.paragraph_start)).write_text(
        json.dumps(asdict(batch), ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Component construction
# ---------------------------------------------------------------------------
def build_components(
    cfg: BookConfig, resolve_path: Callable[[str], Path], *, need_tts: bool
) -> PipelineComponents:
    """Build the configured engines (translation, editor, glossary, QA, TTS).

    ``resolve_path`` maps a config-relative path string to an absolute
    ``Path`` (the same role ``_resolve`` plays in the CLI). QA is built via
    :func:`kannada_epub.qa.build_qa` (which returns ``None`` when disabled),
    and TTS only when `need_tts`.
    """
    translation_engine = build_translation_provider(
        cfg.translation, ct2_model_dir=str(resolve_path(cfg.ct2_model_dir))
    )
    glossary_store = GlossaryStore(resolve_path(cfg.glossary_db))
    provider_cfg = load_provider_config(resolve_path(cfg.provider_config))
    editor = ConsistencyEditor(build_provider(provider_cfg))

    qa = build_qa(
        cfg.qa,
        default_provider_config_path=cfg.provider_config,
        resolve_path=resolve_path,
    )

    tts: TTSProvider | None = None
    if need_tts:
        tts = build_tts_provider(
            cfg.tts, local_model_dir=str(resolve_path("models/indic-parler-tts"))
        )

    return PipelineComponents(
        translation_engine=translation_engine,
        editor=editor,
        glossary_store=glossary_store,
        qa=qa,
        tts=tts,
    )


# ---------------------------------------------------------------------------
# QA
# ---------------------------------------------------------------------------
def output_file_names(epub_stem: str, preview: bool) -> tuple[str, str]:
    """(EPUB, audiobook) file names for a run's outputs.

    Previews get their own names so a partly translated book is never mistaken
    for the whole one (untranslated paragraphs stay English in the EPUB).
    """
    suffix = ".kn.preview" if preview else ".kn"
    return f"{epub_stem}{suffix}.epub", f"{epub_stem}{suffix}.wav"


def _assign_edited(batches: list[TranslatedBatch], edited: list[str]) -> None:
    """Distribute a chapter-flat ``edited`` list back into its batches."""
    expected = sum(batch.paragraph_end - batch.paragraph_start for batch in batches)
    if len(edited) != expected:
        raise ValueError(
            f"Refusing to misalign: {len(edited)} edited paragraphs for {expected} "
            "translated paragraphs"
        )
    offset = 0
    for batch in batches:
        count = batch.paragraph_end - batch.paragraph_start
        batch.edited_kannada = edited[offset : offset + count]
        offset += count


def _qa_cache_path(output_dir: Path, chapter_id: str) -> Path:
    return output_dir / "qa" / f"{chapter_id}.json"


def _run_chapter_qa(
    cfg: BookConfig,
    components: PipelineComponents,
    chapter: Chapter,
    batches: list[TranslatedBatch],
    output_dir: Path,
    progress: Callable[[str], None],
) -> list[QAResult]:
    """Score one chapter, retrying RETRY paragraphs at most once, and cache.

    Returns the (possibly retried) final QA results for the chapter. On a
    resumed chapter the cache is used and no engines are called.
    """
    assert components.qa is not None
    back_translator, embedder = components.qa
    cache_path = _qa_cache_path(output_dir, chapter.id)
    source_english = [text for batch in batches for text in batch.source_english]

    # Reuse cached QA only for exactly the same English paragraphs: a cache
    # from an earlier preview (fewer paragraphs) or an older run of different
    # text must be recomputed, or its Kannada would overwrite the wrong slots.
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("source_english") == source_english:
            results = [QAResult(**item) for item in payload["results"]]
            _assign_edited(batches, list(payload["kannada"]))
            progress(f"[{chapter.id}] QA: resumed {len(results)} results from cache")
            return results
        progress(f"[{chapter.id}] QA: cache is for different paragraphs; re-checking")

    edited = [text for batch in batches for text in batch.edited_kannada]
    paragraph_indices = [p.index for p in chapter.paragraphs]

    results = evaluate(
        source_english,
        edited,
        chapter=chapter.id,
        paragraph_indices=paragraph_indices,
        back_translator=back_translator,
        embedder=embedder,
        pass_threshold=cfg.qa.pass_threshold,
        flag_threshold=cfg.qa.flag_threshold,
    )

    # One retry per RETRY paragraph: re-translate just those English
    # paragraphs and keep the new text/score only when it scores higher.
    retry_positions = [i for i, result in enumerate(results) if result.status == RETRY]
    if retry_positions:
        retry_english = [source_english[i] for i in retry_positions]
        new_drafts = components.translation_engine.translate_paragraphs(
            retry_english, "eng_Latn", "kan_Knda"
        )
        retried_results = evaluate(
            retry_english,
            new_drafts,
            chapter=chapter.id,
            paragraph_indices=[paragraph_indices[i] for i in retry_positions],
            back_translator=back_translator,
            embedder=embedder,
            pass_threshold=cfg.qa.pass_threshold,
            flag_threshold=cfg.qa.flag_threshold,
        )
        for k, position in enumerate(retry_positions):
            if retried_results[k].similarity_score > results[position].similarity_score:
                edited[position] = new_drafts[k]
                results[position] = retried_results[k]

    _assign_edited(batches, edited)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "source_english": source_english,
                "results": [asdict(r) for r in results],
                "kannada": edited,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def _check_source(
    epub_path: Path, progress: Callable[[str], None]
) -> list[dict]:
    """Report defects in the source EPUB; return them for the run manifest.

    The check must never stop a run: an unexpected failure is reported and
    treated as no problems. Problems in the source carry over into the output
    (untouched files are copied byte-for-byte), so users can be told before
    translating that the translated book will inherit them.
    """
    try:
        problems = check_source_epub(epub_path)
    except Exception as exc:  # noqa: BLE001 — never let the check stop a run
        progress(f"Source EPUB check skipped: {exc}")
        return []
    if not problems:
        return []
    count = len(problems)
    progress(
        f"Source EPUB check: {count} problem(s) in the source file itself; "
        "they will carry over into the translated EPUB"
    )
    for problem in problems[:10]:
        basename = posixpath.basename(problem.path) or epub_path.name
        progress(f"  - {basename}: {problem.message}")
    if count > 10:
        progress(f"  … and {count - 10} more")
    return [{"path": problem.path, "message": problem.message} for problem in problems]


def _check_edition(
    epub_path: Path, chapters: list[Chapter], progress: Callable[[str], None]
) -> dict | None:
    """Report front-matter copyright risks; return them for the run manifest.

    Like the source check, this must never stop a run: an unexpected failure
    is reported and treated as no notes. It is computed on the chapters as
    loaded, before ``limit_chapters``/``max_paragraphs`` trimming, so the
    opening matter is always inspected.
    """
    try:
        notes = edition_notes(epub_path, chapters)
    except Exception as exc:  # noqa: BLE001 — never let the check stop a run
        progress(f"Edition check skipped: {exc}")
        return None
    if notes.needs_review:
        progress(
            f"Edition check: this edition may include text first published in "
            f"{notes.cutoff_year} or later, such as a preface or notes; those "
            "parts can still be under copyright even if the original book isn't."
        )
        details = [
            (f"{year} in {label}: {snippet}")
            for label, year, snippet in notes.recent_years
        ] + [
            (f"Copyright notice in {label}: {snippet}")
            for label, snippet in notes.copyright_notices
        ]
        for detail in details[:5]:
            progress(f"  - {detail}")
    return edition_payload(notes)


def _build_cover_text(
    epub_path: Path,
    components: PipelineComponents,
    progress: Callable[[str], None],
) -> CoverText:
    """Translated title/author for a replacement cover.

    Strict by design: a translation that does not return exactly one string
    per input is treated as a failure (never padded or truncated), so the
    title and author can never swap. On any failure the Kannada fields are
    ``None`` and the writer falls back to an English cover.
    """
    raw_title, author_en = read_epub_metadata(epub_path)
    title_en = raw_title or ""
    title_en = _clean_title_text(title_en, "") or title_en
    author_en = author_en or None

    title_kn: str | None = None
    author_kn: str | None = None
    if title_en:
        to_translate = [title_en] + ([author_en] if author_en else [])
        try:
            translated = components.translation_engine.translate_paragraphs(
                to_translate, "eng_Latn", "kan_Knda"
            )
            if len(translated) != len(to_translate):
                raise ValueError(
                    f"returned {len(translated)} strings for {len(to_translate)} inputs"
                )
            title_kn = translated[0].strip() or None
            if author_en:
                author_kn = translated[1].strip() or None
        except Exception as exc:  # noqa: BLE001 — never fail the book
            progress(f"Cover title not translated: {exc}")
            title_kn = None
            author_kn = None
    return CoverText(
        title_kn=title_kn,
        author_kn=author_kn,
        title_en=title_en,
        author_en=author_en,
    )


def run_book(
    cfg: BookConfig,
    *,
    resolve_path: Callable[[str], Path],
    options: RunOptions = RunOptions(),  # noqa: B008 — frozen value object, a safe shared default
    components: PipelineComponents | None = None,
    progress: Callable[[str], None] = print,
    cancel: threading.Event | None = None,
    on_event: Callable[[dict], None] | None = None,
) -> RunResult:
    """Run the full book pipeline; see the module docstring for the contract."""
    emit = on_event or (lambda _event: None)
    epub_path = resolve_path(cfg.epub_path)
    output_dir = resolve_path(cfg.output_dir)
    batch_size = options.batch_size or cfg.batch_size
    limit_chapters = options.limit_chapters or cfg.limit_chapters
    max_paragraphs = (
        options.max_paragraphs
        if options.max_paragraphs is not None
        else cfg.max_paragraphs_per_chapter
    )

    if components is None:
        components = build_components(cfg, resolve_path, need_tts=options.build_audiobook)

    chapters_dir = output_dir / "chapters"
    checkpoints_dir = output_dir / "checkpoints"
    chapters_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    chapters = load_epub_chapters(epub_path, exclude_ids=cfg.exclude_ids)
    source_problems = _check_source(epub_path, progress)
    edition = _check_edition(epub_path, chapters, progress)
    if limit_chapters:
        wanted = set(limit_chapters)
        chapters = [c for c in chapters if c.id in wanted]
    total_paragraphs = {c.id: len(c.paragraphs) for c in chapters}
    is_preview = max_paragraphs is not None and any(
        count > max_paragraphs for count in total_paragraphs.values()
    )
    if max_paragraphs is not None:
        for chapter in chapters:
            chapter.paragraphs = chapter.paragraphs[:max_paragraphs]
    emit({
        "type": "start",
        "chapters": [
            {"id": c.id, "title": c.title or c.id, "paragraphs": len(c.paragraphs)}
            for c in chapters
        ],
    })
    progress(f"Chapters to process: {[(c.id, len(c.paragraphs)) for c in chapters]}")

    t = cfg.translation
    if t.provider == "indictrans2_local":
        progress(f"Translation: local IndicTrans2 ({cfg.ct2_model_dir})")
    else:
        progress(f"Translation: cloud {t.provider} ({t.model} @ {t.base_url})")

    translator = BookTranslator(
        translation_engine=components.translation_engine,
        glossary_store=components.glossary_store,
        consistency_editor=components.editor,
        batch_size=batch_size,
        register=cfg.tone_register,
    )

    manifest: dict = {
        "epub": str(epub_path),
        "preview": is_preview,
        "exclude_ids": list(cfg.exclude_ids),
        "paragraphs_translated": 0,
        "paragraphs_total": 0,
        "chapters": [],
        "skipped": [],
        "source_problems": source_problems,
        "cover_replaced": None,
    }
    if edition is not None:
        manifest["edition"] = edition
    all_batches: list[TranslatedBatch] = []
    all_results: list[QAResult] = []
    cancelled = False

    for chapter_index, chapter in enumerate(chapters, start=1):
        if cancel is not None and cancel.is_set():
            cancelled = True
            progress(f"[{chapter.id}] cancelled before processing")
            break

        batches = _matching_checkpoints(checkpoints_dir, chapter)
        resumed = batches is not None
        emit({
            "type": "chapter",
            "id": chapter.id,
            "index": chapter_index,
            "total": len(chapters),
            "resumed": resumed,
        })
        if batches is not None:
            progress(f"[{chapter.id}] resume: {len(batches)} batches from checkpoints")
            manifest["skipped"].append(chapter.id)
        else:
            progress(f"[{chapter.id}] translating {len(chapter.paragraphs)} paragraphs...")
            batches = translator.translate_chapters([chapter])
            for batch in batches:
                _write_checkpoint(checkpoints_dir, batch)

        if components.qa is not None:
            emit({"type": "stage", "stage": "qa"})
            all_results.extend(
                _run_chapter_qa(cfg, components, chapter, batches, output_dir, progress)
            )

        chapter_out = {
            "chapter_id": chapter.id,
            "chapter_title": chapter.title,
            "batches": [asdict(b) for b in batches],
        }
        (chapters_dir / f"{chapter.id}.json").write_text(
            json.dumps(chapter_out, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        manifest["chapters"].append({
            "id": chapter.id,
            "title": chapter.title,
            "paragraphs": len(chapter.paragraphs),
            "total_paragraphs": total_paragraphs[chapter.id],
        })
        manifest["paragraphs_translated"] = sum(c["paragraphs"] for c in manifest["chapters"])
        manifest["paragraphs_total"] = sum(
            c["total_paragraphs"] for c in manifest["chapters"]
        )
        all_batches.extend(batches)

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    processed = [c["id"] for c in manifest["chapters"]]

    if cancelled:
        # Manifest is already on disk for the completed chapters; skip
        # QA report / EPUB / audiobook as the task specifies.
        emit({"type": "done"})
        return RunResult(
            output_dir=output_dir,
            chapters=processed,
            epub_path=None,
            qa_report_path=None,
            audiobook_path=None,
            cancelled=True,
        )

    qa_report_path: Path | None = None
    if components.qa is not None:
        qa_report_path = output_dir / "qa_report.json"
        write_qa_report(all_results, qa_report_path)
        statuses = [r.status for r in all_results]
        progress(
            f"QA: {statuses.count('PASS')} pass, {statuses.count('RETRY')} retry, "
            f"{statuses.count('FLAGGED_FOR_REVIEW')} flagged of {len(statuses)} -> {qa_report_path}"
        )

    epub_out_path: Path | None = None
    if options.write_epub:
        emit({"type": "stage", "stage": "epub"})
        # Re-load the source with no truncation: positions beyond
        # max_paragraphs were never translated, so they stay English.
        full_chapters = load_epub_chapters(epub_path, exclude_ids=cfg.exclude_ids)
        translations = translations_from_batches(
            full_chapters, [asdict(b) for b in all_batches]
        )
        flagged: dict[str, set[int]] = {}
        for result in all_results:
            if result.status == FLAGGED_FOR_REVIEW:
                flagged.setdefault(result.chapter, set()).add(result.paragraph_index)
        epub_name, _ = output_file_names(epub_path.stem, is_preview)
        epub_out_path = output_dir / epub_name

        cover_text: CoverText | None = None
        source_cover: str | None = None
        if cfg.strip_gutenberg:
            try:
                source_cover = find_gutenberg_cover(epub_path)
            except Exception as exc:  # noqa: BLE001 — never fail the book
                progress(f"Cover check skipped: {exc}")
                source_cover = None
            if source_cover is not None:
                try:
                    cover_text = _build_cover_text(epub_path, components, progress)
                except Exception as exc:  # noqa: BLE001 — never fail the book
                    progress(f"Cover title not translated: {exc}")
                    cover_text = None

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            write_translated_epub(
                epub_path,
                translations,
                epub_out_path,
                flagged=flagged,
                strip_gutenberg=cfg.strip_gutenberg,
                cover_text=cover_text,
            )

        if cfg.strip_gutenberg:
            remaining = find_gutenberg_mentions(epub_out_path)
            if remaining:
                progress(
                    f"Warning: {len(remaining)} Project Gutenberg mention(s) remain; "
                    f"first found in {remaining[0][0]}"
                )
            if source_cover is not None:
                cover_replaced: bool | None = None
                cover_error: Exception | None = None
                try:
                    still_generated = find_gutenberg_cover(epub_out_path)
                except Exception as exc:  # noqa: BLE001 — never fail the book
                    still_generated = source_cover
                    cover_error = exc
                if still_generated is None:
                    progress(
                        "Replaced Project Gutenberg's generated cover with a "
                        "Kannada cover"
                    )
                    cover_replaced = True
                else:
                    reason = str(cover_error) if cover_error is not None else next(
                        (
                            str(item.message)
                            for item in caught
                            if "cover" in str(item.message).lower()
                        ),
                        "cover rendering failed",
                    )
                    progress(
                        "Warning: the cover is still Project Gutenberg's "
                        f"generated cover ({reason})"
                    )
                    cover_replaced = False
                manifest["cover_replaced"] = cover_replaced
                (output_dir / "manifest.json").write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        progress(f"Wrote translated EPUB: {epub_out_path}")

    audiobook_path: Path | None = None
    if options.build_audiobook and components.tts is not None:
        emit({"type": "stage", "stage": "audiobook"})
        _, audio_name = output_file_names(epub_path.stem, is_preview)
        audiobook_path = output_dir / audio_name
        build_audiobook(
            batches=all_batches,
            voice=cfg.tts.voice,
            tts_engine=components.tts,
            output_path=audiobook_path,
            progress=progress,
            on_narrating=lambda done, total: emit(
                {"type": "narrating", "done": done, "total": total}
            ),
        )
        progress(f"Wrote audiobook: {audiobook_path}")

    progress(f"Done. Outputs in {output_dir}")
    emit({"type": "done"})
    return RunResult(
        output_dir=output_dir,
        chapters=processed,
        epub_path=epub_out_path,
        qa_report_path=qa_report_path,
        audiobook_path=audiobook_path,
        cancelled=False,
    )
