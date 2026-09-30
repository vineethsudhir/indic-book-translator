"""Local FastAPI server for the desktop-ready Kannada book translator."""

from __future__ import annotations

import argparse
import hmac
import json
import os
import re
import secrets
import socket
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Callable
from urllib.parse import quote

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from ..edition_check import edition_notes, edition_payload
from ..epub_check import check_source_epub
from ..epub_io import load_epub_chapters, read_epub_metadata
from ..importer import build_epub, import_html, import_text
from ..inline_markup import strip_markers
from ..pipeline import PipelineComponents, RunOptions, RunResult, output_file_names
from .book_queue import BookQueue, ItemNotFound, ItemRunning, StateError
from .paths import books_dir, data_dir, outputs_dir
from .runner import BookRun
from .settings import (
    KNOWN_KEYS,
    AppSettings,
    epubcheck_status,
    load_secrets_into_env,
    load_settings,
    local_mode_status,
    save_secret,
    save_settings,
    secret_status,
)


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


# --- import (text/HTML -> EPUB) -------------------------------------------

# Suffix -> result kind. Everything else is refused with a friendly message.
_IMPORT_EXTENSIONS = {".txt": "text", ".html": "html", ".htm": "html"}
_IMPORT_MAX_BYTES = 20 * 1024 * 1024
_IMPORT_TTL_SECONDS = 24 * 60 * 60
_IMPORT_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_IMPORT_ID_MESSAGE = "That upload has expired. Choose the file again."


def _imports_dir() -> Path:
    """Return (and create) the app-data folder holding pending imports."""
    path = data_dir() / "imports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _clean_stale_imports() -> None:
    """Delete imports older than the 24-hour preview window."""
    imports = data_dir() / "imports"
    if not imports.is_dir():
        return
    cutoff = time.time() - _IMPORT_TTL_SECONDS
    for entry in imports.iterdir():
        try:
            if entry.is_file() and not entry.is_symlink() and entry.stat().st_mtime < cutoff:
                entry.unlink(missing_ok=True)
        except OSError:  # one bad entry must not stop the sweep
            pass


def _resolve_import(import_id: str) -> Path:
    """Resolve an ``import_id`` against the real imports listing, or 404."""
    if not isinstance(import_id, str) or not _IMPORT_ID_RE.fullmatch(import_id):
        raise HTTPException(404, _IMPORT_ID_MESSAGE)
    imports = data_dir() / "imports"
    if imports.is_dir():
        for entry in imports.iterdir():
            if (
                entry.is_file()
                and not entry.is_symlink()
                and entry.stem == import_id
                and entry.suffix.lower() in _IMPORT_EXTENSIONS
            ):
                return entry
    raise HTTPException(404, _IMPORT_ID_MESSAGE)


def _used_headings(result) -> list[str]:
    """The headings the importer actually split on, minus "Front matter"."""
    return [chapter.title for chapter in result.chapters if chapter.title != "Front matter"]


def _safe_import_filename(title: str) -> str:
    """A filesystem-safe stem built from the title (letters, digits, space, - _)."""
    name = re.sub(r"[^\w \-]", "", title)
    name = "_".join(name.split())
    return name[:80] or "imported_book"


def _requirements(settings: AppSettings, audiobook: bool) -> tuple[list[str], list[str]]:
    keys: list[str] = []
    local: list[str] = []
    if settings.translation.provider == "openai_compatible":
        keys.append(settings.translation.api_key_env or "OPENAI_API_KEY")
    elif settings.translation.provider == "indictrans2_local":
        local += ["ctranslate2", "sentencepiece", "IndicTransToolkit"]
    if settings.editor.provider == "anthropic":
        keys.append(settings.editor.api_key_env or "ANTHROPIC_API_KEY")
    elif settings.editor.provider == "openai_compatible":
        keys.append(settings.editor.api_key_env or "OPENAI_API_KEY")
    if audiobook:
        if settings.tts.provider == "sarvam":
            keys.append(settings.tts.api_key_env or "SARVAM_API_KEY")
        elif settings.tts.provider == "openai_compatible":
            keys.append(settings.tts.api_key_env or "OPENAI_API_KEY")
        else:
            local += ["torch", "transformers", "parler_tts"]
    if settings.qa.enabled:
        if settings.qa.back_translation == "indictrans2_local":
            local += ["ctranslate2", "sentencepiece", "IndicTransToolkit"]
        if settings.qa.embedding == "openai_compatible":
            keys.append(settings.qa.embedding_api_key_env or "OPENAI_API_KEY")
        else:
            local += ["torch", "transformers"]
    key_names = _dedupe(keys)
    module_names = _dedupe(local)
    statuses = local_mode_status()
    return (
        [name for name in key_names if not os.environ.get(name)],
        [name for name in module_names if not statuses.get(name, False)],
    )


def _metadata(path: Path) -> tuple[str, str]:
    title, author = read_epub_metadata(path)
    return title or path.stem, author or "Unknown author"


def _book_response(dest: Path) -> dict:
    """The post-upload analysis shared by ``/api/books`` and the importer.

    Reads the EPUB's metadata and chapters (raising if it cannot be read, so
    the caller can report or delete the file) and adds the source-quality and
    edition notes that a failed check must never fail the response over.
    """
    title, author = _metadata(dest)
    settings = load_settings()
    chapters = load_epub_chapters(dest, exclude_ids=settings.exclude_ids)
    try:
        source_problems = check_source_epub(dest)
        problem_messages = [
            f"{Path(problem.path).name or dest.name}: {problem.message}"
            for problem in source_problems
        ][:20]
        problem_count = len(source_problems)
    except Exception:  # noqa: BLE001 — a check failure must not fail the upload
        problem_messages = None
        problem_count = 0
    try:
        edition = edition_payload(edition_notes(dest, chapters))
    except Exception:  # noqa: BLE001 — a check failure must not fail the upload
        edition = None
    response = {
        "book_id": dest.stem,
        "filename": dest.name,
        "title": title,
        "author": author,
        "source_problem_count": problem_count,
        "chapters": [
            {"id": c.id, "title": c.title or c.id, "paragraphs": len(c.paragraphs)}
            for c in chapters
        ],
    }
    if problem_messages is not None:
        response["source_problems"] = problem_messages
    if edition is not None:
        response["edition"] = edition
    return response


def _safe_title(path: Path) -> str:
    """A book's title for queue messages, falling back to its file name."""
    try:
        return _metadata(path)[0]
    except Exception:  # noqa: BLE001 — an unreadable EPUB is failed at run time
        return path.name


class _RequirementsError(HTTPException):
    """An API key or local library is missing; every queue item would fail."""


def _book_source(book_id: str, *, status_code: int = 404) -> Path:
    """Resolve a request's ``book_id`` to a real file directly in books_dir."""
    filename = Path(book_id).name + ".epub"
    source = books_dir() / filename
    if (
        not book_id
        or book_id != Path(book_id).name
        or "/" in book_id
        or "\\" in book_id
        or source.parent != books_dir()
        or not source.is_file()
    ):
        raise HTTPException(status_code, "Uploaded book not found. Choose the EPUB again.")
    return source


def _validated_skip_ids(
    source: Path, skip_chapters: object, settings: AppSettings
) -> list[str]:
    """Validate a run's skipped section ids exactly like ``/api/run``."""
    if not isinstance(skip_chapters, list) or not all(
        isinstance(item, str) for item in skip_chapters
    ):
        raise HTTPException(400, "Sections to skip must be a list of section ids.")
    skip_ids = _dedupe(skip_chapters)
    if skip_ids:
        try:
            available = load_epub_chapters(source, exclude_ids=settings.exclude_ids)
        except Exception as exc:  # noqa: BLE001 — the EPUB was validated at upload
            raise HTTPException(400, f"This EPUB could not be read: {exc}") from exc
        available_ids = {chapter.id for chapter in available}
        for skip_id in skip_ids:
            if skip_id not in available_ids:
                raise HTTPException(400, f"Unknown section: {skip_id[:100]}.")
        if not available_ids - set(skip_ids):
            raise HTTPException(400, "Choose at least one section.")
    return skip_ids


def _start_book(
    app: FastAPI,
    payload: object,
    *,
    on_finish: Callable[[RunResult | None, str | None], None] | None = None,
    queue_item_id: str | None = None,
) -> None:
    """Validate and start one book on the runner.

    Both ``/api/run`` and the queue go through here, so they share the book-id
    check, option checks, ``skip_chapters`` validation, requirement checks, the
    per-run settings copy and the run metadata. Raises ``HTTPException`` on any
    problem.
    """
    runner: BookRun = app.state.runner
    if not isinstance(payload, dict):
        raise HTTPException(400, "Expected book and run options as a JSON object.")
    book_id = str(payload.get("book_id", ""))
    source = _book_source(book_id)
    if runner.is_running():
        raise HTTPException(409, "A translation is already running.")
    settings = load_settings()
    qa_enabled = payload.get("qa", settings.qa.enabled)
    audiobook = payload.get("audiobook", False)
    if not isinstance(qa_enabled, bool) or not isinstance(audiobook, bool):
        raise HTTPException(400, "Quality check and audiobook options must be true or false.")
    skip_ids = _validated_skip_ids(source, payload.get("skip_chapters", []), settings)
    settings = settings.model_copy(update={
        "qa": settings.qa.model_copy(update={"enabled": qa_enabled}),
        "exclude_ids": _dedupe(settings.exclude_ids + skip_ids),
    })
    load_secrets_into_env()
    missing_keys, missing_modules = _requirements(settings, audiobook)
    if missing_keys or missing_modules:
        messages = []
        if missing_keys:
            messages.append("Add these API keys in Settings: " + ", ".join(missing_keys) + ".")
        if missing_modules:
            messages.append(
                "Install Local mode libraries in Settings: " + ", ".join(missing_modules) + "."
            )
        raise _RequirementsError(400, " ".join(messages))
    preview = payload.get("preview_paragraphs")
    try:
        if preview in (None, ""):
            preview = None
        elif isinstance(preview, bool) or (isinstance(preview, float) and not preview.is_integer()):
            raise ValueError
        else:
            preview = int(preview)
        if preview is not None and preview < 1:
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, "Preview paragraphs must be a positive whole number.") from exc
    options = RunOptions(
        max_paragraphs=preview,
        batch_size=settings.batch_size,
        build_audiobook=audiobook,
    )
    app.state.run_meta = {
        "book_id": book_id,
        "filename": source.name,
        "chapters": [],
        "started": time.time(),
        "skipped_chapters": skip_ids,
        "queue_item_id": queue_item_id,
    }
    components_factory = app.state.components_factory
    components = components_factory() if components_factory is not None else None
    try:
        runner.start(source, settings, options, components=components, on_finish=on_finish)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc


def _queue_advance(app: FastAPI) -> None:
    """Start the next pending item when the queue is active and idle."""
    queue: BookQueue = app.state.queue
    runner: BookRun = app.state.runner
    with app.state.queue_advance_lock:
        while queue.active and not runner.is_running():
            item = queue.next_pending()
            if item is None:
                queue.set_active(False)
                return
            queue.mark_running(item.id)
            payload = {
                "book_id": item.book_id,
                "qa": item.qa,
                "audiobook": item.audiobook,
                "skip_chapters": list(item.skip_chapters),
            }
            try:
                _start_book(
                    app,
                    payload,
                    on_finish=lambda result, error, item_id=item.id: _on_queue_item_finished(
                        app, item_id, result, error
                    ),
                    queue_item_id=item.id,
                )
            except _RequirementsError as exc:
                # Missing keys/libraries would fail every item: stop, don't spin.
                queue.mark_failed(item.id, str(exc.detail))
                queue.set_active(False)
                return
            except HTTPException as exc:
                queue.mark_failed(item.id, str(exc.detail))
                continue
            except Exception as exc:  # noqa: BLE001 — a queue item must not crash the app
                queue.mark_failed(item.id, f"{type(exc).__name__}: {exc}")
                continue
            return


def _on_queue_item_finished(
    app: FastAPI, item_id: str, result: RunResult | None, error: str | None
) -> None:
    """Record how a queue run ended, then hand off to the next book.

    The outcome comes from the finished run itself, not from the runner, which
    a new run may already have reset.
    """
    queue: BookQueue = app.state.queue
    if error:
        queue.mark_failed(item_id, error)
    elif result is not None and result.cancelled:
        queue.mark_cancelled(item_id)
        # Stop was pressed: do not start the next book until Start queue.
        queue.set_active(False)
    else:
        queue.mark_done(item_id)
    _queue_advance(app)


def _source_epub(folder: Path, manifest: dict) -> Path | None:
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


def _source_titles(folder: Path, manifest: dict) -> dict[str, str]:
    source = _source_epub(folder, manifest)
    if source is None:
        return {}
    try:
        return {
            chapter.id: chapter.title or chapter.id
            for chapter in load_epub_chapters(source, exclude_ids=[])
        }
    except Exception:
        return {}


def _book_outputs(folder: Path, manifest: dict) -> dict:
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
        source = _source_epub(folder, manifest)
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


def _book_metadata(folder: Path, manifest: dict) -> tuple[str, str]:
    source = _source_epub(folder, manifest)
    if source is not None:
        try:
            return _metadata(source)
        except Exception:
            pass
    return folder.name, "Unknown author"


def _safe_book_dir(book_id: str) -> Path:
    if not book_id or book_id in {".", ".."} or "/" in book_id or "\\" in book_id:
        raise HTTPException(404, "Book not found")
    root = outputs_dir()
    target = next(
        (path for path in root.iterdir() if path.name == book_id and path.is_dir() and not path.is_symlink()),
        None,
    )
    if target is None or not (target / "manifest.json").is_file():
        raise HTTPException(404, "Book not found")
    return target


def _qa_summary(folder: Path) -> dict | None:
    report = folder / "qa_report.json"
    if not report.is_file():
        return None
    try:
        summary = json.loads(report.read_text(encoding="utf-8")).get("summary", {})
        return {
            "pass": int(summary.get("pass", 0)),
            "retry": int(summary.get("retry", 0)),
            "flagged": int(summary.get("flagged", 0)),
            "total": int(summary.get("total", 0)),
        }
    except (OSError, ValueError, TypeError):
        return None


def create_app(
    *,
    components_factory: Callable[[], PipelineComponents] | None = None,
    port: int = 7860,
) -> FastAPI:
    """Create an isolated app instance (also used by API tests)."""
    app = FastAPI(title="Kannada Book Translator", docs_url=None, redoc_url=None)
    token = secrets.token_urlsafe(32)
    runner = BookRun()
    queue = BookQueue()
    app.state.app_token = token
    app.state.runner = runner
    app.state.queue = queue
    app.state.queue_advance_lock = threading.Lock()
    app.state.run_meta = {
        "book_id": None,
        "filename": None,
        "chapters": [],
        "started": None,
        "skipped_chapters": [],
        "queue_item_id": None,
    }
    app.state.components_factory = components_factory

    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    repo_fonts = Path(__file__).resolve().parents[3] / "assets" / "fonts"
    installed_fonts = Path(sys.prefix) / "assets" / "fonts"

    @app.middleware("http")
    async def local_host_only(request: Request, call_next):
        host = request.headers.get("host", "")
        allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if host not in allowed:
            return JSONResponse({"detail": "Requests must use the local app address."}, status_code=403)
        return await call_next(request)

    async def require_token(
        request: Request,
        x_app_token: str | None = Header(default=None, alias="X-App-Token"),
    ) -> None:
        supplied = x_app_token or request.query_params.get("token", "")
        if not hmac.compare_digest(str(supplied), token):
            raise HTTPException(status_code=401, detail="App token required")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        html = (static_dir / "index.html").read_text(encoding="utf-8")
        return HTMLResponse(html.replace("__APP_TOKEN__", token))

    @app.get("/favicon.ico")
    async def favicon():
        icon = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
            '<rect width="64" height="64" rx="16" fill="#28745c"/>'
            '<text x="50%" y="53%" dominant-baseline="middle" text-anchor="middle" '
            'font-family="sans-serif" font-size="38" fill="white">ಕ</text></svg>'
        )
        return Response(content=icon, media_type="image/svg+xml")

    @app.get("/fonts/{font_name}")
    async def font(font_name: str):
        if font_name not in {"NotoSansKannada-Regular.ttf", "NotoSansKannada-Bold.ttf"}:
            raise HTTPException(404, "Font not found")
        path = repo_fonts / font_name
        if not path.is_file():
            path = installed_fonts / font_name
        if not path.is_file():
            raise HTTPException(404, "Kannada font is unavailable")
        return FileResponse(path, media_type="font/ttf")

    @app.get("/api/state", dependencies=[Depends(require_token)])
    async def state():
        return {
            "settings": load_settings().model_dump(mode="json"),
            "key_status": secret_status(),
            "local_mode": local_mode_status(),
            "epubcheck": epubcheck_status(),
            "run": _run_status(app),
        }

    @app.get("/api/settings", dependencies=[Depends(require_token)])
    async def get_settings():
        return load_settings().model_dump(mode="json")

    @app.put("/api/settings", dependencies=[Depends(require_token)])
    async def put_settings(request: Request):
        try:
            settings = AppSettings.model_validate(await request.json())
        except ValidationError as exc:
            errors = [f"{'.'.join(map(str, item['loc']))}: {item['msg']}" for item in exc.errors()]
            raise HTTPException(400, detail="; ".join(errors)) from exc
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, detail=f"Invalid settings: {exc}") from exc
        save_settings(settings)
        return {"settings": settings.model_dump(mode="json"), "saved": True}

    @app.put("/api/keys", dependencies=[Depends(require_token)])
    async def put_keys(request: Request):
        try:
            values = await request.json()
        except ValueError as exc:
            raise HTTPException(400, "Expected a JSON object of API keys") from exc
        if not isinstance(values, dict):
            raise HTTPException(400, "Expected a JSON object of API keys")
        unknown = set(values) - set(KNOWN_KEYS)
        if unknown:
            raise HTTPException(400, f"Unknown key name: {', '.join(sorted(unknown))}")
        for name, value in values.items():
            if not isinstance(value, str):
                raise HTTPException(400, f"{name} must be text")
            try:
                save_secret(name, value.strip())
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
        return {"key_status": secret_status()}

    @app.post("/api/books", dependencies=[Depends(require_token)])
    async def upload_book(file: UploadFile = File(...)):  # noqa: B008 — FastAPI's required-file marker
        filename = Path((file.filename or "").replace("\\", "/")).name
        if not filename.lower().endswith(".epub"):
            raise HTTPException(400, "Choose an EPUB file (.epub).")
        stem = Path(filename).stem.strip() or "book"
        dest_dir = books_dir()
        dest = dest_dir / filename
        suffix = 2
        while dest.exists():
            dest = dest_dir / f"{stem}_{suffix}.epub"
            suffix += 1
        content = await file.read()
        dest.write_bytes(content)
        try:
            return _book_response(dest)
        except Exception as exc:  # malformed zip/container/metadata
            dest.unlink(missing_ok=True)
            raise HTTPException(400, f"This EPUB could not be read: {exc}") from exc

    @app.post("/api/import/preview", dependencies=[Depends(require_token)])
    async def import_preview(
        file: Annotated[UploadFile | None, File()] = None,
        ocr: Annotated[str, Form()] = "false",
        headings: Annotated[str, Form()] = "",
        import_id: Annotated[str, Form()] = "",
    ):
        _clean_stale_imports()
        import_id = import_id.strip()
        ocr_flag = ocr.strip().lower() == "true"
        headings = [line.strip() for line in headings.splitlines() if line.strip()]

        if import_id:
            source = _resolve_import(import_id)
            ext = source.suffix.lower()
            kind = _IMPORT_EXTENSIONS[ext]
            content = source.read_bytes()
        elif file is not None:
            filename = Path((file.filename or "").replace("\\", "/")).name
            ext = Path(filename).suffix.lower()
            kind = _IMPORT_EXTENSIONS.get(ext)
            if kind is None:
                raise HTTPException(400, "Choose a .txt or .html file.")
            # Read at most one byte past the limit, so an oversized upload
            # is refused without holding all of it in memory.
            content = await file.read(_IMPORT_MAX_BYTES + 1)
            if len(content) > _IMPORT_MAX_BYTES:
                raise HTTPException(413, "This file is larger than 20 MB.")
            import_id = uuid.uuid4().hex
            dest = _imports_dir() / f"{import_id}{ext}"
            dest.write_bytes(content)
        else:
            raise HTTPException(400, "Choose a .txt or .html file.")

        text = content.decode("utf-8", errors="replace")
        if kind == "html":
            result = import_html(text, headings=headings or None)
        else:
            result = import_text(text, ocr=ocr_flag, headings=headings or None)

        return {
            "import_id": import_id,
            "kind": kind,
            "chapters": [
                {
                    "title": chapter.title,
                    "paragraphs": len(chapter.paragraphs),
                    "first": (chapter.paragraphs[0] if chapter.paragraphs else "")[:200],
                }
                for chapter in result.chapters
            ],
            "warnings": result.warnings,
            "detected_headings": result.detected_headings,
            "headings": _used_headings(result),
            "total_paragraphs": sum(
                len(chapter.paragraphs) for chapter in result.chapters
            ),
        }

    @app.post("/api/import/create", dependencies=[Depends(require_token)])
    async def import_create(request: Request):
        try:
            payload = await request.json()
        except ValueError as exc:
            raise HTTPException(400, "Expected import options as a JSON object.") from exc
        if not isinstance(payload, dict):
            raise HTTPException(400, "Expected import options as a JSON object.")

        source = _resolve_import(str(payload.get("import_id") or ""))

        title = payload.get("title")
        if not isinstance(title, str) or not title.strip():
            raise HTTPException(400, "Enter a title for the book.")
        title = title.strip()
        if len(title) > 300:
            raise HTTPException(400, "The title is too long (300 characters maximum).")

        def optional_text(value: object, limit: int, label: str) -> str | None:
            if value is None:
                return None
            if not isinstance(value, str):
                raise HTTPException(400, f"{label} must be text.")
            value = value.strip()
            if not value:
                return None
            if len(value) > limit:
                raise HTTPException(
                    400, f"{label} is too long ({limit} characters maximum)."
                )
            return value

        author = optional_text(payload.get("author"), 300, "Author")
        date = optional_text(payload.get("date"), 20, "Year")
        source_url = optional_text(payload.get("source"), 500, "Source")

        headings: list[str] | None = None
        raw_headings = payload.get("headings")
        if raw_headings is not None:
            if not isinstance(raw_headings, list) or not all(
                isinstance(item, str) for item in raw_headings
            ):
                raise HTTPException(400, "Chapter headings must be a list of text.")
            headings = [item.strip() for item in raw_headings if item.strip()] or None

        ocr = payload.get("ocr", False) is True
        ext = source.suffix.lower()
        text = source.read_bytes().decode("utf-8", errors="replace")
        if ext == ".html" or ext == ".htm":
            result = import_html(text, headings=headings)
        else:
            result = import_text(text, ocr=ocr, headings=headings)

        if not any(chapter.paragraphs for chapter in result.chapters):
            raise HTTPException(400, "No text was found in this file.")

        base = _safe_import_filename(title)
        dest_dir = books_dir()
        dest = dest_dir / f"{base}.epub"
        suffix = 2
        while dest.exists():
            dest = dest_dir / f"{base}_{suffix}.epub"
            suffix += 1
        build_epub(
            dest,
            title=title,
            author=author,
            date=date,
            source=source_url,
            chapters=result.chapters,
        )
        response = _book_response(dest)
        source.unlink(missing_ok=True)
        return response

    @app.post("/api/run", dependencies=[Depends(require_token)])
    async def start_run(request: Request):
        payload = await request.json()
        _start_book(app, payload)
        return _run_status(app)

    @app.get("/api/books", dependencies=[Depends(require_token)])
    async def list_books():
        entries = []
        for path in books_dir().iterdir():
            if not path.is_file() or path.is_symlink() or path.suffix.lower() != ".epub":
                continue
            entry = {"book_id": path.stem, "filename": path.name}
            try:
                title, author = _metadata(path)
                entry["title"] = title
                entry["author"] = author
            except Exception:  # noqa: BLE001 — list it, flag it, keep the page working
                entry["title"] = path.name
                entry["author"] = "Unknown author"
                entry["unreadable"] = True
            entries.append(entry)
        entries.sort(key=lambda item: (item.get("title") or "").lower())
        return entries

    @app.get("/api/queue", dependencies=[Depends(require_token)])
    async def get_queue():
        return queue.snapshot()

    @app.post("/api/queue", dependencies=[Depends(require_token)])
    async def add_queue(request: Request):
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(400, "Expected a list of books as a JSON object.")
        items = payload.get("items")
        if not isinstance(items, list) or not items:
            raise HTTPException(400, "Choose at least one book.")
        settings = load_settings()
        qa_enabled = payload.get("qa", settings.qa.enabled)
        audiobook = payload.get("audiobook", False)
        if not isinstance(qa_enabled, bool) or not isinstance(audiobook, bool):
            raise HTTPException(400, "Quality check and audiobook options must be true or false.")
        resolved = []
        for entry in items:
            if not isinstance(entry, dict):
                raise HTTPException(400, "Each queued book needs a book id.")
            source = _book_source(str(entry.get("book_id", "")), status_code=400)
            skip_ids = _validated_skip_ids(
                source, entry.get("skip_chapters", []), settings
            )
            resolved.append((source, skip_ids))
        queued = queue.active_book_ids()
        seen: set[str] = set()
        for source, _skip_ids in resolved:
            book_id = source.stem
            if book_id in queued or book_id in seen:
                raise HTTPException(400, f"{_safe_title(source)} is already in the queue.")
            seen.add(book_id)
        entries = [
            {
                "book_id": source.stem,
                "filename": source.name,
                "title": _safe_title(source),
                "skip_chapters": skip_ids,
                "qa": qa_enabled,
                "audiobook": audiobook,
            }
            for source, skip_ids in resolved
        ]
        queue.add(entries)
        return queue.snapshot()

    @app.delete("/api/queue/{item_id}", dependencies=[Depends(require_token)])
    async def remove_queue_item(item_id: str):
        try:
            queue.remove(item_id)
        except ItemNotFound as exc:
            raise HTTPException(404, "Queue item not found.") from exc
        except ItemRunning as exc:
            raise HTTPException(409, str(exc)) from exc
        return queue.snapshot()

    @app.post("/api/queue/{item_id}/move", dependencies=[Depends(require_token)])
    async def move_queue_item(item_id: str, request: Request):
        payload = await request.json()
        direction = payload.get("direction") if isinstance(payload, dict) else None
        try:
            queue.move(item_id, str(direction))
        except ItemNotFound as exc:
            raise HTTPException(404, "Queue item not found.") from exc
        except StateError as exc:
            raise HTTPException(400, str(exc)) from exc
        return queue.snapshot()

    @app.post("/api/queue/start", dependencies=[Depends(require_token)])
    async def start_queue():
        if runner.is_running() and not app.state.run_meta.get("queue_item_id"):
            raise HTTPException(409, "A translation is already running.")
        if not queue.has_pending():
            raise HTTPException(400, "The queue has no books waiting.")
        queue.set_active(True)
        _queue_advance(app)
        return queue.snapshot()

    @app.post("/api/queue/pause", dependencies=[Depends(require_token)])
    async def pause_queue():
        queue.set_active(False)
        return queue.snapshot()

    @app.post("/api/queue/clear", dependencies=[Depends(require_token)])
    async def clear_queue():
        queue.clear_finished()
        return queue.snapshot()

    @app.post("/api/queue/{item_id}/retry", dependencies=[Depends(require_token)])
    async def retry_queue_item(item_id: str):
        try:
            queue.retry(item_id)
        except ItemNotFound as exc:
            raise HTTPException(404, "Queue item not found.") from exc
        except StateError as exc:
            raise HTTPException(400, str(exc)) from exc
        return queue.snapshot()

    @app.post("/api/run/stop", dependencies=[Depends(require_token)])
    async def stop_run():
        if runner.is_running():
            runner.cancel()
        return _run_status(app)

    @app.get("/api/run", dependencies=[Depends(require_token)])
    async def get_run():
        return _run_status(app)

    @app.get("/api/library", dependencies=[Depends(require_token)])
    async def library():
        entries = []
        for folder in outputs_dir().iterdir():
            manifest_path = folder / "manifest.json"
            if not folder.is_dir() or not manifest_path.is_file():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            title, author = _book_metadata(folder, manifest)
            outputs = _book_outputs(folder, manifest)
            entries.append({
                "book_id": folder.name,
                "title": title,
                "author": author,
                "updated_at": datetime.fromtimestamp(
                    manifest_path.stat().st_mtime, tz=timezone.utc
                ).isoformat(),
                "_mtime": manifest_path.stat().st_mtime,
                "chapters": len(manifest.get("chapters", [])),
                "has_epub": (folder / outputs["epub"]).is_file(),
                "has_audiobook": (folder / outputs["audiobook"]).is_file(),
                "preview": outputs["preview"],
                "paragraphs_translated": outputs["paragraphs_translated"],
                "paragraphs_total": outputs["paragraphs_total"],
                "qa_summary": _qa_summary(folder),
                "epubcheck": manifest.get("epubcheck"),
            })
        entries.sort(key=lambda item: item["_mtime"], reverse=True)
        for item in entries:
            item.pop("_mtime", None)
        return entries

    @app.get("/api/library/{book_id}/chapters", dependencies=[Depends(require_token)])
    async def library_chapters(book_id: str):
        folder = _safe_book_dir(book_id)
        data = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        titles = _source_titles(folder, data)
        return [
            {
                **chapter,
                "title": titles.get(chapter.get("id"), chapter.get("title") or chapter.get("id")),
            }
            for chapter in data.get("chapters", [])
        ]

    @app.get("/api/library/{book_id}/chapters/{chapter_id}", dependencies=[Depends(require_token)])
    async def chapter_detail(book_id: str, chapter_id: str):
        folder = _safe_book_dir(book_id)
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        chapter_info = next((c for c in manifest.get("chapters", []) if c.get("id") == chapter_id), None)
        if chapter_info is None or "/" in chapter_id or "\\" in chapter_id:
            raise HTTPException(404, "Chapter not found")
        chapter_path = folder / "chapters" / f"{chapter_id}.json"
        if not chapter_path.is_file():
            raise HTTPException(404, "Chapter not found")
        chapter_data = json.loads(chapter_path.read_text(encoding="utf-8"))
        qa_path = folder / "qa" / f"{chapter_id}.json"
        qa_results = json.loads(qa_path.read_text(encoding="utf-8")).get("results", []) if qa_path.is_file() else []
        paragraphs = []
        position = 0
        for batch in chapter_data.get("batches", []):
            english = batch.get("source_english", [])
            kannada = batch.get("edited_kannada", [])
            emotions = batch.get("edited_emotions", [])
            for offset, (en, kn) in enumerate(zip(english, kannada, strict=True)):
                qa = qa_results[position] if position < len(qa_results) else {}
                paragraphs.append({
                    "index": qa.get("paragraph_index", batch.get("paragraph_start", 0) + offset),
                    "en": en,
                    "kn": strip_markers(kn),
                    "emotion": emotions[offset] if offset < len(emotions) else None,
                    "qa_status": qa.get("status"),
                    "qa_score": qa.get("similarity_score"),
                    "back_translation": qa.get("back_translated_en"),
                })
                position += 1
        title = _source_titles(folder, manifest).get(
            chapter_id, chapter_info.get("title") or chapter_id
        )
        return {"id": chapter_id, "title": title, "paragraphs": paragraphs}

    @app.get("/api/library/{book_id}/files/{name}")
    async def download_file(book_id: str, name: str, token: str = ""):
        if not hmac.compare_digest(token, app.state.app_token):
            raise HTTPException(401, "App token required")
        folder = _safe_book_dir(book_id)
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        stem = Path(manifest.get("epub", book_id)).stem
        target_language = str(manifest.get("target_language") or "kn")
        allowed = {"qa_report.json"}
        for preview in (False, True):
            allowed.update(output_file_names(stem, preview, target_language))
        # Older runs always wrote .kn.* names.
        for preview in (False, True):
            allowed.update(output_file_names(stem, preview, "kn"))
        if name not in allowed or name not in {p.name for p in folder.iterdir() if p.is_file()}:
            raise HTTPException(404, "File not found")
        return FileResponse(folder / name, filename=name)

    @app.post("/api/library/{book_id}/open", dependencies=[Depends(require_token)])
    async def open_output(book_id: str, request: Request):
        folder = _safe_book_dir(book_id)
        payload = await request.json()
        what = payload.get("what")
        if what == "folder":
            target = folder
        elif what == "epub":
            manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
            target = folder / _book_outputs(folder, manifest)["epub"]
            if not target.is_file():
                raise HTTPException(404, "Translated EPUB not found")
        else:
            raise HTTPException(400, 'Choose "folder" or "epub".')
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", str(target)])
            elif sys.platform == "win32":
                os.startfile(str(target))  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", str(target)])
        except OSError as exc:
            raise HTTPException(500, f"Could not open this item: {exc}") from exc
        return {"opened": True}

    return app


def _run_status(app: FastAPI) -> dict:
    runner: BookRun = app.state.runner
    meta = app.state.run_meta
    events = runner.events()
    chapters = meta.get("chapters", [])
    stage: str | None = None
    chapter_index = 0
    chapter_total = len(chapters)
    chapter_title = None
    narrating = None
    for event in events:
        if event["type"] == "start":
            chapters = event.get("chapters", [])
            chapter_total = len(chapters)
        elif event["type"] == "chapter":
            chapter_index = event.get("index", 0)
            chapter_total = event.get("total", chapter_total)
            chapter_title = next((c.get("title") for c in chapters if c.get("id") == event.get("id")), event.get("id"))
            stage = "translation"
        elif event["type"] == "stage":
            stage = event.get("stage")
        elif event["type"] == "narrating":
            narrating = {"done": event.get("done", 0), "total": event.get("total", 0)}
    if runner.is_running():
        state_name = "running"
    elif runner.error:
        state_name = "failed"
    elif runner.result is None:
        state_name = "idle"
    elif runner.result.cancelled:
        state_name = "cancelled"
    else:
        state_name = "finished"
    result = None
    if runner.result is not None and not runner.result.cancelled:
        run_result = runner.result
        folder = Path(run_result.output_dir)
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8")) if (folder / "manifest.json").is_file() else {}
        summary = _qa_summary(folder)
        book_id = meta.get("book_id")
        token = app.state.app_token
        epub, audio = run_result.epub_path, run_result.audiobook_path
        result = {
            "epub": _file_url(book_id, epub.name, token) if epub else None,
            "qa_report": _file_url(book_id, "qa_report.json", token) if run_result.qa_report_path else None,
            "audiobook": _file_url(book_id, audio.name, token) if audio else None,
            "qa_summary": summary,
            "preview": bool(manifest.get("preview")),
            "paragraphs_translated": manifest.get("paragraphs_translated"),
            "paragraphs_total": manifest.get("paragraphs_total"),
        }
    return {
        "state": state_name,
        "book_id": meta.get("book_id"),
        "chapter_index": chapter_index,
        "chapter_total": chapter_total,
        "current_chapter_title": chapter_title,
        "stage": stage,
        "narrating": narrating,
        "skipped_chapters": meta.get("skipped_chapters", []),
        "queue_item_id": meta.get("queue_item_id"),
        "elapsed_seconds": max(0, int(time.time() - meta["started"])) if meta.get("started") else 0,
        "log_tail": runner.log_tail(200),
        "error": runner.error,
        "result": result,
    }


def _file_url(book_id: str | None, name: str, token: str) -> str:
    return f"/api/library/{quote(book_id or '', safe='')}/files/{quote(name, safe='')}?token={quote(token, safe='')}"


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Kannada Book Translator")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    port = args.port
    while True:
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
                break
            except OSError:
                port += 1
    app = create_app(port=port)
    if not args.no_browser:
        import threading
        import webbrowser
        threading.Timer(0.6, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
