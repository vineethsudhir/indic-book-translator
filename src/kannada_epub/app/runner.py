"""In-process book runner for the app.

Runs :func:`kannada_epub.pipeline.run_book` on a daemon thread (no subprocess,
no ``sys.executable``), streaming progress lines through a queue and keeping
structured events for the local app. ``components`` lets tests inject fakes;
otherwise the pipeline builds the configured engines.
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path

from ..pipeline import PipelineComponents, RunOptions, RunResult, run_book
from .settings import AppSettings, book_config_for


def _resolve_path(path_str: str) -> Path:
    """Resolve a config path relative to the current working directory."""
    path = Path(path_str)
    return path if path.is_absolute() else Path.cwd() / path


class BookRun:
    """One book translation run at a time, on a background thread."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._queue: queue.Queue[str] = queue.Queue()
        self._events: list[dict] = []
        self._log_history: list[str] = []
        self._result: RunResult | None = None
        self._error: str | None = None
        self._lock = threading.Lock()

    # -- control -----------------------------------------------------------
    def start(
        self,
        epub_path: str | Path,
        settings: AppSettings,
        options: RunOptions,
        components: PipelineComponents | None = None,
    ) -> None:
        """Start a run. Raises ``RuntimeError`` if one is already running."""
        with self._lock:
            if self.is_running():
                raise RuntimeError("A translation run is already in progress.")
            self._queue = queue.Queue()
            self._events = []
            self._log_history = []
            self._result = None
            self._error = None
            self._thread = threading.Thread(
                target=self._run,
                args=(Path(epub_path), settings, options, components),
                daemon=True,
                name="kannada-book-run",
            )
            self._thread.start()

    def cancel(self) -> None:
        """Request cancellation; the pipeline stops at the next chapter."""
        self._cancel.set()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- output ------------------------------------------------------------
    def drain(self) -> list[str]:
        """Return and clear the progress lines produced since the last drain."""
        lines: list[str] = []
        while True:
            try:
                lines.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return lines

    def events(self) -> list[dict]:
        """Return a thread-safe snapshot of structured pipeline events."""
        with self._lock:
            return [dict(event) for event in self._events]

    def log_tail(self, limit: int = 200) -> list[str]:
        """Return recent log lines without consuming the legacy drain queue."""
        with self._lock:
            return list(self._log_history[-limit:])

    def _log(self, line: str) -> None:
        with self._lock:
            self._log_history.append(line)
        self._queue.put(line)

    def _event(self, event: dict) -> None:
        with self._lock:
            self._events.append(dict(event))

    @property
    def result(self) -> RunResult | None:
        return self._result

    @property
    def error(self) -> str | None:
        """Exception text if the run failed, else ``None``."""
        return self._error

    # -- thread body -------------------------------------------------------
    def _run(
        self,
        epub_path: Path,
        settings: AppSettings,
        options: RunOptions,
        components: PipelineComponents | None,
    ) -> None:
        try:
            cfg, _output_dir = book_config_for(epub_path, settings)
            result = run_book(
                cfg,
                resolve_path=_resolve_path,
                options=options,
                components=components,
                progress=self._log,
                cancel=self._cancel,
                on_event=self._event,
            )
            self._result = result
            self._log("Cancelled." if result.cancelled else "Finished.")
        except Exception as exc:  # noqa: BLE001 - surfaced through .error
            self._error = f"{type(exc).__name__}: {exc}"
            self._log(f"ERROR: {self._error}")
        finally:
            # A cancel is cleared when the run ends so the next start is not
            # cancelled. Cancelling while idle (before start) still cancels the
            # next run, which is the documented "cancel before processing".
            self._cancel.clear()
