"""Persistent translation queue for the app.

Books are translated one after another. The queue is stored as JSON in
``data_dir()/queue.json`` so it survives an app restart; a missing or corrupt
file is treated as an empty queue. The queue always comes back **paused**, and
an item left ``running`` when the app quit goes back to ``pending`` so the next
Start resumes it (the pipeline resumes from its checkpoints).

The queue owns its data only: it never imports FastAPI and never starts a run.
The server drives it with the same code path as a single-book run.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .paths import data_dir

logger = logging.getLogger(__name__)

PENDING = "pending"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
CANCELLED = "cancelled"

STATES = {PENDING, RUNNING, DONE, FAILED, CANCELLED}
ACTIVE_STATES = {PENDING, RUNNING}


class QueueError(Exception):
    """Base class for queue problems the API maps to HTTP errors."""


class ItemNotFound(QueueError):
    """No queue item has that id."""


class ItemRunning(QueueError):
    """The item is the one currently being translated."""


class StateError(QueueError):
    """The item is in the wrong state for the requested change."""


def queue_path() -> Path:
    return data_dir() / "queue.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class QueueItem:
    """One book waiting to be translated."""

    id: str
    book_id: str
    filename: str
    title: str
    skip_chapters: list[str] = field(default_factory=list)
    qa: bool = False
    audiobook: bool = False
    state: str = PENDING
    error: str | None = None
    added: str = field(default_factory=_now_iso)
    finished: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "book_id": self.book_id,
            "filename": self.filename,
            "title": self.title,
            "skip_chapters": list(self.skip_chapters),
            "qa": self.qa,
            "audiobook": self.audiobook,
            "state": self.state,
            "error": self.error,
            "added": self.added,
            "finished": self.finished,
        }

    @classmethod
    def from_dict(cls, data: dict) -> QueueItem:
        state = str(data.get("state", PENDING))
        if state not in STATES:
            state = PENDING
        return cls(
            id=str(data["id"]),
            book_id=str(data["book_id"]),
            filename=str(data.get("filename", "")),
            title=str(data.get("title", "")),
            skip_chapters=[str(value) for value in data.get("skip_chapters", [])],
            qa=bool(data.get("qa", False)),
            audiobook=bool(data.get("audiobook", False)),
            state=state,
            error=data.get("error"),
            added=str(data.get("added") or _now_iso()),
            finished=data.get("finished"),
        )


class BookQueue:
    """Thread-safe, atomically persisted queue of books to translate.

    ``active`` (running vs paused) is in memory only: every construction loads
    the file paused, which is also what makes a restart never translate on its
    own.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else queue_path()
        self._lock = threading.RLock()
        self._items: list[QueueItem] = []
        self._active = False
        self._load()

    # -- loading / saving --------------------------------------------------
    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("Ignoring unreadable queue file %s: %s", self.path, exc)
            return
        items = raw.get("items") if isinstance(raw, dict) else None
        if not isinstance(items, list):
            logger.warning("Ignoring queue file without an item list: %s", self.path)
            return
        loaded: list[QueueItem] = []
        for entry in items:
            if not isinstance(entry, dict):
                logger.warning("Skipping malformed queue entry in %s", self.path)
                continue
            try:
                item = QueueItem.from_dict(entry)
            except (KeyError, TypeError, ValueError):
                logger.warning("Skipping malformed queue entry in %s", self.path)
                continue
            # The app quit mid-book; resume it the next time the queue runs.
            if item.state == RUNNING:
                item.state = PENDING
                item.error = None
                item.finished = None
            loaded.append(item)
        self._items = loaded

    def _save_locked(self) -> None:
        payload = {"items": [item.to_dict() for item in self._items]}
        directory = self.path.parent
        directory.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=directory,
            prefix="queue-",
            suffix=".tmp",
            delete=False,
        )
        try:
            with handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            os.replace(handle.name, self.path)
        except BaseException:
            try:
                os.unlink(handle.name)
            except OSError:
                pass
            raise

    # -- reads -------------------------------------------------------------
    @property
    def active(self) -> bool:
        with self._lock:
            return self._active

    def items(self) -> list[QueueItem]:
        with self._lock:
            return list(self._items)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "active": self._active,
                "items": [item.to_dict() for item in self._items],
            }

    def get(self, item_id: str) -> QueueItem | None:
        with self._lock:
            return self._find_locked(item_id, required=False)

    def next_pending(self) -> QueueItem | None:
        with self._lock:
            return next((item for item in self._items if item.state == PENDING), None)

    def has_pending(self) -> bool:
        return self.next_pending() is not None

    def active_book_ids(self) -> set[str]:
        with self._lock:
            return {
                item.book_id for item in self._items if item.state in ACTIVE_STATES
            }

    # -- changes -----------------------------------------------------------
    def add(self, entries: list[dict]) -> list[dict]:
        """Append already-validated entries; stamps id/added and persists."""
        with self._lock:
            created: list[QueueItem] = []
            for entry in entries:
                item = QueueItem(
                    id=uuid.uuid4().hex,
                    book_id=str(entry["book_id"]),
                    filename=str(entry.get("filename", "")),
                    title=str(entry.get("title", entry["book_id"])),
                    skip_chapters=[str(value) for value in entry.get("skip_chapters", [])],
                    qa=bool(entry.get("qa", False)),
                    audiobook=bool(entry.get("audiobook", False)),
                )
                self._items.append(item)
                created.append(item)
            self._save_locked()
            return [item.to_dict() for item in created]

    def remove(self, item_id: str) -> dict:
        with self._lock:
            item = self._find_locked(item_id)
            if item.state == RUNNING:
                raise ItemRunning(
                    "Stop the translation before removing this book."
                )
            self._items.remove(item)
            self._save_locked()
            return item.to_dict()

    def move(self, item_id: str, direction: str) -> None:
        if direction not in ("up", "down"):
            raise StateError('Direction must be "up" or "down".')
        with self._lock:
            item = self._find_locked(item_id)
            index = self._items.index(item)
            target = index - 1 if direction == "up" else index + 1
            if 0 <= target < len(self._items):
                self._items[index], self._items[target] = (
                    self._items[target],
                    self._items[index],
                )
                self._save_locked()

    def clear_finished(self) -> None:
        with self._lock:
            self._items = [
                item for item in self._items if item.state in ACTIVE_STATES
            ]
            self._save_locked()

    def retry(self, item_id: str) -> dict:
        with self._lock:
            item = self._find_locked(item_id)
            if item.state not in (FAILED, CANCELLED):
                raise StateError("Only a failed or stopped book can be retried.")
            item.state = PENDING
            item.error = None
            item.finished = None
            self._save_locked()
            return item.to_dict()

    def mark_running(self, item_id: str) -> dict:
        with self._lock:
            item = self._find_locked(item_id)
            item.state = RUNNING
            item.error = None
            item.finished = None
            self._save_locked()
            return item.to_dict()

    def mark_done(self, item_id: str) -> dict:
        return self._finish(item_id, DONE)

    def mark_failed(self, item_id: str, error: str | None) -> dict:
        return self._finish(item_id, FAILED, error)

    def mark_cancelled(self, item_id: str) -> dict:
        return self._finish(item_id, CANCELLED)

    def _finish(self, item_id: str, state: str, error: str | None = None) -> dict:
        with self._lock:
            item = self._find_locked(item_id)
            item.state = state
            item.error = error
            item.finished = _now_iso()
            self._save_locked()
            return item.to_dict()

    def set_active(self, value: bool) -> None:
        with self._lock:
            self._active = bool(value)

    def _find_locked(self, item_id: str, *, required: bool = True) -> QueueItem | None:
        for item in self._items:
            if item.id == item_id:
                return item
        if required:
            raise ItemNotFound(item_id)
        return None
