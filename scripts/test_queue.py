"""Offline tests for the persistent translation queue (issue #10).

Fast, no models, no network: fake components like ``test_app.py``. Every test
gets its own app-data directory so ``queue.json`` and the outputs stay isolated.

Run: .venv/bin/python scripts/test_queue.py
"""

import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE_DATA_DIR = Path(tempfile.mkdtemp(prefix="kannada-queue-test-"))
# Must be set before the app package reads its paths.
os.environ["KANNADA_APP_DATA_DIR"] = str(BASE_DATA_DIR)

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import test_pipeline as tp  # noqa: E402  (reused fake engines)
from epub_fixture import build_epub  # noqa: E402

from kannada_epub.app.paths import books_dir, outputs_dir  # noqa: E402
from kannada_epub.app.settings import KNOWN_KEYS, save_secret  # noqa: E402
from kannada_epub.consistency_editor import ConsistencyEditor  # noqa: E402
from kannada_epub.glossary import GlossaryStore  # noqa: E402
from kannada_epub.pipeline import PipelineComponents  # noqa: E402

_GLOSSARY_COUNTER = [0]
RUN_LOG: list[str] = []
BLOCK_ENGINE: dict = {
    "block": False,
    "entered": threading.Event(),
    "release": threading.Event(),
}


class QueueEngine(tp.FakeTranslationEngine):
    """Records which book it saw, can block, and can fail on demand."""

    def translate_paragraphs(self, paragraphs, src_lang, tgt_lang):
        marker = next((p for p in paragraphs if "BOOK:" in p), "")
        marker = marker.split()[0] if marker else "BOOK:?"
        RUN_LOG.append(marker)
        if BLOCK_ENGINE["block"] and marker == "BOOK:SLOW":
            BLOCK_ENGINE["entered"].set()
            if not BLOCK_ENGINE["release"].wait(timeout=30):
                raise RuntimeError("blocked fake engine was never released")
        if any("FAIL" in p for p in paragraphs):
            raise RuntimeError("synthetic translation failure")
        return super().translate_paragraphs(paragraphs, src_lang, tgt_lang)


def components_factory() -> PipelineComponents:
    _GLOSSARY_COUNTER[0] += 1
    return PipelineComponents(
        translation_engine=QueueEngine(),
        editor=ConsistencyEditor(tp.FakeEditorProvider()),
        glossary_store=GlossaryStore(
            BASE_DATA_DIR / f"queue-glossary-{_GLOSSARY_COUNTER[0]}.db"
        ),
    )


def use_temp_dir(name: str) -> Path:
    path = BASE_DATA_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    os.environ["KANNADA_APP_DATA_DIR"] = str(path)
    return path


def build_book(path: Path, marker: str, title: str, *, chapters: int = 1, fail: bool = False):
    documents = []
    for index in range(chapters):
        text = f"{marker} chapter {index} opening paragraph."
        if fail:
            text = f"FAIL {text}"
        body = "".join(
            f"<p>{text} sentence {number}.</p>" for number in range(2)
        )
        documents.append({
            "id": f"c{index}",
            "href": f"c{index}.xhtml",
            "content": (
                '<?xml version="1.0" encoding="utf-8"?>'
                '<html xmlns="http://www.w3.org/1999/xhtml">'
                f"<head><title>{title} {index}</title></head>"
                f"<body><h1>Chapter {index}</h1>{body}</body></html>"
            ),
        })
    build_epub(path, documents, title=title, creator="Tester", identifier=f"{marker}-id")


def new_client():
    from fastapi.testclient import TestClient

    from kannada_epub.app.server import create_app

    app = create_app(components_factory=components_factory)
    token = app.state.app_token
    headers = {"Host": "127.0.0.1:7860", "X-App-Token": token}
    return app, TestClient(app), headers


def wait_until(predicate, timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def set_keys() -> None:
    os.environ["SARVAM_API_KEY"] = "queue-test-sarvam"
    os.environ["ANTHROPIC_API_KEY"] = "queue-test-anthropic"


def clear_keys() -> None:
    for name in KNOWN_KEYS:
        os.environ.pop(name, None)
        save_secret(name, "")


# ---------------------------------------------------------------------------
# GET /api/books
# ---------------------------------------------------------------------------
def test_list_books() -> None:
    use_temp_dir("list-books")
    books = books_dir()
    build_epub(
        books / "alpha.epub",
        [{"id": "c0", "href": "c0.xhtml", "content": "<html><body><p>Hi.</p></body></html>"}],
        title="Alpha",
        creator="Author A",
    )
    build_epub(
        books / "beta.epub",
        [{"id": "c0", "href": "c0.xhtml", "content": "<html><body><p>Hi.</p></body></html>"}],
        title="Beta",
        creator="Author B",
    )
    (books / "notes.txt").write_text("not a book", encoding="utf-8")
    (books / "subdir").mkdir()
    (books / "broken.epub").write_text("this is not a zip", encoding="utf-8")

    _app, client, headers = new_client()
    with client:
        response = client.get("/api/books", headers=headers)
        assert response.status_code == 200, response.text
        listing = response.json()
        ids = [item["book_id"] for item in listing]
        assert ids == ["alpha", "beta", "broken"], ids
        assert "notes" not in ids and "subdir" not in ids
        alpha = next(item for item in listing if item["book_id"] == "alpha")
        assert alpha["title"] == "Alpha" and alpha["author"] == "Author A"
        assert "unreadable" not in alpha
        broken = next(item for item in listing if item["book_id"] == "broken")
        assert broken["unreadable"] is True, broken
        assert broken["title"] == "broken.epub", broken


# ---------------------------------------------------------------------------
# Running in order
# ---------------------------------------------------------------------------
def test_queue_runs_in_order() -> None:
    use_temp_dir("in-order")
    set_keys()
    build_book(books_dir() / "alpha.epub", "BOOK:ONE", "Alpha")
    build_book(books_dir() / "beta.epub", "BOOK:TWO", "Beta")
    RUN_LOG.clear()

    _app, client, headers = new_client()
    with client:
        added = client.post("/api/queue", headers=headers, json={
            "items": [{"book_id": "alpha"}, {"book_id": "beta"}],
        })
        assert added.status_code == 200, added.text
        assert [item["state"] for item in added.json()["items"]] == ["pending", "pending"]
        assert added.json()["active"] is False

        started = client.post("/api/queue/start", headers=headers)
        assert started.status_code == 200, started.text
        assert started.json()["active"] is True

        saw_two_running = False

        def all_done() -> bool:
            nonlocal saw_two_running
            items = client.get("/api/queue", headers=headers).json()["items"]
            if sum(1 for item in items if item["state"] == "running") > 1:
                saw_two_running = True
            return items and all(item["state"] == "done" for item in items)

        assert wait_until(all_done), "queue did not finish both books"
        assert not saw_two_running, "two books ran at the same time"

        snapshot = client.get("/api/queue", headers=headers).json()
        assert [item["state"] for item in snapshot["items"]] == ["done", "done"]
        assert snapshot["active"] is False, snapshot
        assert RUN_LOG[:2] == ["BOOK:ONE", "BOOK:TWO"], RUN_LOG

        assert (outputs_dir() / "alpha" / "alpha.kn.epub").is_file()
        assert (outputs_dir() / "beta" / "beta.kn.epub").is_file()


# ---------------------------------------------------------------------------
# A failing book does not stop the queue
# ---------------------------------------------------------------------------
def test_failure_continues() -> None:
    use_temp_dir("failure")
    set_keys()
    build_book(books_dir() / "alpha.epub", "BOOK:FAIL", "Alpha", fail=True)
    build_book(books_dir() / "beta.epub", "BOOK:TWO", "Beta")

    _app, client, headers = new_client()
    with client:
        client.post("/api/queue", headers=headers, json={
            "items": [{"book_id": "alpha"}, {"book_id": "beta"}],
        })
        client.post("/api/queue/start", headers=headers)

        def finished() -> bool:
            items = client.get("/api/queue", headers=headers).json()["items"]
            return items and all(item["state"] in {"done", "failed"} for item in items)

        assert wait_until(finished), "queue did not move past the failing book"
        items = client.get("/api/queue", headers=headers).json()["items"]
        assert items[0]["state"] == "failed", items
        assert "synthetic translation failure" in (items[0]["error"] or ""), items
        assert items[1]["state"] == "done", items
        assert (outputs_dir() / "beta" / "beta.kn.epub").is_file()


# ---------------------------------------------------------------------------
# Stop cancels the current book and pauses the queue
# ---------------------------------------------------------------------------
def test_stop_pauses() -> None:
    use_temp_dir("stop")
    set_keys()
    build_book(books_dir() / "slow.epub", "BOOK:SLOW", "Slow", chapters=2)
    build_book(books_dir() / "next.epub", "BOOK:NEXT", "Next")

    BLOCK_ENGINE["block"] = True
    BLOCK_ENGINE["entered"].clear()
    BLOCK_ENGINE["release"].clear()
    try:
        _app, client, headers = new_client()
        with client:
            client.post("/api/queue", headers=headers, json={
                "items": [{"book_id": "slow"}, {"book_id": "next"}],
            })
            client.post("/api/queue/start", headers=headers)
            assert BLOCK_ENGINE["entered"].wait(timeout=30), "slow book never started"

            stopped = client.post("/api/run/stop", headers=headers)
            assert stopped.status_code == 200, stopped.text
            BLOCK_ENGINE["release"].set()

            def cancelled() -> bool:
                items = client.get("/api/queue", headers=headers).json()["items"]
                return items and items[0]["state"] == "cancelled"

            assert wait_until(cancelled), "stop did not cancel the book"
            snapshot = client.get("/api/queue", headers=headers).json()
            assert snapshot["items"][0]["state"] == "cancelled", snapshot
            assert snapshot["items"][1]["state"] == "pending", snapshot
            assert snapshot["active"] is False, snapshot
    finally:
        BLOCK_ENGINE["block"] = False
        BLOCK_ENGINE["release"].set()


# ---------------------------------------------------------------------------
# A missing API key pauses instead of spinning through every book
# ---------------------------------------------------------------------------
def test_missing_key_pauses() -> None:
    use_temp_dir("missing-key")
    clear_keys()
    build_book(books_dir() / "alpha.epub", "BOOK:ONE", "Alpha")
    build_book(books_dir() / "beta.epub", "BOOK:TWO", "Beta")

    _app, client, headers = new_client()
    with client:
        client.post("/api/queue", headers=headers, json={
            "items": [{"book_id": "alpha"}, {"book_id": "beta"}],
        })
        started = client.post("/api/queue/start", headers=headers)
        assert started.status_code == 200, started.text
        snapshot = started.json()
        assert snapshot["items"][0]["state"] == "failed", snapshot
        assert "API keys" in (snapshot["items"][0]["error"] or ""), snapshot
        assert snapshot["items"][1]["state"] == "pending", snapshot
        assert snapshot["active"] is False, snapshot


# ---------------------------------------------------------------------------
# Persistence and restart behaviour
# ---------------------------------------------------------------------------
def test_persistence() -> None:
    use_temp_dir("persistence")
    set_keys()
    build_book(books_dir() / "alpha.epub", "BOOK:ONE", "Alpha")
    build_book(books_dir() / "beta.epub", "BOOK:TWO", "Beta")

    app1, client1, headers1 = new_client()
    with client1:
        added = client1.post("/api/queue", headers=headers1, json={
            "items": [{"book_id": "alpha"}, {"book_id": "beta"}],
        }).json()
        first_id = added["items"][0]["id"]
        app1.state.queue.mark_running(first_id)
        assert app1.state.queue.snapshot()["items"][0]["state"] == "running"

    # A brand-new app on the same data dir: the running item resumes as pending
    # and the queue comes back paused.
    _app2, client2, headers2 = new_client()
    with client2:
        snapshot = client2.get("/api/queue", headers=headers2).json()
        assert snapshot["items"][0]["state"] == "pending", snapshot
        assert snapshot["items"][0]["error"] is None
        assert snapshot["active"] is False, snapshot


def test_corrupt_queue_file() -> None:
    data = use_temp_dir("corrupt")
    (data / "queue.json").write_text("{ this is not json", encoding="utf-8")

    _app, client, headers = new_client()
    with client:
        snapshot = client.get("/api/queue", headers=headers).json()
        assert snapshot == {"active": False, "items": []}, snapshot

    # A structurally wrong (but valid JSON) file is also an empty queue.
    (data / "queue.json").write_text('{"items": "nope"}', encoding="utf-8")
    _app, client, headers = new_client()
    with client:
        snapshot = client.get("/api/queue", headers=headers).json()
        assert snapshot == {"active": False, "items": []}, snapshot


# ---------------------------------------------------------------------------
# Validation and item actions
# ---------------------------------------------------------------------------
def test_validation_and_actions() -> None:
    use_temp_dir("validation")
    set_keys()
    build_book(books_dir() / "alpha.epub", "BOOK:ONE", "Alpha")
    build_book(books_dir() / "beta.epub", "BOOK:TWO", "Beta")

    app, client, headers = new_client()
    with client:
        def queue() -> dict:
            return client.get("/api/queue", headers=headers).json()

        # Unknown book id.
        bad = client.post("/api/queue", headers=headers, json={
            "items": [{"book_id": "missing"}],
        })
        assert bad.status_code == 400, bad.text
        assert queue()["items"] == []

        # Path traversal in the book id.
        bad = client.post("/api/queue", headers=headers, json={
            "items": [{"book_id": "../alpha"}],
        })
        assert bad.status_code == 400, bad.text
        assert queue()["items"] == []

        # skip_chapters must be a list of strings.
        bad = client.post("/api/queue", headers=headers, json={
            "items": [{"book_id": "alpha", "skip_chapters": "c0"}],
        })
        assert bad.status_code == 400, bad.text
        # An unknown section is refused.
        bad = client.post("/api/queue", headers=headers, json={
            "items": [{"book_id": "alpha", "skip_chapters": ["nope"]}],
        })
        assert bad.status_code == 400, bad.text
        # Skipping every section is refused.
        bad = client.post("/api/queue", headers=headers, json={
            "items": [{"book_id": "alpha", "skip_chapters": ["c0"]}],
        })
        assert bad.status_code == 400, bad.text
        assert queue()["items"] == []

        # All-or-nothing: a valid item plus an invalid one adds nothing.
        bad = client.post("/api/queue", headers=headers, json={
            "items": [{"book_id": "alpha"}, {"book_id": "missing"}],
        })
        assert bad.status_code == 400, bad.text
        assert queue()["items"] == []

        # A valid add, then a duplicate while pending is refused.
        added = client.post("/api/queue", headers=headers, json={
            "items": [{"book_id": "alpha"}, {"book_id": "beta"}],
        })
        assert added.status_code == 200, added.text
        items = added.json()["items"]
        assert [item["title"] for item in items] == ["Alpha", "Beta"]
        alpha_id, beta_id = items[0]["id"], items[1]["id"]
        assert items[0]["qa"] is False and items[0]["audiobook"] is False

        duplicate = client.post("/api/queue", headers=headers, json={
            "items": [{"book_id": "alpha"}],
        })
        assert duplicate.status_code == 400, duplicate.text
        assert "already in the queue" in duplicate.json()["detail"]

        # qa/audiobook come from the request body.
        build_book(books_dir() / "gamma.epub", "BOOK:THREE", "Gamma")
        flagged = client.post("/api/queue", headers=headers, json={
            "items": [{"book_id": "gamma"}], "qa": True, "audiobook": True,
        })
        assert flagged.status_code == 200, flagged.text
        gamma_item = next(item for item in flagged.json()["items"] if item["book_id"] == "gamma")
        assert gamma_item["qa"] is True and gamma_item["audiobook"] is True, gamma_item
        # Remove gamma again so the later assertions are about alpha/beta.
        assert client.delete(f"/api/queue/{gamma_item['id']}", headers=headers).status_code == 200

        # Move.
        moved = client.post(f"/api/queue/{beta_id}/move", headers=headers, json={"direction": "up"})
        assert moved.status_code == 200, moved.text
        assert [item["id"] for item in moved.json()["items"]] == [beta_id, alpha_id]
        no_op = client.post(f"/api/queue/{beta_id}/move", headers=headers, json={"direction": "up"})
        assert no_op.status_code == 200
        assert [item["id"] for item in no_op.json()["items"]] == [beta_id, alpha_id]
        bad_direction = client.post(f"/api/queue/{beta_id}/move", headers=headers, json={"direction": "sideways"})
        assert bad_direction.status_code == 400, bad_direction.text

        # Retry rejects a pending item; a failed item goes back to pending.
        cannot_retry = client.post(f"/api/queue/{beta_id}/retry", headers=headers)
        assert cannot_retry.status_code == 400, cannot_retry.text
        app.state.queue.mark_failed(beta_id, "boom")
        retried = client.post(f"/api/queue/{beta_id}/retry", headers=headers)
        assert retried.status_code == 200, retried.text
        retried_item = next(item for item in retried.json()["items"] if item["id"] == beta_id)
        assert retried_item["state"] == "pending" and retried_item["error"] is None

        # Removing the running item is refused.
        app.state.queue.mark_running(beta_id)
        running_remove = client.delete(f"/api/queue/{beta_id}", headers=headers)
        assert running_remove.status_code == 409, running_remove.text
        assert "Stop the translation" in running_remove.json()["detail"]

        # Unknown ids are 404.
        assert client.delete("/api/queue/nope", headers=headers).status_code == 404
        assert client.post("/api/queue/nope/retry", headers=headers).status_code == 404
        assert client.post("/api/queue/nope/move", headers=headers, json={"direction": "up"}).status_code == 404

        # Clear finished removes done/failed/cancelled but keeps the rest.
        app.state.queue.mark_done(beta_id)
        app.state.queue.mark_failed(alpha_id, "gone")
        cleared = client.post("/api/queue/clear", headers=headers)
        assert cleared.status_code == 200, cleared.text
        assert cleared.json()["items"] == [], cleared.json()


def test_queue_start_guards() -> None:
    use_temp_dir("guards")
    set_keys()

    _app, client, headers = new_client()
    with client:
        # Empty queue.
        empty = client.post("/api/queue/start", headers=headers)
        assert empty.status_code == 400, empty.text
        assert empty.json()["detail"] == "The queue has no books waiting."

        build_book(books_dir() / "alpha.epub", "BOOK:ONE", "Alpha")
        client.post("/api/queue", headers=headers, json={"items": [{"book_id": "alpha"}]})

        # A single-book run in progress blocks starting the queue.
        build_book(books_dir() / "slow.epub", "BOOK:SLOW", "Slow")
        BLOCK_ENGINE["block"] = True
        BLOCK_ENGINE["entered"].clear()
        BLOCK_ENGINE["release"].clear()
        try:
            started = client.post(
                "/api/run", headers=headers, json={"book_id": "slow"}
            )
            assert started.status_code == 200, started.text
            assert BLOCK_ENGINE["entered"].wait(timeout=30), "slow run never started"
            blocked = client.post("/api/queue/start", headers=headers)
            assert blocked.status_code == 409, blocked.text
            assert blocked.json()["detail"] == "A translation is already running."
        finally:
            BLOCK_ENGINE["block"] = False
            BLOCK_ENGINE["release"].set()
        assert wait_until(lambda: not _app.state.runner.is_running())


def test_token_required() -> None:
    use_temp_dir("token")
    _app, client, headers = new_client()
    no_token = {"Host": "127.0.0.1:7860"}
    with client:
        requests = [
            ("get", "/api/books", None),
            ("get", "/api/queue", None),
            ("post", "/api/queue", {"items": [{"book_id": "x"}]}),
            ("delete", "/api/queue/x", None),
            ("post", "/api/queue/x/move", {"direction": "up"}),
            ("post", "/api/queue/start", None),
            ("post", "/api/queue/pause", None),
            ("post", "/api/queue/clear", None),
            ("post", "/api/queue/x/retry", None),
        ]
        for method, url, body in requests:
            call = getattr(client, method)
            response = call(url, headers=no_token, json=body) if body is not None else call(url, headers=no_token)
            assert response.status_code == 401, (method, url, response.status_code)


def test_callback_gets_its_own_outcome() -> None:
    """A new run started before the callback reads the runner can't swap outcomes."""
    from kannada_epub.app.runner import BookRun
    from kannada_epub.app.settings import load_settings
    from kannada_epub.pipeline import RunOptions

    data = use_temp_dir("callback")
    failing, passing = data / "failing.epub", data / "passing.epub"
    build_book(failing, "BOOK:CBFAIL", "Failing", fail=True)
    build_book(passing, "BOOK:CBPASS", "Passing")
    settings = load_settings()
    runner = BookRun()
    seen: list = []
    done = threading.Event()

    def second_finished(result, error):
        seen.append(("second", result, error))
        done.set()

    def first_finished(result, error):
        # Start another run first, as a user click could: it resets the
        # runner's own result/error before this callback records anything.
        runner.start(passing, settings, RunOptions(), components=components_factory(),
                     on_finish=second_finished)
        seen.append(("first", result, error))

    runner.start(failing, settings, RunOptions(), components=components_factory(),
                 on_finish=first_finished)
    assert done.wait(60), seen
    first = next(entry for entry in seen if entry[0] == "first")
    second = next(entry for entry in seen if entry[0] == "second")
    assert first[2] and "synthetic translation failure" in first[2], first
    assert second[2] is None and second[1] is not None, second


def main() -> None:
    try:
        test_list_books()
        test_queue_runs_in_order()
        test_failure_continues()
        test_stop_pauses()
        test_missing_key_pauses()
        test_persistence()
        test_corrupt_queue_file()
        test_validation_and_actions()
        test_queue_start_guards()
        test_token_required()
        test_callback_gets_its_own_outcome()
    finally:
        shutil.rmtree(BASE_DATA_DIR, ignore_errors=True)

    print("test_queue: all assertions passed")


if __name__ == "__main__":
    main()
