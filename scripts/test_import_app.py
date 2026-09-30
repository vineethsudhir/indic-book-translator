"""Offline tests for the text/HTML import endpoints (issue #7, part B).

Fast, no models, no network: ``create_app`` and FastAPI's ``TestClient``, like
``test_app.py``. Each test gets its own app-data directory so stored imports,
books and settings stay isolated.

Run: .venv/bin/python scripts/test_import_app.py
"""

import os
import re
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE_DATA_DIR = Path(tempfile.mkdtemp(prefix="kannada-import-test-"))
# Must be set before the app package reads its paths.
os.environ["KANNADA_APP_DATA_DIR"] = str(BASE_DATA_DIR)

sys.path.insert(0, str(ROOT / "src"))

from kannada_epub.app import server as app_server  # noqa: E402
from kannada_epub.app.paths import books_dir, data_dir  # noqa: E402
from kannada_epub.epub_check import check_source_epub  # noqa: E402
from kannada_epub.epub_io import load_epub_chapters  # noqa: E402

CONTENTS_TEXT = (
    "CONTENTS\n\nPAGE\n\n"
    "Alpha Story 1\n\nBeta Story 15\n\nGamma Story 29\n\n"
    "Alpha Story\n\nThe alpha body.\n\n"
    "Beta Story\n\nThe beta body.\n\n"
    "Gamma Story\n\nThe gamma body.\n"
)

IMPORT_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def use_temp_dir(name: str) -> Path:
    path = BASE_DATA_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    os.environ["KANNADA_APP_DATA_DIR"] = str(path)
    return path


def new_client():
    from fastapi.testclient import TestClient

    from kannada_epub.app.server import create_app

    app = create_app()
    token = app.state.app_token
    headers = {"Host": "127.0.0.1:7860", "X-App-Token": token}
    return app, TestClient(app), headers


def preview(client, headers, *, filename=None, content=None, data=None):
    files = None
    if filename is not None:
        files = {"file": (filename, content, "application/octet-stream")}
    return client.post(
        "/api/import/preview", headers=headers, data=data or {}, files=files
    )


def title_of(chapter: dict) -> str:
    return chapter["title"]


# ---------------------------------------------------------------------------
# Plain text
# ---------------------------------------------------------------------------
def test_preview_text() -> None:
    use_temp_dir("preview-text")
    _app, client, headers = new_client()
    with client:
        response = preview(
            client, headers, filename="book.txt", content=CONTENTS_TEXT.encode()
        )
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["kind"] == "text", data
        assert IMPORT_ID_RE.fullmatch(data["import_id"]), data["import_id"]
        titles = [title_of(chapter) for chapter in data["chapters"]]
        assert len(titles) >= 3, titles
        assert "Alpha Story" in titles and "Beta Story" in titles, titles
        assert data["total_paragraphs"] > 0, data
        assert data["headings"] == ["Alpha Story", "Beta Story", "Gamma Story"], data
        stored = data_dir() / "imports" / f"{data['import_id']}.txt"
        assert stored.is_file(), stored


# ---------------------------------------------------------------------------
# Re-preview with edited headings (no second upload)
# ---------------------------------------------------------------------------
def test_repreview_with_headings() -> None:
    use_temp_dir("repreview")
    _app, client, headers = new_client()
    with client:
        first = preview(
            client, headers, filename="book.txt", content=CONTENTS_TEXT.encode()
        ).json()
        import_id = first["import_id"]
        assert (data_dir() / "imports" / f"{import_id}.txt").is_file()

        response = preview(
            client,
            headers,
            data={"import_id": import_id, "headings": "Beta Story"},
        )
        assert response.status_code == 200, response.text
        updated = response.json()
        assert updated["import_id"] == import_id, updated
        titles = [title_of(chapter) for chapter in updated["chapters"]]
        assert titles == ["Front matter", "Beta Story"], titles

        # Re-previewing must not store a second file.
        stored = list((data_dir() / "imports").glob("*.txt"))
        assert len(stored) == 1, stored
        assert (data_dir() / "imports" / f"{import_id}.txt").is_file()


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
def test_preview_html() -> None:
    use_temp_dir("preview-html")
    _app, client, headers = new_client()
    with client:
        html = b"<html><body><p>First para.</p><p>Second para.</p></body></html>"
        response = preview(client, headers, filename="book.html", content=html)
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["kind"] == "html", data
        assert data["total_paragraphs"] == 2, data
        assert [title_of(chapter) for chapter in data["chapters"]] == ["Front matter"]
        assert data["chapters"][0]["first"] == "First para.", data


# ---------------------------------------------------------------------------
# OCR clean-up: page numbers and repeated running headers
# ---------------------------------------------------------------------------
def test_preview_ocr() -> None:
    use_temp_dir("preview-ocr")
    _app, client, headers = new_client()
    with client:
        text = (
            "RUNNING HEAD\n\nBody one.\n\n12\n\nRUNNING HEAD\n\n"
            "Body two.\n\nRUNNING HEAD\n\nBody three.\n"
        )
        response = preview(
            client,
            headers,
            filename="scan.txt",
            content=text.encode(),
            data={"ocr": "true"},
        )
        assert response.status_code == 200, response.text
        data = response.json()
        # One page number and two repeated running headers are dropped.
        assert data["total_paragraphs"] == 4, data
        assert any("running page header" in w for w in data["warnings"]), data
        assert all(chapter["first"] != "12" for chapter in data["chapters"]), data


# ---------------------------------------------------------------------------
# Create: EPUB written to books_dir and loaded like an upload
# ---------------------------------------------------------------------------
def test_create() -> None:
    use_temp_dir("create")
    _app, client, headers = new_client()
    with client:
        first = preview(
            client, headers, filename="book.txt", content=CONTENTS_TEXT.encode()
        ).json()
        import_id = first["import_id"]
        response = client.post(
            "/api/import/create",
            headers=headers,
            json={
                "import_id": import_id,
                "ocr": False,
                "title": "My Imported Book",
                "author": "Someone",
                "date": "1900",
                "source": "https://example.org/book",
            },
        )
        assert response.status_code == 200, response.text
        book = response.json()
        assert book["book_id"] == "My_Imported_Book", book
        assert book["filename"] == "My_Imported_Book.epub", book
        assert book["title"] == "My Imported Book", book
        assert book["author"] == "Someone", book
        assert book["chapters"], book
        assert book["source_problem_count"] == 0, book
        assert book["source_problems"] == [], book
        assert "edition" in book, book

        dest = books_dir() / book["filename"]
        assert dest.is_file(), dest
        loaded = load_epub_chapters(dest)
        assert "Alpha Story" in [chapter.title for chapter in loaded], loaded
        assert check_source_epub(dest) == [], check_source_epub(dest)

        # The stored import is deleted once it becomes a book.
        assert not (data_dir() / "imports" / f"{import_id}.txt").exists()

        # A second book with the same title gets a _2 name.
        second = preview(
            client, headers, filename="book.txt", content=CONTENTS_TEXT.encode()
        ).json()
        again = client.post(
            "/api/import/create",
            headers=headers,
            json={"import_id": second["import_id"], "title": "My Imported Book"},
        )
        assert again.status_code == 200, again.text
        assert again.json()["book_id"] == "My_Imported_Book_2", again.json()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def test_validation() -> None:
    use_temp_dir("validation")
    _app, client, headers = new_client()
    no_token = {"Host": "127.0.0.1:7860"}
    with client:
        # Wrong extension.
        bad = preview(client, headers, filename="book.md", content=b"# Title")
        assert bad.status_code == 400, bad.text
        assert bad.json()["detail"] == "Choose a .txt or .html file."

        # Oversize (patch the limit down instead of sending 20 MB).
        original_limit = app_server._IMPORT_MAX_BYTES
        app_server._IMPORT_MAX_BYTES = 16
        try:
            big = preview(client, headers, filename="big.txt", content=b"x" * 64)
        finally:
            app_server._IMPORT_MAX_BYTES = original_limit
        assert big.status_code == 413, big.text

        # Bad, unknown and traversing import ids are all 404.
        for bad_id in ("nothex", uuid.uuid4().hex, "../" + "a" * 29):
            response = preview(client, headers, data={"import_id": bad_id})
            assert response.status_code == 404, (bad_id, response.status_code)

        # Missing title.
        good = preview(
            client, headers, filename="book.txt", content=CONTENTS_TEXT.encode()
        ).json()
        missing = client.post(
            "/api/import/create",
            headers=headers,
            json={"import_id": good["import_id"], "title": ""},
        )
        assert missing.status_code == 400, missing.text
        assert missing.json()["detail"] == "Enter a title for the book."

        # Empty text produces no paragraphs.
        empty = preview(
            client, headers, filename="empty.txt", content=b"   \n  \n"
        ).json()
        assert empty["total_paragraphs"] == 0, empty
        no_text = client.post(
            "/api/import/create",
            headers=headers,
            json={"import_id": empty["import_id"], "title": "Nothing"},
        )
        assert no_text.status_code == 400, no_text.text
        assert no_text.json()["detail"] == "No text was found in this file."

        # The token is required on all three endpoints.
        assert preview(
            client, no_token, filename="book.txt", content=CONTENTS_TEXT.encode()
        ).status_code in (401, 403)
        assert client.post(
            "/api/import/create", headers=no_token, json={"title": "x"}
        ).status_code in (401, 403)


# ---------------------------------------------------------------------------
# Stale imports are swept on the next preview
# ---------------------------------------------------------------------------
def test_stale_imports() -> None:
    use_temp_dir("stale")
    _app, client, headers = new_client()
    with client:
        old = preview(
            client, headers, filename="old.txt", content=CONTENTS_TEXT.encode()
        ).json()
        old_path = data_dir() / "imports" / f"{old['import_id']}.txt"
        assert old_path.is_file()
        aged = time.time() - app_server._IMPORT_TTL_SECONDS - 60
        os.utime(old_path, (aged, aged))

        fresh = preview(
            client, headers, filename="new.txt", content=CONTENTS_TEXT.encode()
        ).json()
        assert not old_path.exists(), "stale import was not deleted"
        assert (data_dir() / "imports" / f"{fresh['import_id']}.txt").is_file()


def main() -> None:
    try:
        test_preview_text()
        test_repreview_with_headings()
        test_preview_html()
        test_preview_ocr()
        test_create()
        test_validation()
        test_stale_imports()
    finally:
        shutil.rmtree(BASE_DATA_DIR, ignore_errors=True)

    print("test_import_app: all assertions passed")


if __name__ == "__main__":
    main()
