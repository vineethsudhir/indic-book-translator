"""Headless browser smoke test for the local web app, driven through CDP."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EPUB = ROOT / "data" / "sherlock_holmes.epub"
CHROME_MACOS = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")


def find_chrome() -> str | None:
    if sys.platform == "darwin" and CHROME_MACOS.is_file():
        return str(CHROME_MACOS)
    return shutil.which("google-chrome") or shutil.which("chromium")


CHROME = find_chrome()
if not CHROME:
    print("test_ui_browser: skipped — Chrome not found")
    raise SystemExit(0)


def free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


DATA_DIR = Path(tempfile.mkdtemp(prefix="kannada-browser-test-"))
os.environ["KANNADA_APP_DATA_DIR"] = str(DATA_DIR)
os.environ["SARVAM_API_KEY"] = "browser-test-sarvam-key"
os.environ["ANTHROPIC_API_KEY"] = "browser-test-anthropic-key"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import test_pipeline as pipeline_test  # noqa: E402
import uvicorn  # noqa: E402
import websockets  # noqa: E402

from kannada_epub.app.server import create_app  # noqa: E402
from kannada_epub.consistency_editor import ConsistencyEditor  # noqa: E402
from kannada_epub.glossary import GlossaryStore  # noqa: E402
from kannada_epub.pipeline import PipelineComponents  # noqa: E402

# A small text book for the import panel, with a contents list so the preview
# splits into several chapters.
IMPORT_SAMPLE = DATA_DIR / "import-sample.txt"
IMPORT_SAMPLE.write_text(
    "CONTENTS\n\nPAGE\n\n"
    "Alpha Story 1\n\nBeta Story 5\n\nGamma Story 9\n\n"
    "Alpha Story\n\nThe alpha body.\n\n"
    "Beta Story\n\nThe beta body.\n\n"
    "Gamma Story\n\nThe gamma body.\n",
    encoding="utf-8",
)


def fake_components() -> PipelineComponents:
    return PipelineComponents(
        translation_engine=pipeline_test.FakeTranslationEngine(),
        editor=ConsistencyEditor(pipeline_test.FakeEditorProvider()),
        glossary_store=GlossaryStore(DATA_DIR / "browser-glossary.db"),
    )


class DevTools:
    """Small CDP websocket client that collects browser events while awaiting replies."""

    def __init__(self, socket_url: str):
        self.socket_url = socket_url
        self.socket = None
        self.next_id = 0
        self.pending: dict[int, asyncio.Future] = {}
        self.events: list[dict] = []
        self.reader_task: asyncio.Task | None = None

    async def __aenter__(self):
        self.socket = await websockets.connect(
            self.socket_url,
            max_size=50_000_000,
            origin=None,
        )
        self.reader_task = asyncio.create_task(self._read_messages())
        return self

    async def __aexit__(self, *_exc):
        if self.reader_task:
            self.reader_task.cancel()
        if self.socket:
            await self.socket.close()

    async def _read_messages(self) -> None:
        assert self.socket is not None
        async for raw in self.socket:
            message = json.loads(raw)
            command_id = message.get("id")
            if command_id is not None and command_id in self.pending:
                future = self.pending.pop(command_id)
                if "error" in message:
                    future.set_exception(RuntimeError(message["error"]))
                else:
                    future.set_result(message.get("result", {}))
            else:
                self.events.append(message)

    async def call(self, method: str, **params):
        assert self.socket is not None
        self.next_id += 1
        command_id = self.next_id
        future = asyncio.get_running_loop().create_future()
        self.pending[command_id] = future
        await self.socket.send(json.dumps({
            "id": command_id,
            "method": method,
            "params": params,
        }))
        return await asyncio.wait_for(future, timeout=30)

    async def evaluate(self, expression: str):
        result = await self.call(
            "Runtime.evaluate",
            expression=expression,
            awaitPromise=True,
            returnByValue=True,
        )
        if result.get("exceptionDetails"):
            raise AssertionError(result["exceptionDetails"].get("text", "Runtime error"))
        remote = result.get("result", {})
        if remote.get("subtype") == "error":
            raise AssertionError(remote.get("description", "JavaScript evaluation failed"))
        return remote.get("value")

    def clear_events(self) -> None:
        self.events.clear()

    def assert_no_browser_errors(self, context: str) -> None:
        errors = []
        for event in self.events:
            method = event.get("method")
            params = event.get("params", {})
            if method == "Runtime.exceptionThrown":
                details = params.get("exceptionDetails", {})
                description = details.get("exception", {}).get("description")
                errors.append(description or details.get("text", "JavaScript exception"))
            elif method == "Runtime.consoleAPICalled" and params.get("type") == "error":
                args = params.get("args", [])
                errors.append(" ".join(
                    str(argument.get("value", argument.get("description", "")))
                    for argument in args
                ))
            elif method == "Log.entryAdded":
                entry = params.get("entry", {})
                if entry.get("level") == "error":
                    errors.append(entry.get("text", "Browser log error"))
        assert not errors, f"Browser errors during {context}: {errors}"


async def wait_for_value(devtools: DevTools, expression: str, wanted, timeout: float = 30):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = await devtools.evaluate(expression)
        if wanted(last):
            return last
        await asyncio.sleep(0.1)
    raise AssertionError(f"Timed out waiting for browser value; last value was {last!r}")


async def wait_for_heading(devtools: DevTools, expected: str) -> None:
    await wait_for_value(
        devtools,
        "document.querySelector('#content h1')?.innerText || ''",
        lambda value: expected in value,
    )


async def check_screen(devtools: DevTools, route: str, title: str, base_url: str) -> None:
    devtools.clear_events()
    await devtools.call("Page.navigate", url=f"{base_url}/{route}")
    await wait_for_heading(devtools, title)
    content_length = await devtools.evaluate(
        "document.querySelector('#content')?.innerHTML.length || 0"
    )
    assert content_length > 0, f"{route} rendered an empty content area"
    await asyncio.sleep(0.25)
    devtools.assert_no_browser_errors(route)


async def check_sidebar_navigation(devtools: DevTools) -> None:
    destinations = [
        ("translate", "Bring a book to Kannada"),
        ("queue", "Queue"),
        ("library", "Library"),
        ("settings", "Settings"),
    ]
    for route, heading in destinations:
        devtools.clear_events()
        await devtools.evaluate(
            f"document.querySelector('[data-route={json.dumps(route)}]').click()"
        )
        await wait_for_heading(devtools, heading)
        await asyncio.sleep(0.2)
        devtools.assert_no_browser_errors(f"sidebar {route}")


async def check_sections_controls(devtools: DevTools) -> None:
    """The per-book sections list renders one checkbox per chapter and gates
    the Translate button."""
    count = await devtools.evaluate(
        "document.querySelectorAll('#sections-list [data-section-id]').length"
    )
    assert count == 13, f"expected 13 section checkboxes, got {count}"
    card_text = await devtools.evaluate(
        "document.querySelector('.book-card')?.innerText || ''"
    )
    assert f"{count} chapters" in card_text, card_text
    summary = await devtools.evaluate(
        "document.getElementById('sections-summary')?.innerText || ''"
    )
    assert summary.startswith("13 of 13"), summary

    # Unticking one chapter updates the summary count and paragraph total.
    await devtools.evaluate(
        "document.querySelector('#sections-list [data-section-id]').click()"
    )
    summary = await devtools.evaluate(
        "document.getElementById('sections-summary')?.innerText || ''"
    )
    assert summary.startswith("12 of 13"), summary

    # No sections ticked: the Translate button is disabled with a message.
    await devtools.evaluate("document.getElementById('sections-none').click()")
    disabled = await devtools.evaluate("document.getElementById('start-run').disabled")
    warning = await devtools.evaluate(
        "document.getElementById('sections-warning')?.innerText || ''"
    )
    assert disabled is True, "Translate must be disabled with no sections selected"
    assert "Choose at least one section" in warning, warning

    # Select all restores every box and re-enables the button.
    await devtools.evaluate("document.getElementById('sections-all').click()")
    disabled = await devtools.evaluate("document.getElementById('start-run').disabled")
    summary = await devtools.evaluate(
        "document.getElementById('sections-summary')?.innerText || ''"
    )
    assert disabled is False, "Translate must be enabled once sections are chosen"
    assert summary.startswith("13 of 13"), summary
    devtools.assert_no_browser_errors("sections controls")


async def check_import_panel(devtools: DevTools) -> None:
    """The import button opens the panel, and a preview lists chapters."""
    devtools.clear_events()
    await devtools.evaluate("location.hash = '#/translate'")
    await wait_for_value(
        devtools,
        "document.querySelector('#open-import') ? 1 : 0",
        lambda value: value == 1,
    )
    await devtools.evaluate("document.getElementById('open-import').click()")
    await wait_for_value(
        devtools,
        "document.querySelector('#import-panel') ? 1 : 0",
        lambda value: value == 1,
    )
    controls = await devtools.evaluate(
        "['#import-file','#import-ocr','#import-title','#import-preview',"
        "'#import-cancel'].every(s => document.querySelector(s))"
    )
    assert controls is True, "import panel controls are missing"

    document = await devtools.call("DOM.getDocument")
    node = await devtools.call(
        "DOM.querySelector", nodeId=document["root"]["nodeId"], selector="#import-file"
    )
    assert node.get("nodeId"), "import file input was not found"
    await devtools.call(
        "DOM.setFileInputFiles", files=[str(IMPORT_SAMPLE)], nodeId=node["nodeId"]
    )
    prefilled = await wait_for_value(
        devtools,
        "document.getElementById('import-title').value",
        lambda value: value == "import-sample",
    )
    assert prefilled == "import-sample", prefilled

    await devtools.evaluate("document.getElementById('import-preview').click()")
    chapter_count = await wait_for_value(
        devtools,
        "document.querySelectorAll('.chapter-preview-row').length",
        lambda value: value and value >= 3,
    )
    assert chapter_count >= 3, chapter_count
    headings = await devtools.evaluate(
        "document.getElementById('import-headings')?.value || ''"
    )
    assert "Alpha Story" in headings and "Beta Story" in headings, headings
    await asyncio.sleep(0.2)
    devtools.assert_no_browser_errors("import panel")


async def upload_and_translate(devtools: DevTools) -> tuple[str, str]:
    await devtools.evaluate("location.hash = '#/translate'")
    await wait_for_heading(devtools, "Bring a book to Kannada")
    await wait_for_value(
        devtools,
        "document.querySelector('#book-file') ? 1 : 0",
        lambda value: value == 1,
    )
    document = await devtools.call("DOM.getDocument")
    input_node = await devtools.call(
        "DOM.querySelector",
        nodeId=document["root"]["nodeId"],
        selector="#book-file",
    )
    assert input_node.get("nodeId"), "EPUB file input was not found"
    await devtools.call(
        "DOM.setFileInputFiles",
        files=[str(EPUB)],
        nodeId=input_node["nodeId"],
    )
    await wait_for_value(
        devtools,
        "document.querySelector('.book-card h3')?.innerText || ''",
        lambda value: value and "Sherlock Holmes" in value,
    )

    await check_sections_controls(devtools)

    await devtools.evaluate("document.getElementById('preview-option').click()")
    preview_is_visible = await devtools.evaluate(
        "document.getElementById('preview-wrap').classList.contains('show')"
    )
    preview_value = await devtools.evaluate(
        "document.getElementById('preview-count').value"
    )
    assert preview_is_visible and preview_value == "2"

    devtools.clear_events()
    await devtools.evaluate("document.getElementById('start-run').click()")
    await wait_for_value(
        devtools,
        "document.querySelector('.result-card h2')?.innerText || ''",
        lambda value: value == "Your preview is ready",
        timeout=120,
    )
    # A preview must say how much was translated, so it isn't mistaken for
    # the whole book (untranslated paragraphs stay English in the EPUB).
    card_text = await devtools.evaluate(
        "document.querySelector('.result-card')?.innerText || ''"
    )
    assert "Preview:" in card_text and "paragraphs translated" in card_text, card_text
    assert "stays in English" in card_text, card_text
    devtools.assert_no_browser_errors("upload and translation")
    return await get_library_book(devtools)


async def check_queue_page(devtools: DevTools, book_id: str) -> None:
    """The Queue page lists the uploaded book; adding it shows it as Waiting."""
    devtools.clear_events()
    await devtools.evaluate("location.hash = '#/queue'")
    await wait_for_heading(devtools, "Queue")
    selector = f"[data-queue-book={json.dumps(book_id)}]"
    await wait_for_value(
        devtools,
        f"document.querySelector({json.dumps(selector)}) ? 1 : 0",
        lambda value: value == 1,
    )
    await devtools.evaluate(f"document.querySelector({json.dumps(selector)}).click()")
    await devtools.evaluate("document.getElementById('queue-add').click()")
    await wait_for_value(
        devtools,
        "document.querySelector('#queue-list .queue-state')?.innerText || ''",
        lambda value: "Waiting" in value,
        timeout=120,
    )
    await asyncio.sleep(0.2)
    devtools.assert_no_browser_errors("queue page")


async def get_library_book(devtools: DevTools) -> tuple[str, str]:
    await devtools.evaluate("location.hash = '#/library'")
    await wait_for_heading(devtools, "Library")
    token = await devtools.evaluate(
        "document.querySelector('meta[name=app-token]').content"
    )
    library = await devtools.evaluate(
        "(async token => await (await fetch('/api/library', {" +
        "headers: {'X-App-Token': token}})).json())(" + json.dumps(token) + ")"
    )
    assert library, "Library API returned no translated book"
    book = library[0]
    assert book["title"] == "The Adventures of Sherlock Holmes"
    assert book["title"] != book["book_id"]
    assert book["author"]
    row_text = await devtools.evaluate(
        "document.querySelector('.library-row')?.innerText || ''"
    )
    assert book["title"] in row_text
    assert book["author"] in row_text
    assert re.search(r"\d{1,2} \w{3,4} \d{4}, \d{2}:\d{2}", row_text), row_text

    # Model a manifest created before chapter-title spacing was corrected.
    manifest_path = DATA_DIR / "outputs" / book["book_id"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_chapter = next(
        chapter for chapter in manifest["chapters"] if chapter["id"] == "item4"
    )
    manifest_chapter["title"] = "I.A SCANDAL IN BOHEMIA"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    chapters = await devtools.evaluate(
        "(async (token, bookId) => await (await fetch(" +
        "`/api/library/${encodeURIComponent(bookId)}/chapters`, {" +
        "headers: {'X-App-Token': token}})).json())(" +
        json.dumps(token) + "," + json.dumps(book["book_id"]) + ")"
    )
    item4 = next((chapter for chapter in chapters if chapter["id"] == "item4"), None)
    assert item4 is not None
    assert re.search(r"I\.\s+A", item4["title"]), item4["title"]
    return book["book_id"], item4["id"]


async def check_reader(devtools: DevTools, book_id: str, chapter_id: str) -> None:
    route = f"#/library/{book_id}/{chapter_id}"
    await devtools.evaluate(f"location.hash = {json.dumps(route)}")
    await wait_for_heading(devtools, "I.")
    title = await devtools.evaluate("document.querySelector('#content h1').innerText")
    assert re.search(r"I\.\s+A", title), f"Chapter title has missing spacing: {title!r}"
    paragraph_count = await devtools.evaluate(
        "document.querySelectorAll('.paragraph').length"
    )
    assert paragraph_count > 0
    navigation = await devtools.evaluate(
        "[...document.querySelectorAll('.chapter-navigation')].map(nav => nav.innerText)"
    )
    assert len(navigation) == 2
    assert all("Chapter 2 of 13" in value for value in navigation)
    next_link = await devtools.evaluate(
        "document.querySelector('.chapter-navigation .next[href]')?.innerText || ''"
    )
    assert "Next chapter" in next_link

    devtools.clear_events()
    await devtools.evaluate(
        "document.querySelector('.chapter-navigation .next[href]').click()"
    )
    await wait_for_value(
        devtools,
        "document.querySelector('.chapter-navigation')?.innerText || ''",
        lambda value: "Chapter 3 of 13" in value,
    )
    await asyncio.sleep(0.2)
    devtools.assert_no_browser_errors("reader next chapter")


async def run_browser_checks(base_url: str, devtools_port: int, profile: str) -> None:
    chrome = subprocess.Popen(
        [
            CHROME,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-background-networking",
            "--remote-allow-origins=*",
            f"--remote-debugging-port={devtools_port}",
            f"--user-data-dir={profile}",
            "--window-size=1280,900",
            f"{base_url}/#/translate",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        version_url = f"http://127.0.0.1:{devtools_port}/json/version"
        # A cold Chrome on a CI runner can take well over 10 s to open CDP.
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(version_url, timeout=1) as response:
                    version = json.load(response)
                break
            except Exception:
                if chrome.poll() is not None:
                    raise AssertionError("Headless Chrome exited before CDP was ready") from None
                await asyncio.sleep(0.1)
        else:
            raise AssertionError("Timed out waiting for Chrome DevTools Protocol")

        with urllib.request.urlopen(
            f"http://127.0.0.1:{devtools_port}/json/list", timeout=3
        ) as response:
            targets = json.load(response)
        page = next(target for target in targets if target.get("type") == "page")
        socket_url = page.get("webSocketDebuggerUrl") or version["webSocketDebuggerUrl"]
        async with DevTools(socket_url) as devtools:
            await devtools.call("Runtime.enable")
            await devtools.call("Log.enable")
            await devtools.call("Page.enable")
            await devtools.call("DOM.enable")
            await devtools.call("Network.enable")

            expected = [
                ("#/translate", "Bring a book to Kannada"),
                ("#/queue", "Queue"),
                ("#/library", "Library"),
                ("#/settings", "Settings"),
            ]
            for scheme in ("light", "dark"):
                await devtools.call(
                    "Emulation.setEmulatedMedia",
                    features=[{"name": "prefers-color-scheme", "value": scheme}],
                )
                for route, heading in expected:
                    await check_screen(devtools, route, heading, base_url)
                await check_sidebar_navigation(devtools)

            await devtools.call(
                "Emulation.setEmulatedMedia",
                features=[{"name": "prefers-color-scheme", "value": "light"}],
            )
            await check_import_panel(devtools)
            book_id, first_chapter = await upload_and_translate(devtools)
            await check_queue_page(devtools, book_id)
            await check_reader(devtools, book_id, first_chapter)

            favicon_status = await devtools.evaluate(
                "(async () => (await fetch('/favicon.ico')).status)()"
            )
            assert favicon_status == 200
            print("test_ui_browser: all assertions passed")
    finally:
        chrome.terminate()
        try:
            chrome.wait(timeout=5)
        except subprocess.TimeoutExpired:
            chrome.kill()
            chrome.wait()


def main() -> None:
    port = free_port()
    devtools_port = free_port()
    app = create_app(components_factory=fake_components, port=port)
    server = uvicorn.Server(uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="error",
        access_log=False,
    ))
    server_thread = threading.Thread(target=server.run, daemon=True)
    profile = tempfile.mkdtemp(prefix="kannada-chrome-profile-")
    server_thread.start()
    try:
        base_url = f"http://127.0.0.1:{port}"
        for _ in range(100):
            try:
                with urllib.request.urlopen(base_url, timeout=1) as response:
                    assert response.status == 200
                break
            except Exception:
                if not server_thread.is_alive():
                    raise AssertionError("In-process app server exited before startup") from None
                time.sleep(0.1)
        else:
            raise AssertionError("Timed out waiting for the in-process app server")
        asyncio.run(run_browser_checks(base_url, devtools_port, profile))
    finally:
        server.should_exit = True
        server_thread.join(timeout=10)
        shutil.rmtree(profile, ignore_errors=True)
        shutil.rmtree(DATA_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
