"""Smoke test for the desktop-ready app package (``kannada_epub.app``).

Exercises settings/secrets persistence, the in-process background runner
(real pipeline, FAKE engines, real Sherlock EPUB), the Gradio app
construction, and a cloud-only import of the UI with the local ML stack
blocked. No models, no network.

Run: .venv/bin/python scripts/test_app.py
"""

import os
import stat
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(tempfile.mkdtemp(prefix="kannada-app-test-"))
# Must be set before the app package reads its paths.
os.environ["KANNADA_APP_DATA_DIR"] = str(DATA_DIR)

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import test_pipeline as tp  # noqa: E402  (reused fake engines)

from kannada_epub.app.paths import secrets_path, settings_path  # noqa: E402
from kannada_epub.app.runner import BookRun  # noqa: E402
from kannada_epub.app.settings import (  # noqa: E402
    KNOWN_KEYS,
    AppSettings,
    book_config_for,
    load_secrets_into_env,
    load_settings,
    local_mode_status,
    save_secret,
    save_settings,
    secret_status,
)
from kannada_epub.config import BookConfig, load_provider_config  # noqa: E402
from kannada_epub.consistency_editor import ConsistencyEditor  # noqa: E402
from kannada_epub.glossary import GlossaryStore  # noqa: E402
from kannada_epub.pipeline import PipelineComponents, RunOptions  # noqa: E402

EPUB = ROOT / "data" / "sherlock_holmes.epub"


def assert_raises(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type as exc:
        return exc
    raise AssertionError(f"expected {exc_type.__name__}, but no exception was raised")


def wait_until(predicate, timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def make_components(glossary_name: str) -> PipelineComponents:
    """Fake translation + editor, QA disabled (no model/network needed)."""
    return PipelineComponents(
        translation_engine=tp.FakeTranslationEngine(),
        editor=ConsistencyEditor(tp.FakeEditorProvider()),
        glossary_store=GlossaryStore(DATA_DIR / glossary_name),
    )


class SlowFakeTranslationEngine(tp.FakeTranslationEngine):
    """Blocks inside translation so the run thread stays alive on demand."""

    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def translate_paragraphs(self, paragraphs, src_lang, tgt_lang):
        self.entered.set()
        if not self.release.wait(timeout=30):
            raise RuntimeError("slow fake engine was never released")
        return super().translate_paragraphs(paragraphs, src_lang, tgt_lang)


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------
def test_settings() -> None:
    assert not settings_path().exists()
    defaults = load_settings()
    assert isinstance(defaults, AppSettings)
    assert defaults.translation.provider == "openai_compatible"
    assert defaults.translation.model == "sarvam-m"
    assert defaults.translation.base_url == "https://api.sarvam.ai/v1"
    assert defaults.translation.api_key_env == "SARVAM_API_KEY"
    assert defaults.editor.provider == "anthropic"
    assert defaults.editor.model == "claude-haiku-4-5"
    assert defaults.editor.api_key_env == "ANTHROPIC_API_KEY"
    assert defaults.tts.provider == "sarvam"
    assert defaults.tts.model == "bulbul:v3"
    assert defaults.tts.voice == "anushka"
    assert defaults.tts.api_key_env == "SARVAM_API_KEY"
    assert defaults.qa.enabled is False
    assert defaults.batch_size == 20
    assert defaults.tone_register == "neutral, standard written Kannada"
    assert defaults.exclude_ids == ["pg-header", "pg-footer", "coverpage-wrapper"]

    changed = defaults.model_copy(update={"batch_size": 7, "tone_register": "formal Kannada"})
    save_settings(changed)
    assert settings_path().exists()
    reloaded = load_settings()
    assert reloaded.batch_size == 7
    assert reloaded.tone_register == "formal Kannada"
    assert reloaded.translation == defaults.translation

    cfg, out_dir = book_config_for(EPUB, reloaded)
    assert isinstance(cfg, BookConfig)
    assert cfg.epub_path == str(EPUB)
    assert Path(cfg.output_dir) == out_dir
    assert out_dir.name == "sherlock_holmes"
    assert out_dir.exists()
    assert cfg.batch_size == 7
    assert cfg.translation == reloaded.translation

    provider_cfg = load_provider_config(cfg.provider_config)
    assert provider_cfg.provider == reloaded.editor.provider
    assert provider_cfg.model == reloaded.editor.model
    assert provider_cfg.api_key_env == reloaded.editor.api_key_env


# ---------------------------------------------------------------------------
# secrets
# ---------------------------------------------------------------------------
def test_secrets() -> None:
    for key in KNOWN_KEYS:
        os.environ.pop(key, None)

    save_secret("SARVAM_API_KEY", "sarvam-value")
    assert secrets_path().exists()
    assert stat.S_IMODE(os.stat(secrets_path()).st_mode) == 0o600
    assert secret_status()["SARVAM_API_KEY"] is True
    assert "sarvam-value" in secrets_path().read_text(encoding="utf-8")

    save_secret("SARVAM_API_KEY", "sarvam-value-2")
    text = secrets_path().read_text(encoding="utf-8")
    assert "sarvam-value-2" in text and "sarvam-value\n" not in text

    assert_raises(ValueError, save_secret, "bad-name", "x")
    assert_raises(ValueError, save_secret, "lowercase", "x")
    assert_raises(ValueError, save_secret, "OPENAI_API_KEY", "a\nb")

    save_secret("SARVAM_API_KEY", "")
    assert secret_status()["SARVAM_API_KEY"] is False
    assert "SARVAM_API_KEY" not in secrets_path().read_text(encoding="utf-8")

    os.environ["ANTHROPIC_API_KEY"] = "already-set"
    save_secret("ANTHROPIC_API_KEY", "from-file")
    save_secret("SARVAM_API_KEY", "from-file-sarvam")
    load_secrets_into_env()
    assert os.environ["ANTHROPIC_API_KEY"] == "already-set"
    assert os.environ["SARVAM_API_KEY"] == "from-file-sarvam"


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------
def test_runner(settings: AppSettings) -> None:
    # A normal run finishes, streams lines and writes a translated EPUB.
    run = BookRun()
    run.start(
        EPUB,
        settings,
        RunOptions(limit_chapters=["item4"], max_paragraphs=3, batch_size=3),
        components=make_components("run-glossary.db"),
    )
    assert wait_until(lambda: not run.is_running()), "run did not finish"
    lines = run.drain()
    assert lines, "expected progress lines from drain()"
    assert run.error is None, run.error
    result = run.result
    assert result is not None
    assert result.cancelled is False
    assert result.epub_path is not None and Path(result.epub_path).exists()
    assert Path(result.epub_path).name == "sherlock_holmes.kn.epub"

    # A second start while running is refused.
    slow = SlowFakeTranslationEngine()
    slow_components = PipelineComponents(
        translation_engine=slow,
        editor=ConsistencyEditor(tp.FakeEditorProvider()),
        glossary_store=GlossaryStore(DATA_DIR / "slow-glossary.db"),
    )
    run2 = BookRun()
    run2.start(
        EPUB,
        settings,
        RunOptions(limit_chapters=["item5"], max_paragraphs=3, batch_size=3),
        components=slow_components,
    )
    assert slow.entered.wait(timeout=30), "slow fake engine never started"
    assert run2.is_running()
    assert_raises(
        RuntimeError,
        run2.start,
        EPUB,
        settings,
        RunOptions(limit_chapters=["item5"]),
        slow_components,
    )
    slow.release.set()
    assert wait_until(lambda: not run2.is_running()), "slow run did not finish"
    assert run2.error is None, run2.error

    # Cancelling before processing yields a cancelled result with no EPUB.
    run3 = BookRun()
    run3.cancel()
    run3.start(
        EPUB,
        settings,
        RunOptions(limit_chapters=["item6"], max_paragraphs=3, batch_size=3),
        components=make_components("cancel-glossary.db"),
    )
    assert wait_until(lambda: not run3.is_running()), "cancelled run did not finish"
    assert run3.error is None, run3.error
    cancelled = run3.result
    assert cancelled is not None
    assert cancelled.cancelled is True
    assert cancelled.epub_path is None


# ---------------------------------------------------------------------------
# Gradio construction + local-mode check
# ---------------------------------------------------------------------------
def test_build_app() -> None:
    import gradio as gr

    from kannada_epub.app.ui import build_app

    app = build_app()
    assert isinstance(app, gr.Blocks)
    assert set(local_mode_status()).issuperset(
        {"torch", "transformers", "ctranslate2", "sentencepiece", "IndicTransToolkit", "parler_tts"}
    )


def check_cloud_only_ui() -> None:
    """Import the UI and build the app with the local ML stack blocked.

    ``gradio`` itself imports ``huggingface_hub`` (a GUI dependency, pinned in
    the ``gui`` extra), so it is imported before the blocker is installed; the
    blocker still stops any *new* local-ML import and any later
    ``huggingface_hub`` submodule pull.
    """
    src = str(ROOT / "src")
    code = r'''
import sys
sys.path.insert(0, "__SRC__")

import gradio  # noqa: F401  (needs huggingface_hub; GUI extra installs it)

BLOCKED = {"torch", "transformers", "ctranslate2", "sentencepiece",
           "IndicTransToolkit", "parler_tts", "huggingface_hub"}


class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError("blocked local ML dependency: " + name)
        return None


sys.meta_path.insert(0, Blocker())

import kannada_epub.app.ui as ui
import gradio as gr

app = ui.build_app()
assert isinstance(app, gr.Blocks)
print("app-cloud-only-import-ok")
'''.replace("__SRC__", src)

    env = dict(os.environ)
    env["KANNADA_APP_DATA_DIR"] = str(DATA_DIR)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env=env,
    )
    assert proc.returncode == 0, (
        f"subprocess failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    assert "app-cloud-only-import-ok" in proc.stdout


def main() -> None:
    try:
        test_settings()
        test_secrets()
        test_runner(load_settings())
        test_build_app()
        check_cloud_only_ui()
    finally:
        import shutil

        shutil.rmtree(DATA_DIR, ignore_errors=True)

    print("test_app: all assertions passed")


if __name__ == "__main__":
    main()
