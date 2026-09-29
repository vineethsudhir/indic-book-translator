"""Gradio UI for the desktop-ready Kannada translator app.

``build_app()`` is safe to call without launching (tests do), and the module
never imports the optional local-ML stack. The pipeline runs in-process via
:class:`kannada_epub.app.runner.BookRun`.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import time
from pathlib import Path

import gradio as gr

from ..config import ProviderConfig, QAConfig, TranslationModelConfig, TTSModelConfig
from ..pipeline import RunOptions
from .paths import books_dir, outputs_dir, settings_path
from .runner import BookRun
from .settings import (
    KNOWN_KEYS,
    AppSettings,
    load_settings,
    load_secrets_into_env,
    local_mode_status,
    save_secret,
    save_settings,
    secret_status,
)

_runner = BookRun()

_STATUS_COLORS = {
    "PASS": "#1a7f37",
    "RETRY": "#9a6700",
    "FLAGGED_FOR_REVIEW": "#cf222e",
}
_STATUS_BG = {
    "PASS": "#dafbe1",
    "RETRY": "#fff8c5",
    "FLAGGED_FOR_REVIEW": "#ffebe9",
}


# ---------------------------------------------------------------------------
# Pre-start checks
# ---------------------------------------------------------------------------
def _required_key_envs(settings: AppSettings, build_audiobook: bool) -> list[str]:
    envs: list[str] = []
    if settings.translation.provider == "openai_compatible":
        envs.append(settings.translation.api_key_env or "OPENAI_API_KEY")
    if settings.editor.provider == "anthropic":
        envs.append(settings.editor.api_key_env or "ANTHROPIC_API_KEY")
    elif settings.editor.provider == "openai_compatible":
        envs.append(settings.editor.api_key_env or "OPENAI_API_KEY")
    if build_audiobook:
        if settings.tts.provider == "sarvam":
            envs.append(settings.tts.api_key_env or "SARVAM_API_KEY")
        elif settings.tts.provider == "openai_compatible":
            envs.append(settings.tts.api_key_env or "OPENAI_API_KEY")
    if settings.qa.enabled and settings.qa.embedding == "openai_compatible":
        envs.append(settings.qa.embedding_api_key_env or "OPENAI_API_KEY")
    return _dedupe(envs)


def _required_local_modules(settings: AppSettings, build_audiobook: bool) -> list[str]:
    modules: list[str] = []
    if settings.translation.provider == "indictrans2_local":
        modules += ["ctranslate2", "sentencepiece", "IndicTransToolkit"]
    if build_audiobook and settings.tts.provider == "parler_local":
        modules += ["torch", "transformers", "parler_tts"]
    if settings.qa.enabled:
        if settings.qa.back_translation == "indictrans2_local":
            modules += ["ctranslate2", "sentencepiece", "IndicTransToolkit"]
        if settings.qa.embedding == "local_minilm":
            modules += ["torch", "transformers"]
    return _dedupe(modules)


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _missing_key_envs(settings: AppSettings, build_audiobook: bool) -> list[str]:
    return [e for e in _required_key_envs(settings, build_audiobook) if not os.environ.get(e)]


def _missing_local_modules(settings: AppSettings, build_audiobook: bool) -> list[str]:
    status = local_mode_status()
    return [m for m in _required_local_modules(settings, build_audiobook) if not status.get(m, False)]


# ---------------------------------------------------------------------------
# Translate tab
# ---------------------------------------------------------------------------
def _output_paths(result) -> tuple[str | None, str | None, str | None]:
    def existing(path):
        return str(path) if path is not None and Path(path).exists() else None

    return (
        existing(result.epub_path),
        existing(result.qa_report_path),
        existing(result.audiobook_path),
    )


def run_translation(epub_path, run_qa, build_audiobook, preview_n):
    """Generator: start a run and stream progress until it finishes.

    Yields ``(log, status, epub, qa_report, audiobook)``.
    """
    if not epub_path:
        yield "Please choose an EPUB file first.", "No file selected", None, None, None
        return
    if _runner.is_running():
        yield "A run is already in progress. Stop it first.", "already running", None, None, None
        return

    settings = load_settings()
    if bool(run_qa) != settings.qa.enabled:
        settings = settings.model_copy(
            update={"qa": settings.qa.model_copy(update={"enabled": bool(run_qa)})}
        )
    build_audio = bool(build_audiobook)

    load_secrets_into_env()
    missing_keys = _missing_key_envs(settings, build_audio)
    if missing_keys:
        message = (
            "Missing API key(s): " + ", ".join(missing_keys) + ". "
            "Open the Settings tab -> API keys, paste the key(s) and press Save keys."
        )
        yield message, "Not started", None, None, None
        return

    missing_local = _missing_local_modules(settings, build_audio)
    if missing_local:
        message = (
            "A local provider is selected but the required libraries are missing: "
            + ", ".join(missing_local)
            + ". See the Local mode tab for the one-time install steps."
        )
        yield message, "Not started", None, None, None
        return

    source = Path(epub_path)
    if not source.exists():
        yield f"File not found: {source}", "Not started", None, None, None
        return

    dest = books_dir() / source.name
    if source.resolve() != dest.resolve():
        shutil.copy(source, dest)

    max_paragraphs = int(preview_n) if preview_n else None
    if max_paragraphs == 0:
        max_paragraphs = None
    options = RunOptions(
        max_paragraphs=max_paragraphs,
        batch_size=settings.batch_size,
        build_audiobook=build_audio,
    )

    _runner.start(dest, settings, options)
    lines: list[str] = []
    while True:
        lines.extend(_runner.drain())
        running = _runner.is_running()
        if running:
            status = "running…"
        elif _runner.error:
            status = f"failed: {_runner.error}"
        elif _runner.result is not None and _runner.result.cancelled:
            status = "cancelled"
        else:
            status = "finished"
        yield "\n".join(lines), status, None, None, None
        if not running:
            break
        time.sleep(1)

    result = _runner.result
    if result is not None and not result.cancelled and _runner.error is None:
        epub_out, qa_out, audio_out = _output_paths(result)
        yield (
            "\n".join(lines),
            f"Finished. Outputs in {result.output_dir}",
            epub_out,
            qa_out,
            audio_out,
        )
    elif result is not None and result.cancelled:
        yield "\n".join(lines), "Cancelled — finished chapters are checkpointed.", None, None, None
    else:
        yield "\n".join(lines), f"Failed: {_runner.error}", None, None, None


def stop_translation() -> str:
    if _runner.is_running():
        _runner.cancel()
        return "Stop requested — finished chapters are kept; re-run resumes."
    return "No run in progress."


# ---------------------------------------------------------------------------
# Review tab
# ---------------------------------------------------------------------------
def _latest_output_dir() -> Path | None:
    root = outputs_dir()
    candidates = [
        path for path in root.iterdir() if path.is_dir() and (path / "chapters").is_dir()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path / "chapters").stat().st_mtime)


def list_chapters() -> list[str]:
    out = _latest_output_dir()
    if out is None:
        return []
    return sorted(path.stem for path in (out / "chapters").glob("*.json"))


def refresh_chapters():
    return gr.update(choices=list_chapters())


def _badge(status, score) -> str:
    if not status:
        return ""
    color = _STATUS_COLORS.get(status, "#57606a")
    background = _STATUS_BG.get(status, "#f6f8fa")
    score_text = f" {float(score):.2f}" if score is not None else ""
    return (
        f"<span style='background:{background};color:{color};border-radius:6px;"
        f"padding:2px 6px;font-size:0.8em'>{html.escape(str(status))}{score_text}</span>"
    )


def show_chapter(chapter_id, flagged_only) -> str:
    out = _latest_output_dir()
    if out is None or not chapter_id:
        return "<p>No translated book yet. Run a translation first.</p>"

    chapter_file = out / "chapters" / f"{chapter_id}.json"
    if not chapter_file.exists():
        return f"<p>Chapter {html.escape(str(chapter_id))} not found.</p>"
    data = json.loads(chapter_file.read_text(encoding="utf-8"))

    qa_file = out / "qa" / f"{chapter_id}.json"
    qa_results: list[dict] = []
    if qa_file.exists():
        qa_results = json.loads(qa_file.read_text(encoding="utf-8")).get("results", [])

    blocks = [f"<h3>{html.escape(data.get('chapter_title') or str(chapter_id))}</h3>"]
    position = 0
    shown = 0
    truncated = False
    for batch in data.get("batches", []):
        english = batch.get("source_english", [])
        kannada = batch.get("edited_kannada", [])
        for en, kn in zip(english, kannada):
            qa = qa_results[position] if position < len(qa_results) else {}
            status = qa.get("status")
            if flagged_only and status not in ("RETRY", "FLAGGED_FOR_REVIEW"):
                position += 1
                continue
            if shown >= 100:
                truncated = True
                break
            blocks.append(
                "<div style='border:1px solid #ddd;border-radius:8px;padding:8px;margin:8px 0'>"
                f"<div>{_badge(status, qa.get('similarity_score'))}</div>"
                f"<p><b>EN:</b> {html.escape(en)}</p>"
                f"<p><b>KN:</b> {html.escape(kn)}</p></div>"
            )
            shown += 1
            position += 1
        if truncated:
            break

    if truncated:
        blocks.append("<p><i>Showing the first 100 matching paragraphs.</i></p>")
    if shown == 0:
        blocks.append("<p><i>Nothing matches the current filter.</i></p>")
    return "\n".join(blocks)


# ---------------------------------------------------------------------------
# Settings tab
# ---------------------------------------------------------------------------
def _blank(value) -> str:
    if value is None:
        return ""
    return str(value)


def load_settings_form():
    s = load_settings()
    return (
        s.translation.provider,
        s.translation.model,
        _blank(s.translation.base_url),
        _blank(s.translation.api_key_env),
        s.editor.provider,
        s.editor.model,
        _blank(s.editor.base_url),
        _blank(s.editor.api_key_env),
        s.tts.provider,
        s.tts.model,
        s.tts.voice,
        _blank(s.tts.base_url),
        _blank(s.tts.api_key_env),
        bool(s.qa.enabled),
        s.qa.back_translation,
        s.qa.embedding,
        s.qa.embedding_model,
        _blank(s.qa.embedding_base_url),
        _blank(s.qa.embedding_api_key_env),
        s.qa.pass_threshold,
        s.qa.flag_threshold,
        s.batch_size,
        s.tone_register,
        ", ".join(s.exclude_ids),
    )


def _none_or_text(value) -> str | None:
    text = (value or "").strip()
    return text or None


def save_settings_form(*values):
    (
        t_provider,
        t_model,
        t_base_url,
        t_api_key_env,
        e_provider,
        e_model,
        e_base_url,
        e_api_key_env,
        tts_provider,
        tts_model,
        tts_voice,
        tts_base_url,
        tts_api_key_env,
        qa_enabled,
        qa_back,
        qa_embed,
        qa_embed_model,
        qa_embed_base,
        qa_embed_key,
        qa_pass,
        qa_flag,
        batch_size,
        tone_register,
        exclude_ids,
    ) = values
    try:
        settings = AppSettings(
            translation=TranslationModelConfig(
                provider=t_provider,
                model=t_model,
                base_url=_none_or_text(t_base_url),
                api_key_env=_none_or_text(t_api_key_env),
            ),
            editor=ProviderConfig(
                provider=e_provider,
                model=e_model,
                base_url=_none_or_text(e_base_url),
                api_key_env=_none_or_text(e_api_key_env),
            ),
            tts=TTSModelConfig(
                provider=tts_provider,
                model=tts_model,
                voice=tts_voice,
                base_url=_none_or_text(tts_base_url),
                api_key_env=_none_or_text(tts_api_key_env),
            ),
            qa=QAConfig(
                enabled=bool(qa_enabled),
                back_translation=qa_back,
                embedding=qa_embed,
                embedding_model=qa_embed_model,
                embedding_base_url=_none_or_text(qa_embed_base),
                embedding_api_key_env=_none_or_text(qa_embed_key),
                pass_threshold=float(qa_pass),
                flag_threshold=float(qa_flag),
            ),
            batch_size=int(batch_size),
            tone_register=tone_register,
            exclude_ids=[item.strip() for item in (exclude_ids or "").split(",") if item.strip()],
        )
    except Exception as exc:  # noqa: BLE001 - shown to the user
        return f"Could not save: {type(exc).__name__}: {exc}"
    save_settings(settings)
    return f"Saved to {settings_path()}"


def _key_status_md(name: str) -> str:
    return f"**{name}**: {'set' if secret_status().get(name) else 'not set'}"


def save_api_keys(*values):
    for name, value in zip(KNOWN_KEYS, values):
        text = (value or "").strip() if isinstance(value, str) else ""
        if text:
            save_secret(name, text)
    statuses = tuple(_key_status_md(name) for name in KNOWN_KEYS)
    return statuses + ("Keys saved. Values are never shown again.",)


# ---------------------------------------------------------------------------
# Local mode tab
# ---------------------------------------------------------------------------
def _local_mode_markdown(status: dict[str, bool] | None = None) -> str:
    status = status or local_mode_status()
    checklist = "\n".join(
        f"- {'✅' if installed else '❌'} `{name}`" for name, installed in status.items()
    )
    return f"""### Local (on-device) mode

By default this app translates through cloud APIs. Local translation, audiobook
and QA run the models on **this computer**: they need the optional Python
libraries below and the downloaded model files. They are installed from source:

```bash
git clone <the repository>
cd <the repository>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\\\Scripts\\\\activate
pip install -e ".[local,gui]"
python scripts/download_qa_models.py
python scripts/download_tts_model.py
python -m kannada_epub.app
```

**Optional libraries found here:**

{checklist}

If a local provider is selected in Settings but a library above is missing, the
Translate tab will stop and point you back here.
"""


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
def build_app() -> gr.Blocks:
    settings = load_settings()
    with gr.Blocks(title="English → Kannada Book Translator") as demo:
        gr.Markdown("# English → Kannada Book Translator")

        with gr.Tab("Translate"):
            gr.Markdown(
                "Pick an EPUB, then press **Translate**. The translated EPUB"
                " (and QA report / audiobook, if requested) appears below."
            )
            epub_file = gr.File(label="EPUB file", file_types=[".epub"], type="filepath")
            with gr.Row():
                run_qa = gr.Checkbox(label="Run QA check", value=bool(settings.qa.enabled))
                build_audio = gr.Checkbox(label="Build audiobook", value=False)
            preview_n = gr.Number(
                label="Preview: first N paragraphs per chapter (blank or 0 = whole book)",
                value=None,
                precision=0,
            )
            with gr.Row():
                translate_btn = gr.Button("Translate", variant="primary")
                stop_btn = gr.Button("Stop")
            run_log = gr.Code(label="Live log", language="shell")
            run_status = gr.Textbox(label="Status")
            with gr.Row():
                out_epub = gr.File(label="Translated EPUB")
                out_qa = gr.File(label="QA report")
                out_audio = gr.File(label="Audiobook")
            translate_btn.click(
                run_translation,
                inputs=[epub_file, run_qa, build_audio, preview_n],
                outputs=[run_log, run_status, out_epub, out_qa, out_audio],
            )
            stop_btn.click(stop_translation, outputs=run_status)

        with gr.Tab("Review"):
            gr.Markdown("Side-by-side English / Kannada for the most recent book.")
            with gr.Row():
                refresh_btn = gr.Button("Refresh chapter list")
                chapter_dd = gr.Dropdown(label="Chapter", choices=list_chapters())
            flagged_only = gr.Checkbox(label="Show only flagged/retry", value=False)
            review_view = gr.HTML(label="Side-by-side")
            refresh_btn.click(refresh_chapters, outputs=chapter_dd)
            chapter_dd.change(show_chapter, inputs=[chapter_dd, flagged_only], outputs=review_view)
            flagged_only.change(show_chapter, inputs=[chapter_dd, flagged_only], outputs=review_view)

        with gr.Tab("Settings"):
            with gr.Accordion("Translation", open=True):
                with gr.Row():
                    t_provider = gr.Dropdown(
                        ["openai_compatible", "indictrans2_local"],
                        label="Provider",
                        value=settings.translation.provider,
                    )
                    t_model = gr.Textbox(label="Model", value=settings.translation.model)
                with gr.Row():
                    t_base_url = gr.Textbox(
                        label="Base URL (cloud)", value=_blank(settings.translation.base_url)
                    )
                    t_key_env = gr.Textbox(
                        label="API key env var", value=_blank(settings.translation.api_key_env)
                    )
            with gr.Accordion("Consistency editor", open=False):
                with gr.Row():
                    e_provider = gr.Dropdown(
                        ["anthropic", "openai_compatible", "ollama"],
                        label="Provider",
                        value=settings.editor.provider,
                    )
                    e_model = gr.Textbox(label="Model", value=settings.editor.model)
                with gr.Row():
                    e_base_url = gr.Textbox(
                        label="Base URL", value=_blank(settings.editor.base_url)
                    )
                    e_key_env = gr.Textbox(
                        label="API key env var", value=_blank(settings.editor.api_key_env)
                    )
            with gr.Accordion("Audiobook (TTS)", open=False):
                with gr.Row():
                    tts_provider = gr.Dropdown(
                        ["sarvam", "openai_compatible", "parler_local"],
                        label="Provider",
                        value=settings.tts.provider,
                    )
                    tts_model = gr.Textbox(label="Model", value=settings.tts.model)
                    tts_voice = gr.Textbox(label="Voice", value=settings.tts.voice)
                with gr.Row():
                    tts_base_url = gr.Textbox(
                        label="Base URL", value=_blank(settings.tts.base_url)
                    )
                    tts_key_env = gr.Textbox(
                        label="API key env var", value=_blank(settings.tts.api_key_env)
                    )
            with gr.Accordion("QA check", open=False):
                qa_enabled = gr.Checkbox(label="Enabled", value=bool(settings.qa.enabled))
                with gr.Row():
                    qa_back = gr.Dropdown(
                        ["llm", "indictrans2_local"],
                        label="Back-translation",
                        value=settings.qa.back_translation,
                    )
                    qa_embed = gr.Dropdown(
                        ["openai_compatible", "local_minilm"],
                        label="Embedding",
                        value=settings.qa.embedding,
                    )
                with gr.Row():
                    qa_embed_model = gr.Textbox(
                        label="Embedding model", value=settings.qa.embedding_model
                    )
                    qa_embed_base = gr.Textbox(
                        label="Embedding base URL", value=_blank(settings.qa.embedding_base_url)
                    )
                    qa_embed_key = gr.Textbox(
                        label="Embedding key env var",
                        value=_blank(settings.qa.embedding_api_key_env),
                    )
                with gr.Row():
                    qa_pass = gr.Number(
                        label="Pass threshold", value=settings.qa.pass_threshold
                    )
                    qa_flag = gr.Number(
                        label="Flag threshold", value=settings.qa.flag_threshold
                    )
                gr.Markdown(
                    "Without local mode use back-translation **llm** and embedding"
                    " **openai_compatible** — e.g. Ollama at"
                    " `http://localhost:11434/v1` with model `nomic-embed-text`,"
                    " or OpenAI."
                )
            with gr.Accordion("Advanced", open=False):
                batch_size = gr.Number(label="Batch size", value=settings.batch_size, precision=0)
                tone_register = gr.Textbox(label="Register", value=settings.tone_register)
                exclude_ids = gr.Textbox(
                    label="Exclude spine ids (comma-separated)",
                    value=", ".join(settings.exclude_ids),
                )
            save_btn = gr.Button("Save settings", variant="primary")
            save_status = gr.Textbox(label="Settings status")
            form_inputs = [
                t_provider,
                t_model,
                t_base_url,
                t_key_env,
                e_provider,
                e_model,
                e_base_url,
                e_key_env,
                tts_provider,
                tts_model,
                tts_voice,
                tts_base_url,
                tts_key_env,
                qa_enabled,
                qa_back,
                qa_embed,
                qa_embed_model,
                qa_embed_base,
                qa_embed_key,
                qa_pass,
                qa_flag,
                batch_size,
                tone_register,
                exclude_ids,
            ]
            save_btn.click(save_settings_form, inputs=form_inputs, outputs=save_status)

            gr.Markdown("### API keys")
            gr.Markdown(
                "Keys are stored in a `secrets.env` file in your data directory"
                " (never in the repository). Values are never shown again."
            )
            key_inputs = []
            key_statuses = []
            status = secret_status()
            for name in KNOWN_KEYS:
                with gr.Row():
                    box = gr.Textbox(label=name, type="password", value="")
                    key_status = gr.Markdown(
                        f"**{name}**: {'set' if status.get(name) else 'not set'}"
                    )
                key_inputs.append(box)
                key_statuses.append(key_status)
            keys_summary = gr.Textbox(label="API keys status")
            save_keys_btn = gr.Button("Save keys")
            save_keys_btn.click(
                save_api_keys, inputs=key_inputs, outputs=key_statuses + [keys_summary]
            )

        with gr.Tab("Local mode"):
            gr.Markdown(_local_mode_markdown())

    return demo


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="English → Kannada Book Translator")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args(argv)
    build_app().launch(server_name=args.host, server_port=args.port, inbrowser=True)


if __name__ == "__main__":
    main()
