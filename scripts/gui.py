"""Minimal Gradio GUI over the existing CLI pipeline.

One code path: every tab shells out to the same scripts a terminal user
runs (setup.py, translate_book.py) and reads the same artifacts
(book.yaml, manifest.json, checkpoints/, chapters/*.json). No pipeline
logic lives here.

Usage:
  pip install -e .[gui]
  python scripts/gui.py [--port 7860] [--share]
"""

import argparse
import html
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

ROOT = Path(__file__).resolve().parent.parent
BOOK_DEFAULT = ROOT / "config" / "book.yaml"
BOOK_EXAMPLE = ROOT / "config" / "book.example.yaml"

import gradio as gr  # noqa: E402


def _book_config_path() -> Path:
    return BOOK_DEFAULT if BOOK_DEFAULT.exists() else BOOK_EXAMPLE


def _run_capture(cmd: list[str]) -> tuple[str, int]:
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    return (proc.stdout + proc.stderr), proc.returncode


# --- Setup tab -------------------------------------------------------------

def setup_check(config_path: str) -> tuple[str, str]:
    out, code = _run_capture(
        [sys.executable, "scripts/setup.py", "--config", config_path or str(_book_config_path())]
    )
    status = "PASS — ready to translate." if code == 0 else "FAIL — fix the [MISSING] items above, then re-check."
    return out, status


# --- Configure tab ---------------------------------------------------------

FIELD_ORDER = [
    "epub_path", "output_dir", "glossary_db", "provider_config",
    "ct2_model_dir", "batch_size", "tone_register", "exclude_ids",
]


def load_config_for_form(config_path: str) -> tuple:
    import yaml

    raw = yaml.safe_load(Path(config_path or str(_book_config_path())).read_text())["book"]
    t = raw.get("translation", {}) or {}
    tts = raw.get("tts", {}) or {}
    return (
        raw.get("epub_path", ""),
        raw.get("output_dir", ""),
        raw.get("glossary_db", ""),
        t.get("provider", "indictrans2_local"),
        t.get("model", "sarvam-m"),
        t.get("base_url", "") or "",
        t.get("api_key_env", "") or "",
        tts.get("provider", "parler_local"),
        tts.get("model", "bulbul:v3"),
        tts.get("voice", "anushka"),
        tts.get("base_url", "") or "",
        tts.get("api_key_env", "") or "",
        str(raw.get("batch_size", 20)),
        (raw.get("tone_register", "")),
        ", ".join(raw.get("exclude_ids", [])),
        "Loaded. Edit, then Save.",
    )


def save_config(
    config_path: str, epub_path: str, output_dir: str, glossary_db: str,
    t_provider: str, t_model: str, t_base_url: str, t_api_key_env: str,
    tts_provider: str, tts_model: str, tts_voice: str, tts_base_url: str,
    tts_api_key_env: str, batch_size: str, tone_register: str, exclude_ids: str,
) -> str:
    import yaml

    path = Path(config_path or str(BOOK_DEFAULT))
    raw: dict = {"book": {}}
    if path.exists():
        raw = yaml.safe_load(path.read_text()) or {"book": {}}
    book = raw.setdefault("book", {})
    book.update({
        "epub_path": epub_path.strip(),
        "output_dir": output_dir.strip(),
        "glossary_db": glossary_db.strip(),
        "batch_size": int(batch_size),
        "tone_register": tone_register.strip(),
        "exclude_ids": [e.strip() for e in exclude_ids.split(",") if e.strip()],
        "translation": {
            "provider": t_provider,
            "model": t_model.strip(),
            "base_url": t_base_url.strip() or None,
            "api_key_env": t_api_key_env.strip() or None,
        },
        "tts": {
            "provider": tts_provider,
            "model": tts_model.strip(),
            "voice": tts_voice.strip(),
            "base_url": tts_base_url.strip() or None,
            "api_key_env": tts_api_key_env.strip() or None,
        },
    })
    path.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True))
    return f"Saved to {path}. API keys stay in .env, never here."


# --- Run tab ---------------------------------------------------------------

_runner_proc: subprocess.Popen | None = None


def _drain(pipe, q: queue.Queue) -> None:
    for line in iter(pipe.readline, ""):
        q.put(line)
    pipe.close()


def start_run(config_path: str, limit_chapters: str, max_paragraphs: str):
    global _runner_proc
    if _runner_proc is not None and _runner_proc.poll() is None:
        yield "A run is already in progress. Stop it first.", "already running"
        return
    cmd = [sys.executable, "scripts/translate_book.py", "--config",
           config_path or str(_book_config_path())]
    if limit_chapters.strip():
        cmd += ["--limit-chapters"] + limit_chapters.split()
    if max_paragraphs.strip():
        cmd += ["--max-paragraphs", max_paragraphs.strip()]
    _runner_proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, cwd=ROOT,
    )
    q: queue.Queue = queue.Queue()
    threading.Thread(target=_drain, args=(_runner_proc.stdout, q), daemon=True).start()
    log_lines: list[str] = []
    while True:
        try:
            while True:
                log_lines.append(q.get_nowait())
        except queue.Empty:
            pass
        done = _runner_proc.poll() is not None and q.empty()
        yield "".join(log_lines), ("finished (exit %d)" % _runner_proc.returncode if done else "running…")
        if done:
            _runner_proc = None
            return
        time.sleep(1)


def stop_run() -> str:
    global _runner_proc
    if _runner_proc is not None and _runner_proc.poll() is None:
        _runner_proc.terminate()
        return "Stop requested — finished chapters are kept as checkpoints; re-start resumes."
    return "No run in progress."


def run_progress(config_path: str) -> str:
    import yaml

    try:
        raw = yaml.safe_load(Path(config_path or str(_book_config_path())).read_text())["book"]
        out = _output_dir(raw.get("output_dir", "data/book_output"))
    except Exception:
        out = _output_dir("data/book_output")
    n_checkpoints = len(list((out / "checkpoints").glob("*.json"))) if (out / "checkpoints").exists() else 0
    manifest = out / "manifest.json"
    chapters = "?"
    if manifest.exists():
        try:
            chapters = str(len(json.loads(manifest.read_text()).get("chapters", [])))
        except Exception:
            pass
    return f"Batches checkpointed: {n_checkpoints} | chapters in manifest: {chapters}"


# --- Review tab ------------------------------------------------------------

def _output_dir(path: str) -> Path:
    out = Path(path or "data/book_output")
    return out if out.is_absolute() else ROOT / out


def list_chapters(output_dir: str) -> list[str]:
    d = _output_dir(output_dir) / "chapters"
    return sorted(p.stem for p in d.glob("*.json")) if d.exists() else []


def refresh_chapters(output_dir: str):
    return gr.Dropdown(choices=list_chapters(output_dir))


def show_chapter(output_dir: str, chapter_id: str) -> tuple[str, str | None]:
    if not chapter_id:
        return "Pick a chapter.", None
    data = json.loads((_output_dir(output_dir) / "chapters" / f"{chapter_id}.json").read_text())
    blocks = [f"<h3>{html.escape(data.get('chapter_title') or chapter_id)}</h3>"]
    shown = 0
    for b in data["batches"]:
        for en, kn, emo in zip(b["source_english"], b["edited_kannada"], b["edited_emotions"]):
            if shown >= 50:
                blocks.append("<p><i>…truncated to 50 paragraphs in the GUI; see the JSON for the rest.</i></p>")
                break
            blocks.append(
                f"<div style='border:1px solid #ccc;border-radius:8px;padding:8px;margin:8px 0'>"
                f"<div style='font-size:0.85em;color:#666'>[{html.escape(emo)}]</div>"
                f"<p><b>EN:</b> {html.escape(en)}</p>"
                f"<p><b>KN:</b> {html.escape(kn)}</p></div>"
            )
            shown += 1
    wavs = sorted((_output_dir(output_dir)).glob("*.wav"))
    audio = str(wavs[0]) if wavs else None
    return "\n".join(blocks), audio


# --- App -------------------------------------------------------------------

def build_app() -> gr.Blocks:
    with gr.Blocks(title="EN → KN Book Translator") as demo:
        gr.Markdown("# English → Kannada Book Translator")
        with gr.Tab("Setup"):
            cfg_setup = gr.Textbox(value=str(_book_config_path()), label="Book config")
            check_btn = gr.Button("Run setup checks")
            setup_log = gr.Code(label="setup.py output", language="shell")
            setup_status = gr.Textbox(label="Status")
            check_btn.click(setup_check, inputs=cfg_setup, outputs=[setup_log, setup_status])
        with gr.Tab("Configure"):
            cfg_path = gr.Textbox(value=str(BOOK_DEFAULT), label="Save to (book.yaml)")
            load_btn = gr.Button("Load current values")
            epub_path = gr.Textbox(label="EPUB path")
            output_dir = gr.Textbox(label="Output dir")
            glossary_db = gr.Textbox(label="Glossary DB")
            with gr.Row():
                t_provider = gr.Dropdown(["indictrans2_local", "openai_compatible"], label="Translation provider")
                t_model = gr.Textbox(label="Translation model")
                t_base_url = gr.Textbox(label="Translation base_url (cloud)")
                t_api_key_env = gr.Textbox(label="Translation key env (cloud)")
            with gr.Row():
                tts_provider = gr.Dropdown(["parler_local", "sarvam", "openai_compatible"], label="TTS provider")
                tts_model = gr.Textbox(label="TTS model")
                tts_voice = gr.Textbox(label="TTS voice")
                tts_base_url = gr.Textbox(label="TTS base_url (cloud)")
                tts_api_key_env = gr.Textbox(label="TTS key env (cloud)")
            batch_size = gr.Textbox(label="Batch size")
            tone_register = gr.Textbox(label="Register")
            exclude_ids = gr.Textbox(label="Exclude spine ids (comma-separated)")
            save_btn = gr.Button("Save book.yaml")
            save_status = gr.Textbox(label="Status")
            form_inputs = [epub_path, output_dir, glossary_db, t_provider, t_model,
                           t_base_url, t_api_key_env, tts_provider, tts_model,
                           tts_voice, tts_base_url, tts_api_key_env, batch_size,
                           tone_register, exclude_ids]
            load_btn.click(load_config_for_form, inputs=cfg_path,
                           outputs=form_inputs + [save_status])
            save_btn.click(save_config, inputs=[cfg_path] + form_inputs, outputs=save_status)
        with gr.Tab("Run"):
            cfg_run = gr.Textbox(value=str(_book_config_path()), label="Book config")
            with gr.Row():
                limit_ch = gr.Textbox(label="Limit chapters (space-separated, blank = all)")
                max_para = gr.Textbox(label="Max paragraphs per chapter (blank = all)")
            with gr.Row():
                start_btn = gr.Button("Start", variant="primary")
                stop_btn = gr.Button("Stop")
            run_log = gr.Code(label="Live log", language="shell")
            run_state = gr.Textbox(label="State")
            prog_box = gr.Textbox(label="Progress")
            prog_btn = gr.Button("Refresh progress")
            start_btn.click(start_run, inputs=[cfg_run, limit_ch, max_para],
                            outputs=[run_log, run_state])
            stop_btn.click(stop_run, outputs=run_state)
            prog_btn.click(run_progress, inputs=cfg_run, outputs=prog_box)
        with gr.Tab("Review"):
            out_dir = gr.Textbox(value="data/book_output", label="Output dir")
            with gr.Row():
                refresh_btn = gr.Button("Refresh chapter list")
                chapter_dd = gr.Dropdown(label="Chapter")
            html_view = gr.HTML(label="Side-by-side")
            audio_view = gr.Audio(label="Audiobook WAV (if present)")
            refresh_btn.click(refresh_chapters, inputs=out_dir, outputs=chapter_dd)
            chapter_dd.change(show_chapter, inputs=[out_dir, chapter_dd],
                              outputs=[html_view, audio_view])
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    build_app().launch(server_name=args.host, server_port=args.port, share=args.share)
