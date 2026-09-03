"""Single setup checker for a fresh clone.

Validates everything `scripts/translate_book.py` needs and tells you exactly
what to fix. One place for all credentials and model setup:

  1. `cp config/book.example.yaml config/book.yaml` (+ adjust paths)
  2. `cp config/consistency_editor.example.yaml config/consistency_editor.yaml`
     (+ pick a provider: local Ollama, or your own OpenAI-compatible /
     Anthropic API credits)
  3. `cp .env.example .env` (+ fill in keys if you chose a cloud provider)
  4. Download models into `models/` (see messages below)
  5. `python scripts/setup.py` until all checks pass, then run the book.

Usage:  python scripts/setup.py [--config config/book.yaml]
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

CHECKS_PASSED = 0
CHECKS_FAILED = 0


def ok(msg: str) -> None:
    global CHECKS_PASSED
    CHECKS_PASSED += 1
    print(f"  [ok] {msg}")


def fail(msg: str, hint: str = "") -> None:
    global CHECKS_FAILED
    CHECKS_FAILED += 1
    print(f"  [MISSING] {msg}")
    if hint:
        print(f"          -> {hint}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate translator setup.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    if sys.version_info < (3, 11):
        fail(f"Python {sys.version.split()[0]} < 3.11", "Install Python 3.11+.")
    else:
        ok(f"Python {sys.version.split()[0]}")

    try:
        import yaml  # noqa: F401
        from kannada_epub.config import load_book_config, load_provider_config
    except ImportError as e:
        fail(f"dependencies not installed ({e})", "Run: pip install -e .")
        print(f"\n{CHECKS_PASSED} passed, {CHECKS_FAILED} failed.")
        return 1

    if args.config:
        book_cfg_path = ROOT / args.config
    elif (ROOT / "config" / "book.yaml").exists():
        book_cfg_path = ROOT / "config" / "book.yaml"
    else:
        book_cfg_path = ROOT / "config" / "book.example.yaml"
    try:
        book = load_book_config(book_cfg_path)
        try:
            shown = str(book_cfg_path.relative_to(ROOT))
        except ValueError:
            shown = str(book_cfg_path)
        ok(f"book config: {shown}")
    except Exception as e:
        fail(f"book config unreadable: {e}", "Copy config/book.example.yaml to config/book.yaml.")
        print(f"\n{CHECKS_PASSED} passed, {CHECKS_FAILED} failed.")
        return 1

    def resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else ROOT / path

    if resolve(book.epub_path).exists():
        ok(f"EPUB: {book.epub_path}")
    else:
        fail(f"EPUB not found: {book.epub_path}", "Point book.epub_path at your .epub file.")

    t = book.translation
    if t.provider == "indictrans2_local":
        ct2 = resolve(book.ct2_model_dir)
        vocab_src = ct2 / "vocab" / "model.SRC"
        vocab_tgt = ct2 / "vocab" / "model.TGT"
        model_bin = ct2 / "model.bin"
        if model_bin.exists() and vocab_src.exists() and vocab_tgt.exists():
            ok(f"IndicTrans2 CT2 model: {book.ct2_model_dir}")
        else:
            fail(
                f"translation model incomplete at {book.ct2_model_dir}",
                "Download a CTranslate2 conversion of ai4bharat/indictrans2-en-indic-1B "
                "so the dir contains model.bin and vocab/model.SRC + vocab/model.TGT, "
                "or switch book.translation.provider to openai_compatible for cloud.",
            )
    elif t.provider == "openai_compatible":
        if not t.base_url:
            fail("translation needs base_url", "Set book.translation.base_url.")
        elif os.environ.get(t.api_key_env or "OPENAI_API_KEY"):
            ok(f"Cloud translation ready: {t.model} @ {t.base_url} (bring your own credits)")
        else:
            fail(
                f"{t.api_key_env or 'OPENAI_API_KEY'} not set",
                "Add it to .env — no local translation model needed on this machine.",
            )

    tts_cfg = book.tts
    if tts_cfg.provider == "parler_local":
        tts_dir = ROOT / "models" / "indic-parler-tts"
        if any(tts_dir.glob("*.safetensors")) or (tts_dir / "config.json").exists():
            ok("TTS model: models/indic-parler-tts (audiobook support)")
        else:
            fail(
                "TTS model not found (only needed for audiobooks)",
                "Request access at https://huggingface.co/ai4bharat/indic-parler-tts, "
                "run `huggingface-cli login`, then: python scripts/download_tts_model.py — "
                "or switch book.tts.provider to sarvam for cloud.",
            )
    elif tts_cfg.provider == "sarvam":
        if os.environ.get(tts_cfg.api_key_env or "SARVAM_API_KEY"):
            ok(f"Cloud TTS ready: {tts_cfg.model} / {tts_cfg.voice} (bring your own credits)")
        else:
            fail(
                f"{tts_cfg.api_key_env or 'SARVAM_API_KEY'} not set",
                "Sign up at https://dashboard.sarvam.ai, add the key to .env — "
                "no local TTS model needed on this machine.",
            )
    elif tts_cfg.provider == "openai_compatible":
        if not tts_cfg.base_url:
            fail("TTS needs base_url", "Set book.tts.base_url.")
        elif os.environ.get(tts_cfg.api_key_env or "OPENAI_API_KEY"):
            ok(f"Cloud TTS ready: {tts_cfg.model} @ {tts_cfg.base_url}")
        else:
            fail(
                f"{tts_cfg.api_key_env or 'OPENAI_API_KEY'} not set",
                "Add it to .env — no local TTS model needed on this machine.",
            )

    try:
        provider = load_provider_config(resolve(book.provider_config))
        ok(f"provider config: {provider.provider} / {provider.model}")
    except Exception as e:
        fail(
            f"provider config unreadable: {e}",
            "Copy config/consistency_editor.example.yaml to config/consistency_editor.yaml "
            "and point book.provider_config at it.",
        )
        print(f"\n{CHECKS_PASSED} passed, {CHECKS_FAILED} failed.")
        return 1

    if provider.provider == "ollama":
        import httpx

        base = (provider.base_url or "http://localhost:11434").rstrip("/")
        try:
            tags = httpx.get(f"{base}/api/tags", timeout=10).json().get("models", [])
            names = [m.get("name", "") for m in tags]
            if any(provider.model in n for n in names):
                ok(f"Ollama model ready: {provider.model}")
            else:
                fail(
                    f"Ollama up, but model {provider.model!r} not pulled",
                    f"Run: ollama pull {provider.model}",
                )
        except Exception:
            fail(
                f"Ollama not reachable at {base}",
                "Install from https://ollama.com and run `ollama serve`.",
            )
    elif provider.provider == "openai_compatible":
        if not provider.base_url:
            fail("openai_compatible needs base_url", "Set base_url in the provider config.")
        env_var = provider.api_key_env or "OPENAI_API_KEY"
        if os.environ.get(env_var):
            ok(f"API key present: {env_var} (bring your own credits)")
        else:
            fail(f"{env_var} not set", f"Add it to .env (copied from .env.example), e.g. {env_var}=sk-...")
    elif provider.provider == "anthropic":
        env_var = provider.api_key_env or "ANTHROPIC_API_KEY"
        if os.environ.get(env_var):
            ok(f"API key present: {env_var} (bring your own credits)")
        else:
            fail(f"{env_var} not set", f"Add it to .env (copied from .env.example), e.g. {env_var}=sk-ant-...")

    print(f"\n{CHECKS_PASSED} passed, {CHECKS_FAILED} failed.")
    if CHECKS_FAILED:
        print("Fix the items above, then re-run setup. Run the book with:")
        print("  python scripts/translate_book.py")
        return 1
    print("Setup complete. Run the book with:")
    print("  python scripts/translate_book.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
