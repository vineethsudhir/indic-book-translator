"""Smoke test: run the configured provider against a tiny sample edit."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kannada_epub.config import load_provider_config
from kannada_epub.consistency_editor import ConsistencyEditor
from kannada_epub.providers.factory import build_provider

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "consistency_editor.example.yaml"

DRAFT = (
    "ಥ್ರೆಡ್ ಶೆಡ್ಯೂಲರ್ ಚಾಲನೆಯಲ್ಲಿರುವ ಪ್ರಕ್ರಿಯೆಗಳನ್ನು ತಡೆಯುತ್ತದೆ. "
    "ಅವಳು ಅದನ್ನು ಮತ್ತೆ ಪ್ರಾರಂಭಿಸಿದಳು."
)
GLOSSARY = {"thread scheduler": "ಥ್ರೆಡ್ ಶೆಡ್ಯೂಲರ್", "process": "ಪ್ರಕ್ರಿಯೆ"}

if __name__ == "__main__":
    config = load_provider_config(CONFIG_PATH)
    print(f"Provider: {config.provider}  Model: {config.model}")

    provider = build_provider(config)
    editor = ConsistencyEditor(provider)

    result = editor.edit_chapter(
        draft_kannada_text=DRAFT,
        glossary=GLOSSARY,
        prior_chapter_context="Chapter 2 introduced Meera, a systems engineer debugging a kernel scheduler.",
        register="neutral, technical, third-person narration",
    )

    print("\n--- Edited chapter ---")
    for p in result:
        print(f"[{p.emotion}] {p.text}")