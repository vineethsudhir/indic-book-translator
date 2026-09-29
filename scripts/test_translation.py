"""Smoke test: translate a handful of English sentences to Kannada with the
CTranslate2 IndicTrans2 engine and print the output for manual inspection."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kannada_epub.translation import IndicTrans2Engine

MODEL_DIR = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "en-indic-1b-ct2"
    / "en-indic-1b-ct2"
    / "ctranslate2_model"
)

SAMPLE_SENTENCES = [
    "The thread scheduler preempts running processes.",
    "Meera opened her laptop and stared at the kernel logs.",
    "It is the most distinguished thing about him, said Sherlock Holmes.",
    "Good morning, how are you today?",
]

if __name__ == "__main__":
    engine = IndicTrans2Engine(
        ct2_model_dir=MODEL_DIR,
        spm_src_path=MODEL_DIR / "vocab" / "model.SRC",
        spm_tgt_path=MODEL_DIR / "vocab" / "model.TGT",
        device="cpu",
        compute_type="int8",
    )

    translations = engine.translate(SAMPLE_SENTENCES, src_lang="eng_Latn", tgt_lang="kan_Knda")

    for src, tgt in zip(SAMPLE_SENTENCES, translations, strict=True):
        print(f"EN: {src}")
        print(f"KN: {tgt}")
        print()