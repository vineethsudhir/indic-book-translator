"""Download the QA models used by FR-4 (back-translation + embeddings):

1. The CTranslate2 Kannada->English IndicTrans2 model that
   `LocalBackTranslator` wraps.
2. The `all-MiniLM-L6-v2` tokenizer + model that `LocalMiniLMEmbedder`
   loads (pre-cached so the first QA run works offline).

The embedding text encoder is fetched through `transformers`, not
`sentence-transformers`, so no extra dependency is needed.

Usage:
  python scripts/download_qa_models.py
"""

from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE

ROOT = Path(__file__).resolve().parent.parent

INDIC_EN_REPO = "adalat-ai/ct2-rotary-indictrans2-indic-en-dist-200M"
INDIC_EN_DIR = ROOT / "models" / "indic-en-200m-ct2"

MINILM_REPO = "sentence-transformers/all-MiniLM-L6-v2"

if __name__ == "__main__":
    import argparse

    # Parse before downloading so `--help` prints usage instead of fetching ~1 GB.
    argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    ).parse_args()

    indic_path = snapshot_download(
        repo_id=INDIC_EN_REPO,
        local_dir=INDIC_EN_DIR,
    )
    print(f"IndicTrans2 Kannada->English model ({INDIC_EN_REPO}) downloaded to: {indic_path}")

    # Imported here so the script's other half works even without transformers.
    from transformers import AutoModel, AutoTokenizer

    AutoTokenizer.from_pretrained(MINILM_REPO)
    AutoModel.from_pretrained(MINILM_REPO)
    print(f"Embedding model ({MINILM_REPO}) cached under: {HF_HUB_CACHE}")
