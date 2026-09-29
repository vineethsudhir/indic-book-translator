"""Download the (gated) Indic Parler-TTS checkpoint. Requires prior
authentication via `huggingface-cli login` or the HF_TOKEN env var —
run after requesting/receiving access at
https://huggingface.co/ai4bharat/indic-parler-tts
"""

from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent
TARGET_DIR = ROOT / "models" / "indic-parler-tts"

if __name__ == "__main__":
    import argparse

    # Parse before downloading so `--help` prints usage instead of fetching ~3.5 GB.
    argparse.ArgumentParser(description=__doc__).parse_args()

    path = snapshot_download(
        repo_id="ai4bharat/indic-parler-tts",
        local_dir=TARGET_DIR,
    )
    print(f"Downloaded to {path}")