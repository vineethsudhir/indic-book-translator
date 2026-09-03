"""Smoke test: synthesize a short Kannada utterance with Indic Parler-TTS and
save it to a wav file for manual listening."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "indic-parler-tts"
OUTPUT_PATH = ROOT / "data" / "tts_sample.wav"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Reusing an actual line from the earlier IndicTrans2 test output.
PROMPT = "ಥ್ರೆಡ್ ಶೆಡ್ಯೂಲರ್ ಚಾಲನೆಯಲ್ಲಿರುವ ಪ್ರಕ್ರಿಯೆಗಳನ್ನು ಪೂರ್ವಭಾವಿಯಾಗಿ ತಡೆಯುತ್ತದೆ."
DESCRIPTION = (
    "Suresh speaks in a clear narration tone at a moderate pace, "
    "close-sounding and high quality, with no background noise."
)

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    model = ParlerTTSForConditionalGeneration.from_pretrained(str(MODEL_DIR)).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

    desc_ids = description_tokenizer(DESCRIPTION, return_tensors="pt").to(DEVICE)
    prompt_ids = tokenizer(PROMPT, return_tensors="pt").to(DEVICE)

    print("Generating...")
    generation = model.generate(
        input_ids=desc_ids.input_ids,
        attention_mask=desc_ids.attention_mask,
        prompt_input_ids=prompt_ids.input_ids,
        prompt_attention_mask=prompt_ids.attention_mask,
    )
    audio_arr = generation.to(torch.float32).cpu().numpy().squeeze()
    sf.write(str(OUTPUT_PATH), audio_arr, model.config.sampling_rate)

    duration_s = len(audio_arr) / model.config.sampling_rate
    print(f"Wrote {OUTPUT_PATH} — {duration_s:.2f}s at {model.config.sampling_rate} Hz")