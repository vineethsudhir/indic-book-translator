"""Generate the same Kannada sentence across all four available Kannada
voices for direct comparison."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "indic-parler-tts"
OUTPUT_DIR = ROOT / "data" / "tts_voices"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

PROMPT = "ಥ್ರೆಡ್ ಶೆಡ್ಯೂಲರ್ ಚಾಲನೆಯಲ್ಲಿರುವ ಪ್ರಕ್ರಿಯೆಗಳನ್ನು ಪೂರ್ವಭಾವಿಯಾಗಿ ತಡೆಯುತ್ತದೆ."

VOICES = ["Suresh", "Anu", "Chetan", "Vidya"]


def description_for(voice: str) -> str:
    return (
        f"{voice} speaks in a clear narration tone at a moderate pace, "
        f"close-sounding and high quality, with no background noise."
    )


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}")

    model = ParlerTTSForConditionalGeneration.from_pretrained(str(MODEL_DIR)).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
    prompt_ids = tokenizer(PROMPT, return_tensors="pt").to(DEVICE)

    for voice in VOICES:
        description = description_for(voice)
        desc_ids = description_tokenizer(description, return_tensors="pt").to(DEVICE)

        print(f"Generating {voice}...")
        generation = model.generate(
            input_ids=desc_ids.input_ids,
            attention_mask=desc_ids.attention_mask,
            prompt_input_ids=prompt_ids.input_ids,
            prompt_attention_mask=prompt_ids.attention_mask,
        )
        audio_arr = generation.to(torch.float32).cpu().numpy().squeeze()
        out_path = OUTPUT_DIR / f"{voice.lower()}.wav"
        sf.write(str(out_path), audio_arr, model.config.sampling_rate)
        duration_s = len(audio_arr) / model.config.sampling_rate
        print(f"  wrote {out_path} — {duration_s:.2f}s")