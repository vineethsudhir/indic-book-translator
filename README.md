# English → Kannada Book Translator + Audiobook Narrator

Local-first pipeline that translates English EPUB books into Kannada and can
narrate the result as an audiobook. Draft translation runs locally by default
(IndicTrans2); a consistency-editing pass enforces a persistent glossary,
fixes pronoun/register drift, and tags each paragraph with an emotion for
expressive TTS. **No powerful computer?** Every stage — translation, editing,
and TTS — can run on cloud APIs with your own credits instead; see
[Low-end machines: full-cloud setup](#low-end-machines-full-cloud-setup).

```
EPUB → extract chapters → draft translation → glossary + LLM consistency edit
     → per-chapter Kannada JSON (+ optional WAV audiobook)
```

Sample book included: Project Gutenberg's *Adventures of Sherlock Holmes*
(public domain).

## Prerequisites

- Python 3.11+
- ~8 GB disk for models (`models/`, git-ignored)
- For the default local editing pass: [Ollama](https://ollama.com)
- For audiobooks: Hugging Face access to `ai4bharat/indic-parler-tts` (gated)

## Setup (fresh clone)

```bash
pip install -e .
cp config/book.example.yaml config/book.yaml
cp config/consistency_editor.example.yaml config/consistency_editor.yaml
cp .env.example .env            # only needed for cloud providers (see below)
python scripts/setup.py         # checks everything, tells you what to fix
```

`setup.py` validates the Python version, installed deps, book + provider
configs, the EPUB path, the translation model layout, the TTS model, and your
chosen provider's credentials (Ollama reachable + model pulled, or API key
present). Fix what it flags, re-run until it passes.

Smoke-test the whole pipeline on 2 paragraphs before committing to a full run:

```bash
python scripts/translate_book.py --limit-chapters item4 --max-paragraphs 2 --batch-size 2
```

## Translating your own EPUB

1. Put your `.epub` somewhere (e.g. `data/my_book.epub`).
2. Point the book config at it:
   ```yaml
   # config/book.yaml
   book:
     epub_path: data/my_book.epub
     output_dir: data/book_output
     glossary_db: data/project.glossary.db
   ```
3. Run setup, then translate:
   ```bash
   python scripts/setup.py
   python scripts/translate_book.py
   ```

### Useful options

```bash
python scripts/translate_book.py --help
# --config CONFIG            use a different book config file
# --epub PATH                override the configured EPUB
# --output-dir DIR           override the output directory
# --limit-chapters ID [ID..] translate only these spine ids (find them in manifest.json after a run, or check the EPUB)
# --max-paragraphs N         trim every chapter to N paragraphs (cheap trial runs)
# --batch-size N             paragraphs per consistency-edit call (default 20)
```

Config-file equivalents of the last three are `limit_chapters`,
`max_paragraphs_per_chapter`, and `batch_size` in `config/book.yaml`.
`exclude_ids` lists spine ids to skip (defaults cover Project Gutenberg
`pg-header`/`pg-footer` boilerplate and the cover wrapper) — add your book's
front/back matter ids there if you don't want them translated.

### Outputs

```
data/book_output/
  chapters/<chapter_id>.json   per-chapter: source English, draft + edited
                               Kannada, emotion tags, batch spans
  checkpoints/<id>_<start>.json  per-batch files; re-runs resume from these
  manifest.json                chapter list + titles + skip record
```

Rolling context resets at every chapter boundary (right for story
collections; see the PRD). Re-runs skip finished chapters, so an interrupted
full-book run resumes instead of restarting.

### Glossary workflow (recommended before a full run)

Character names, acronyms, and technical terms stay consistent via a SQLite
translation memory. Mine candidates from your book, review them in a
spreadsheet, then translate:

```python
from kannada_epub.epub_io import load_epub_chapters
from kannada_epub.glossary import (
    GlossaryStore, extract_candidates,
    export_pending_for_review, import_reviewed,
)

store = GlossaryStore("data/project.glossary.db")
for ch in load_epub_chapters("data/my_book.epub"):
    store.add_candidates(extract_candidates(ch.text, ch.id))
export_pending_for_review(store, "glossary_review.csv")
# ... fill in target_term + approved/rejected in the CSV, then:
import_reviewed(store, "glossary_review.csv")
```

Only `approved` rows with a `target_term` take effect; the runner picks up
relevant terms per batch automatically.

### Audiobook

`scripts/build_audiobook.py` narrates the first 20 paragraphs of the sample
chapter end to end (translate → edit → Parler-TTS → single WAV) — it is a
sample-span demo with hardcoded paths, meant for voice/quality checks. Full-book
narration consumes the runner's `chapters/*.json` (same `TranslatedBatch`
contract as `audiobook_builder.build_audiobook`).

## Bring your own models / credits

All credentials and model paths live in **three files** (plus `models/`):

| What | Where | Default |
|---|---|---|
| Book, output paths, model dirs, batch size | `config/book.yaml` (from `book.example.yaml`) | local paths |
| Editing LLM provider + model | `config/consistency_editor.yaml` (from `consistency_editor.example.yaml`) | local Ollama |
| API keys (cloud only) | `.env` (from `.env.example`, auto-loaded) | empty |
| Translation + TTS weights | `models/` (git-ignored, ~8 GB) | — |

**Option A — fully local (default, no keys):** install
[Ollama](https://ollama.com), `ollama pull gemma4:26b`, keep
`provider: ollama` in the provider config, and download both local models
above.

**Option B — your OpenAI-compatible credits** (OpenAI, OpenRouter, Together,
Groq, Sarvam, …): set `provider: openai_compatible`, `model:`, and
`base_url:` in the provider config — and/or in `book.translation` /
`book.tts` — put your key in `.env` as `OPENAI_API_KEY` (or `SARVAM_API_KEY`
with `base_url: https://api.sarvam.ai/v1`, model `sarvam-m`).

**Option C — your Anthropic credits:** set `provider: anthropic`,
`model: claude-sonnet-4-…`, put your key in `.env` as `ANTHROPIC_API_KEY`.
(Editing pass only; translation/TTS cloud options are Sarvam or
OpenAI-compatible.)

**Option D — Sarvam cloud TTS:** `book.tts.provider: sarvam`,
`model: bulbul:v3`, `voice:` a lowercase speaker (`anushka`, …),
`SARVAM_API_KEY` in `.env`. Native `kn-IN` voices, no local TTS download.

**Models to download (local path only):** a CTranslate2 conversion of
`ai4bharat/indictrans2-en-indic-1B` into the `ct2_model_dir` layout
(`model.bin` + `vocab/model.SRC` + `vocab/model.TGT`), and — for audiobooks
only — `ai4bharat/indic-parler-tts` via
`python scripts/download_tts_model.py` (gated; needs `huggingface-cli login`).
`python scripts/setup.py` verifies all of it. Skip both if you use the
full-cloud setup below.

## Low-end machines: full-cloud setup

If your computer can't hold the local models (~8 GB + RAM/VRAM), run all
three LLM stages on cloud APIs. Recommended: [Sarvam AI](https://dashboard.sarvam.ai)
(native Kannada translation + TTS); any OpenAI-compatible vendor works for
translation and editing.

```bash
cp config/book.example.yaml config/book.yaml
cp config/consistency_editor.example.yaml config/consistency_editor.yaml
cp .env.example .env
```

1. Put your key in `.env`: `SARVAM_API_KEY=...`
2. In `config/book.yaml`, uncomment/set:
   ```yaml
   translation:
     provider: openai_compatible
     model: sarvam-m
     base_url: https://api.sarvam.ai/v1
     api_key_env: SARVAM_API_KEY
   tts:
     provider: sarvam
     model: bulbul:v3
     voice: anushka
     api_key_env: SARVAM_API_KEY
   ```
3. In `config/consistency_editor.yaml`, either keep local Ollama (lightest
   cloud bill — editing is the highest-volume stage) or point it at the same
   cloud vendor:
   ```yaml
   consistency_editor:
     provider: openai_compatible
     model: sarvam-m
     base_url: https://api.sarvam.ai/v1
     api_key_env: SARVAM_API_KEY
   ```
4. `python scripts/setup.py` — passes without any local model, then
   `python scripts/translate_book.py` as usual.

Notes:
- Cloud translation enforces the same paragraph-count guarantee as local
  (mismatched responses are retried once, then fail loudly instead of
  misaligning).
- Cloud TTS ignores the `emotion` tag (neither Sarvam Bulbul nor
  OpenAI-compatible `/audio/speech` has an emotion parameter); `voice`
  selects the speaker (`anushka`, `alloy`, …).
- Cost control: do the 2-paragraph smoke test first, keep the glossary
  approved (fewer editor retries), and translate before narrating — TTS is
  the most expensive stage per word.

## Troubleshooting

- **`Consistency editor returned empty output`** — a reasoning model burned
  `max_tokens` on hidden reasoning. Keep `think: false` for Ollama reasoning
  models, or raise `max_tokens`.
- **Ollama model not pulled** — `setup.py` tells you; run
  `ollama pull <model>`.
- **Weird first/last "chapters"** — some EPUBs put navigation or boilerplate
  in the spine. Non-linear items are skipped automatically; add the rest to
  `exclude_ids` in `config/book.yaml`.
- **Paragraph-count mismatch errors** — the editor split or merged paragraphs;
  the run refuses to misalign rather than corrupt. Retry the chapter (resume
  handles it); lower `batch_size` if it recurs.
- **Out of memory in translation** — the engine already runs quantized
  (`int8`) on CPU by default; if you raised the batch size or run on a small
  machine, lower `batch_size` in `config/book.yaml`.

## Repo layout

```
src/kannada_epub/      pipeline library (extract, translate, glossary, edit, TTS)
scripts/translate_book.py   main entry point (full book)
scripts/build_audiobook.py  sample-span demo: translate + narrate 20 paragraphs
scripts/setup.py            setup checker (all credentials + models, one place)
config/*.example.yaml    versioned templates — copy without `.example` to use
data/sherlock_holmes.epub   sample book (public domain)
epub_kannada_translation_prd.md   full PRD (v2.0)
```

See the PRD for architecture, deferred items (QA back-translation loop, table
handling, EPUB repackaging), and roadmap.

## License

MIT — see [LICENSE](LICENSE).
