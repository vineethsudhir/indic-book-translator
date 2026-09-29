# English → Kannada Book Translator + Audiobook Narrator

Local-first pipeline that translates English EPUB books into Kannada and can
narrate the result as an audiobook. Draft translation runs locally by default
(IndicTrans2); a consistency-editing pass enforces a persistent glossary,
fixes pronoun/register drift, and tags each paragraph with an emotion for
expressive TTS. **No powerful computer?** Every stage — translation, editing,
and TTS — can run on cloud APIs with your own credits instead; see
[Low-end machines: full-cloud setup](#low-end-machines-full-cloud-setup).

```
EPUB → extract chapters + table cells → draft translation → glossary + LLM consistency edit
     → optional QA (back-translate + similarity score) → translated Kannada EPUB
     (+ per-chapter JSON, qa_report.json, optional WAV audiobook)
```

Sample book included: Project Gutenberg's *Adventures of Sherlock Holmes*
(public domain).

## Quick start: the app

The easiest way to use the translator is the app: pick an EPUB, press
**Translate**, download the Kannada EPUB. It runs in your browser.

**1. Install** (once, from the repo folder):

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[gui]"            # cloud providers only (small)
# or, to also run models on this computer:
.venv/bin/pip install -e ".[gui,local]"      # adds torch, IndicTrans2, Parler-TTS
```

On Windows use `.venv\Scripts\pip` and `.venv\Scripts\python` instead of
`.venv/bin/...`.

**2. Start it:**

```bash
.venv/bin/python -m kannada_epub.app
```

Your browser opens at http://127.0.0.1:7860. Leave the terminal open while
you use the app; press Ctrl+C there to quit.

**3. Choose your providers (Settings tab), then Save.** The defaults are
cloud services, so either add keys or switch to local:

| Stage | Cloud default | Key needed | Local alternative (needs `.[local]`) |
|---|---|---|---|
| Translation | Sarvam `sarvam-m` | `SARVAM_API_KEY` | `indictrans2_local` |
| Consistency edit | Anthropic `claude-haiku-4-5` | `ANTHROPIC_API_KEY` | `ollama` + `gemma4:26b` (install [Ollama](https://ollama.com)) |
| Audiobook (TTS) | Sarvam `bulbul:v3` | `SARVAM_API_KEY` | `parler_local` |

Paste keys into the **API keys** section of Settings. They are saved in your
user data folder (below), never in the repo, and never shown again. Keys
already set in your shell environment also work.

**4. Translate (Translate tab).** Choose the `.epub`, optionally tick
**Run QA check** / **Build audiobook**, and for a first try set
**Preview: first N paragraphs per chapter** to 2 — a whole book takes much
longer. When it finishes, download the results from the buttons under the log,
or press **Open output folder**.

**5. Review (Review tab)** shows English and Kannada side by side per chapter,
with each paragraph's QA status; tick *Show only flagged/retry* to see the
ones worth checking by hand.

**Where your files are.** Settings, keys, uploaded books and outputs live in a
per-user folder, one sub-folder per book under `outputs/`:

| OS | Folder |
|---|---|
| macOS | `~/Library/Application Support/KannadaBookTranslator` |
| Windows | `%LOCALAPPDATA%\KannadaBookTranslator` |
| Linux | `~/.local/share/KannadaBookTranslator` |

On macOS `~/Library` is hidden in Finder and the path contains a space, so use
the app's **Open output folder** button, or:
`open "$HOME/Library/Application Support/KannadaBookTranslator/outputs"`.
Set `KANNADA_APP_DATA_DIR` to use a different folder.

**QA check.** QA translates the Kannada back to English and compares it with
the original: ≥ 0.85 similarity passes, 0.70–0.85 gets one automatic
re-translation, below 0.70 is flagged (highlighted in the EPUB and listed in
`qa_report.json`). In Settings, pick one of:

- *Cloud only:* back-translation `llm` (uses your editing provider) +
  embedding `openai_compatible` with an OpenAI key (`text-embedding-3-small`),
  or Ollama (`http://localhost:11434/v1`, model `nomic-embed-text` after
  `ollama pull nomic-embed-text`).
- *Local (needs `.[local]`):* back-translation `indictrans2_local` (run
  `python scripts/download_qa_models.py` once) and/or embedding
  `local_minilm` with model `sentence-transformers/all-MiniLM-L6-v2`.

**Local mode tab** shows which local libraries are installed and how to add
them. Local models are downloaded separately — see
[Bring your own models / credits](#bring-your-own-models--credits).

## Prerequisites

- Python 3.11+
- ~8 GB disk for models (`models/`, git-ignored)
- For the default local editing pass: [Ollama](https://ollama.com)
- For audiobooks: Hugging Face access to `ai4bharat/indic-parler-tts` (gated)

## Command-line setup (fresh clone)

The app above covers most uses. The command line runs the same pipeline from
config files, which suits long unattended runs and scripting.

```bash
pip install -e ".[local]"       # local defaults; plain `pip install -e .` is cloud-only
cp config/book.example.yaml config/book.yaml
cp config/consistency_editor.example.yaml config/consistency_editor.yaml
cp .env.example .env            # only needed for cloud providers (below)
python scripts/setup.py         # checks everything, tells you what to fix
python scripts/translate_book.py --limit-chapters item4 --max-paragraphs 2 --batch-size 2  # smoke test
python scripts/translate_book.py  # full book
```

`scripts/gui.py` is the older developer GUI over these config files
(`book.yaml` editor, setup checks); for everyday use prefer the app
(`python -m kannada_epub.app`).

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
# --no-epub                  skip writing the translated .kn.epub
# --audiobook                also narrate the translation to a WAV
```

To turn on QA from the command line, add a `qa:` block to `config/book.yaml`
(fields as in the app's Settings tab):

```yaml
  qa:
    enabled: true
    back_translation: indictrans2_local    # or llm
    embedding: local_minilm                # or openai_compatible
    embedding_model: sentence-transformers/all-MiniLM-L6-v2
```

`scripts/build_epub.py` rebuilds the translated EPUB from an existing output
folder without re-translating.

Config-file equivalents of the last three are `limit_chapters`,
`max_paragraphs_per_chapter`, and `batch_size` in `config/book.yaml`.
`exclude_ids` lists spine ids to skip (defaults cover Project Gutenberg
`pg-header`/`pg-footer` boilerplate and the cover wrapper) — add your book's
front/back matter ids there if you don't want them translated.

### Outputs

```
data/book_output/              (app: <data folder>/outputs/<book name>/)
  <book>.kn.epub               the translated book — original layout, images and
                               links kept, Noto Sans Kannada embedded
  qa_report.json               QA scores + status per paragraph (when QA is on)
  <book>.kn.wav                audiobook (when requested)
  chapters/<chapter_id>.json   per-chapter: source English, draft + edited
                               Kannada, emotion tags, batch spans
  checkpoints/<id>_<start>.json  per-batch files; re-runs resume from these
  qa/<chapter_id>.json         per-chapter QA cache (resumed runs skip QA)
  manifest.json                chapter list + titles + skip record
```

Known limitation: bold/italic/links *inside* a translated paragraph are
dropped (the paragraph itself, its ids and all other markup are kept).

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

Tick **Build audiobook** in the app, or pass `--audiobook` to
`scripts/translate_book.py`, to narrate the translation into one WAV after the
EPUB is written. Narration is the slowest stage (local Parler-TTS takes
seconds per sentence), so try it on a preview first.
`scripts/build_audiobook.py` is a sample-span demo (first 20 paragraphs of the
sample chapter) for voice/quality checks.

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

- **App: "A local provider is selected but the required libraries are
  missing"** — switch that stage to a cloud provider in Settings, or
  `pip install -e ".[gui,local]"` and restart the app.
- **App: "Missing API key(s): …"** — add them under Settings → API keys (or
  export them in your shell before starting the app).
- **App: embedding model error mentioning Hugging Face** — the QA embedding
  model doesn't match the backend. For `local_minilm` use
  `sentence-transformers/all-MiniLM-L6-v2`; for `openai_compatible` use the
  API's model name (`text-embedding-3-small`, `nomic-embed-text`, …).
- **Can't find the output files** — use **Open output folder** in the app;
  see *Where your files are* above.
- **Browser didn't open / can't connect** — go to http://127.0.0.1:7860
  yourself while the terminal running the app is still open; if the port is
  taken, start with `python -m kannada_epub.app --port 7861`.
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
src/kannada_epub/app/       the app (python -m kannada_epub.app)
src/kannada_epub/pipeline.py  the whole run: translate, QA, EPUB, audiobook
src/kannada_epub/           library (extract, translate, glossary, edit, QA, EPUB writer, TTS)
scripts/translate_book.py   command-line entry point (full book)
scripts/build_epub.py       rebuild the .kn.epub from an output folder
scripts/download_qa_models.py  local QA models (indic→en translation + MiniLM)
scripts/gui.py              older developer GUI over the config files
scripts/build_audiobook.py  sample-span demo: translate + narrate 20 paragraphs
scripts/setup.py            setup checker (all credentials + models, one place)
scripts/test_*.py           regression tests (plain scripts: python scripts/test_x.py)
config/*.example.yaml       versioned templates — copy without `.example` to use
assets/fonts/               Noto Sans Kannada (OFL) embedded into output EPUBs
data/sherlock_holmes.epub   sample book (public domain)
epub_kannada_translation_prd.md   full PRD (v2.0)
```

See the PRD for architecture and roadmap.

## License

MIT — see [LICENSE](LICENSE).
