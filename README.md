# English → Kannada Book Translator + Audiobook Narrator

Local-first pipeline that translates English EPUB books into Kannada and can
narrate the result as an audiobook. Draft translation runs fully offline
(IndicTrans2); a consistency-editing pass enforces a persistent glossary,
fixes pronoun/register drift, and tags each paragraph with an emotion for
expressive TTS (Indic Parler-TTS).

```
EPUB → extract chapters → IndicTrans2 draft → glossary + LLM consistency edit
     → per-chapter Kannada JSON (+ optional WAV audiobook)
```

Sample book included: Project Gutenberg's *Adventures of Sherlock Holmes*
(public domain).

## Quickstart (fresh clone)

```bash
pip install -e .
cp config/book.example.yaml config/book.yaml
cp config/consistency_editor.example.yaml config/consistency_editor.yaml
cp .env.example .env            # only needed for cloud providers (below)
python scripts/setup.py         # checks everything, tells you what to fix
python scripts/translate_book.py --limit-chapters item4 --max-paragraphs 2 --batch-size 2  # smoke test
python scripts/translate_book.py  # full book
```

Outputs land in `data/book_output/` (`chapters/*.json`, `checkpoints/`,
`manifest.json`). Re-runs resume finished chapters from checkpoints.

## Bring your own models / credits

All credentials and model paths live in **three files** (plus `models/`):

| What | Where | Default |
|---|---|---|
| Book, output paths, model dirs, batch size | `config/book.yaml` (from `book.example.yaml`) | local paths |
| Editing LLM provider + model | `config/consistency_editor.yaml` (from `consistency_editor.example.yaml`) | local Ollama |
| API keys (cloud only) | `.env` (from `.env.example`) | empty |
| Translation + TTS weights | `models/` (git-ignored, ~8 GB) | — |

**Option A — fully local (default, no keys):** install
[Ollama](https://ollama.com), `ollama pull gemma4:26b`, keep
`provider: ollama` in the provider config.

**Option B — your OpenAI-compatible credits** (OpenAI, OpenRouter, Together,
Groq, …): set `provider: openai_compatible`, `model:`, and `base_url:` in the
provider config, put your key in `.env` as `OPENAI_API_KEY`.

**Option C — your Anthropic credits:** set `provider: anthropic`,
`model: claude-sonnet-4-…`, put your key in `.env` as `ANTHROPIC_API_KEY`.

**Models to download:** a CTranslate2 conversion of
`ai4bharat/indictrans2-en-indic-1B` into the `ct2_model_dir` layout
(`model.bin` + `vocab/model.SRC` + `vocab/model.TGT`), and — for audiobooks
only — `ai4bharat/indic-parler-tts` via
`python scripts/download_tts_model.py` (gated; needs `huggingface-cli login`).
`python scripts/setup.py` verifies all of it.

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
