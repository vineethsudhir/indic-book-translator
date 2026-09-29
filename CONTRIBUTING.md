# Contributing

Thanks for helping improve Indic Book Translator. The distribution name is
`indic-book-translator`; its Python import package is `kannada_epub`. Optional
extras are `gui` for the app and `local` for local model support.

## Development setup

Use Python 3.11 or later and a virtual environment from the repository root:

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[gui]"
```

For local IndicTrans2, Parler-TTS, or MiniLM providers, install both extras:

```bash
.venv/bin/pip install -e ".[gui,local]"
```

Local model files are separate downloads. Do not download models as part of
routine CI or a cloud-only test run.

## Tests and lint

Tests are plain Python scripts, not pytest tests. Run the fast, offline set
after a change:

```bash
.venv/bin/python scripts/test_sentence_split.py
.venv/bin/python scripts/test_epub_writer.py
.venv/bin/python scripts/test_tables.py
.venv/bin/python scripts/test_qa.py
.venv/bin/python scripts/test_pipeline.py
.venv/bin/python scripts/test_app.py
.venv/bin/python scripts/test_tts_engine.py
.venv/bin/python scripts/test_ui_browser.py
```

Install the contributor tools (Ruff and the browser test's websocket client)
with `.venv/bin/pip install -e ".[gui,dev]"`. The browser test skips itself if
Chrome is unavailable. These scripts need local
models, a running Ollama instance, or API keys and are run only when relevant
and available:

```text
test_translation, test_tts, test_tts_voices, test_glossary,
test_consistency_editor, test_book_translator
```

Before opening a pull request, run:

```bash
.venv/bin/ruff check src scripts
```

## Project invariants

Read [AGENTS.md](AGENTS.md) before changing the pipeline or EPUB handling. In
particular:

- Never misalign source paragraphs and translated paragraphs. Every N-to-N
  stage must raise on a count mismatch rather than pad, truncate, or guess.
- `Paragraph.index` is the join key between EPUB extraction and writing. Keep
  the block-tag enumeration and skip rules in sync.
- Batch positions refer to the filtered chapter paragraph list, not DOM indices;
  use `translations_from_batches` to map them.
- Sentence splitting must not split after closing quotes and should merge
  forward after abbreviation-shaped tokens. Measure changes against the sample
  EPUB as well as focused cases.
- Keep local ML imports lazy so the app runs with cloud dependencies only.
- EPUB writing edits the ZIP directly, retains `mimetype` first and stored,
  copies untouched entries byte-for-byte, and writes through a temporary file
  before renaming.

## Branches and pull requests

Work on a topic branch and open a pull request with a concise description of
the change and its verification. Do not commit directly to `main`. Commit
subjects should be imperative; the body should explain why the change is
needed and end with what was verified and how.

## Adding another Indic language

The translation engine and sentence splitter currently use the FLORES codes
`eng_Latn` and `kan_Knda`. Start by identifying the language codes and model
support, then inspect the source/target handling and language-specific sentence
splitting in `src/kannada_epub/translation/engine.py`. The splitter currently
has explicit English and Kannada rules; other languages pass through without
sentence splitting. Review provider behavior, UI language assumptions, QA, and
tests as part of any language change.
