# AGENTS.md

Guidance for coding agents working in this repo. The code, `README.md` and
`epub_kannada_translation_prd.md` explain what exists; this file covers what
they don't: invariants that break silently, and how to work here.

## Running things

- Use the repo venv: `.venv/bin/python`. System Python lacks the dependencies.
- Tests are plain scripts, not pytest: `.venv/bin/python scripts/test_<name>.py`
  (each prints `...: all assertions passed`).
  - Fast, offline, no models — run these after any change:
    `test_sentence_split`, `test_epub_io`, `test_epub_writer`, `test_tables`,
    `test_qa`,
    `test_pipeline`, `test_app`, `test_tts_engine`.
  - Need local models, Ollama or API keys — only run when asked:
    `test_translation`, `test_tts`, `test_tts_voices`, `test_glossary`,
    `test_consistency_editor`, `test_book_translator`.
- Validate EPUB output with EPUBCheck 5 (`java -jar epubcheck.jar out.epub`);
  the sample book passes with 0 errors and output must too.
- Install extras: `.[gui]` is the cloud-only app, `.[local]` adds
  torch/IndicTrans2/Parler-TTS. `pip install -e .` alone is cloud-only.

## Invariants (break these and output is silently wrong)

- **Never misalign.** Every stage that maps N inputs to N outputs (cloud
  translation, consistency editor, QA back-translation, embeddings) must
  raise on a count mismatch, never pad, truncate or guess. A wrong Kannada
  paragraph under the right English one is the worst failure this project has.
- **`Paragraph.index` is the join key** between `epub_io.load_epub_chapters`
  and `epub_writer.write_translated_epub`. Both enumerate
  `BeautifulSoup(content, "lxml").find_all(BLOCK_TAGS)`; nested blocks are
  skipped but still advance the enumeration. Changing `BLOCK_TAGS`, the
  parser, or the skip rule shifts indices and invalidates existing
  checkpoints/outputs.
- **Batch positions are not DOM indices.** `TranslatedBatch.paragraph_start`
  indexes the filtered `chapter.paragraphs` list; convert with
  `translations_from_batches`.
- **Sentence splitting** (`translation/engine.py::_split_sentences`): a false
  merge is harmless (two sentences translated together), a false split cuts a
  sentence in half. So: never split after a closing quote (Kannada reporting
  verbs follow quotes), and merge forward after abbreviation-shaped tokens.
  Measure changes against `data/sherlock_holmes.epub`, not just unit cases.
- **Local ML imports stay lazy.** `torch`, `transformers`, `ctranslate2`,
  `sentencepiece`, `IndicTransToolkit`, `parler_tts`, `huggingface_hub` may be
  imported only inside the code path that uses them. The app must import and
  run cloud-only without them; `test_pipeline`/`test_app` enforce this with an
  import blocker.
- **Parler-TTS output is sampled.** Short prompts sometimes return shape
  `(1, 1)`: flatten with `reshape(-1)`, never `squeeze()`.
- **EPUB reading and writing use `zipfile` + `lxml` directly** (no EbookLib,
  which is AGPL; don't add it back). Writing: `mimetype`
  first and stored, every untouched entry byte-identical, write to a temp file
  then rename.

## App rules

- The app must run the pipeline in-process. No `subprocess` of Python or
  `scripts/`; it gets packaged as a desktop app where neither exists.
- File locations come from `src/kannada_epub/app/paths.py`: user files
  (books, outputs) in `Documents/KannadaBookTranslator`, private state
  (settings, `secrets.env`, glossary) in the OS app-data folder. Never write
  into the repo at runtime. Tests set `KANNADA_APP_DATA_DIR` to a temp dir.
- API keys: never return, log, or display stored values; report set/not-set
  only. `secrets.env` is mode 0600.
- The local web server binds `127.0.0.1` only, checks the `Host` header, and
  requires the per-launch token on API calls; ids and file names from requests
  are validated against real directory listings.
- The UI must work offline: no CDN or other remote resources.

## Working here

- Don't commit to `main`; work on a branch. Commit only when asked.
- Commit messages: imperative subject line; the body says why, and ends with
  what was verified and how (e.g. "Verified: … passes epubcheck with 0 errors").
- Don't add dependencies without saying so; don't download models unless the
  task says to. `models/`, `data/` outputs, `.env` and `config/*.yaml` (not the
  `.example` files) are git-ignored — keep it that way.
- If a task conflicts with existing code or tests, don't guess: keep the
  existing tested behavior and state the conflict. Non-interactive runs
  (e.g. `opencode run`) cannot ask questions — end your summary with a line
  starting `QUESTION:` instead.
