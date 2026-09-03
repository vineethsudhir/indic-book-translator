# Product Requirements Document (PRD)
## Local English-to-Kannada EPUB Translation Pipeline, QA Engine & Audiobook Narrator

**Document Version:** 2.0.0
**Supersedes:** 1.0.0
**Target Architecture:** Local-first Execution (Offline / Edge Hardware, with optional cloud LLM for editing pass)
**Primary Focus:** Automated EPUB Translation, Term Preservation, Table Layout Integrity, Automated Semantic QA, Cross-Chapter Consistency Editing, Emotion-Tagged TTS Audiobook Generation & Human Proofing Workflow

> v2.0 change log: preserves all v1.0 requirements (FR-1..FR-5) with implementation-status annotations, and adds the scope actually built since v1.0 — persistent glossary / Translation Memory (FR-6), LLM consistency editor with provider abstraction (FR-7), per-paragraph emotion tagging (FR-8), Indic Parler-TTS audiobook synthesis (FR-9), and EN/KN proofing artifacts (FR-10). Translation-hardening lessons (sentence-splitting, token-chunking, Roman-numeral bypass, sentinel masking) are now normative.

---

## 1. Executive Summary

The objective of this project is to build a fully local, automated software pipeline designed to convert English EPUB ebooks into publishable, structurally intact Kannada EPUB ebooks **plus** a narrated Kannada audiobook of the same content.

To address common failure modes in machine translation—such as context loss, missing technical terms, corrupted HTML/table formatting, and unvetted translation errors—the system incorporates:
1. **Selective Term Preserving Masking:** Preventing unnecessary translation of code, brand names, proper nouns, and technical jargon (v1 HTML-span proposal superseded by sentinel-token masking — see FR-2.4).
2. **Smart Table AST Extraction:** Isolating, translating, and re-assembling complex HTML tables without compromising grid layout or cell alignment (still deferred — see FR-3.2 status).
3. **Automated QA & Redundancy Loop:** Executing back-translation (Kannada → English) combined with vector embedding semantic similarity scoring to flag low-confidence passages automatically (still deferred — see FR-4 status).
4. **Persistent Glossary / Translation Memory (NEW):** SQLite-backed per-book term store with heuristic candidate mining, CSV human-review loop, and forced substitution (FR-6).
5. **LLM Consistency Editing (NEW):** Chapter-scoped editing pass that enforces the approved glossary, fixes pronoun/referent drift using rolling intra-chapter context, and normalizes register — with strict paragraph-count preservation (FR-7).
6. **Emotion-Tagged Narration (NEW):** Same editing call classifies each paragraph into a fixed TTS emotion set, driving downstream expressive synthesis with no second classification pass (FR-8).
7. **Local TTS Audiobook Generation (NEW):** Paragraph-level Kannada narration via `ai4bharat/indic-parler-tts`, chunked to avoid silent truncation, peak-normalized and concatenated into a single WAV (FR-9).
8. **Human Proofing Artifacts (NEW):** EN/KN side-by-side sample files and print-ready proofing HTML/PDF with reviewer correction blocks (FR-10).

---

## 2. System Architecture & High-Level Workflow

The pipeline operates as a multi-stage sequential processing engine with recursive fallback mechanisms for low-confidence translations.

```
[Input EPUB File]
       │
       ▼
[1. Unpacker & Document Parser] ──► Extracts HTML/XHTML, CSS, Images, Manifests
       │                             (CURRENT: spine-order chapter -> paragraphs only)
       ▼
[2. AST Tree Builder & Reader] ──► Maps Text Nodes, Tables, & Metadata
       │                             (DEFERRED: full DOM/attribute preservation)
       ▼
[3. Selective Masking Engine] ──► Flags DNT (Do-Not-Translate) Terms & NER Tags
       │                             (CURRENT: sentinel tokens ⟦DNTn⟧ + glossary masking)
       ▼
[4. Translation Engine] ─────────► Batch Translates via Local NMT (IndicTrans2-1B via CTranslate2)
       │                             (CURRENT: sentence-split + token-chunked + roman-numeral bypass)
       ▼
[5. Glossary Enforcement] ───────► Forced substitution of approved terms, longest-first (NEW, FR-6)
       │
       ▼
[6. LLM Consistency Edit] ───────► Glossary + pronoun/register fix + emotion tag per para (NEW, FR-7/FR-8)
       │                             Context = rolling tail of SAME chapter only; resets at chapter boundary
       ▼
[7. QA & Back-Translation Loop] ──► Translates back to EN, Computes Vector Similarity (DEFERRED, FR-4)
       │                             ├──► Passes: Proceed to Reassembly
       │                             └──► Fails: Retry with Fallback Parameters / Log Flag
       ▼
[8. Layout & Font Re-injector] ──► Embeds Kannada Web Fonts (Noto Sans) & CSS Adjustments (DEFERRED, FR-5)
       │
       ▼
[9. EPUB Packager & Validator] ──► Generates Compliant Output .epub File (DEFERRED, FR-5)
       │
       ▼
[10. TTS Audiobook Builder] ─────► Emotion-tagged narration -> single WAV audiobook (NEW, FR-9)
       │
       ▼
[Proofing Artifacts] ────────────► EN/KN sample txt + proof HTML/PDF for human review (NEW, FR-10)
```

Continuity boundary rule (normative): **EPUB chapter == story boundary** for the current book shape (short-story collection, e.g. Sherlock Holmes). Rolling context carries forward within a chapter only and resets to empty at every chapter boundary. A single continuous novel split into chapters would want cross-chapter context — that is explicitly out of scope for `BookTranslator` as built.

---

## 3. Detailed Functional Requirements

### FR-1: EPUB Parsing & Container Unpacking
* **FR-1.1 Format Compliance:** Support EPUB 2.0 and EPUB 3.0 specification containers.
* **FR-1.2 Asset Preservation:** Preserve all structural elements without alteration, including:
  * CSS stylesheets and custom rules.
  * Raster images (PNG, JPEG, WebP) and SVG graphics.
  * Embedded fonts, audio clips, and metadata files (`content.opf`, `toc.ncx`, `nav.xhtml`).
* **FR-1.3 DOM & AST Extraction:** Parse XHTML body text into an Abstract Syntax Tree (AST) preserving parent-child tag hierarchies, inline attributes, and unique element IDs.
* **Implementation status (v2.0): PARTIAL.** `src/kannada_epub/epub_io.py::load_epub_chapters` implements spine-order (spec-correct reading order) extraction to `Chapter(id, title, paragraphs[])` over block tags `p/li/blockquote/h1-h6` with whitespace normalization. No DOM/attribute preservation, no table extraction, no reassembly. Full AST preservation, asset passthrough, and repackaging remain deferred to FR-5 work.

### FR-2: Selective Masking & Term Preservation (Do-Not-Translate)
* **FR-2.1 Named Entity & Context Protection:** Identify tokens that must remain in original English script or explicit transliterated forms:
  * Technical terminology, API names, function signatures, and mathematical variables.
  * Code blocks (`<code>`, `<pre>`), markup elements, and inline syntax.
  * Brand names, acronyms (e.g., HTTP, CPU, DNA), and specific proper nouns.
* **FR-2.2 Token Masking Protocol:** Replace protected strings with deterministic placeholder tags prior to feeding text to the translation model.
  * v1.0 proposed `<span class="dnt">` HTML tags. **Superseded in implementation:** NMT models reflow/duplicate/drop inline markup they were not trained on. Current protocol uses plain sentinel tokens (FR-2.4).
* **FR-2.3 Custom Dictionary & Regex Rules:** Support a user-defined configuration file (`dictionary.json`) allowing exact string matches and custom regex rules to force specific terms to bypass translation.
  * **Implementation status:** superseded by FR-6 glossary store (SQLite + CSV review) plus `mask_dnt_terms()` / `apply_forced_substitutions()`. A static `dictionary.json` is no longer the interface.
* **FR-2.4 Sentinel-token masking (NEW, normative):** `glossary/substitution.py::mask_dnt_terms` replaces each DNT term with `⟦DNT{index}⟧` (longest terms first, case-insensitive), translates the masked text, then `unmask_dnt_terms` restores originals via the restore map. Returns `(masked_text, restore_map)` as a pair; restore map must be consumed after translation, never persisted across batches.

### FR-3: Local Machine Translation & Table Handling

#### FR-3.1 Batch Translation Engine
* Utilize local open-weight Neural Machine Translation (NMT) models optimized for Indic languages (e.g., **IndicTrans2** ~1B via CTranslate2 runtime or **Sarvam-1**).
* Execute batch inference grouped by block elements (paragraphs, list items, callouts) to maintain contextual coherence.
* **Implementation status (v2.0): IMPLEMENTED with hardening (normative), v2.1 adds cloud alternative.** `src/kannada_epub/translation/engine.py::IndicTrans2Engine` (local default) plus `translation/cloud.py::OpenAICompatibleTranslationProvider` (any OpenAI-compatible chat endpoint — Sarvam `sarvam-m` recommended for Kannada) behind `translation/base.py::TranslationProvider` and `translation/factory.py`. Cloud path keeps the paragraph-count guarantee (retry once, then refuse rather than misalign). Selected via `book.translation` in `config/book.yaml`; `scripts/setup.py` validates whichever is configured.
  * CTranslate2 over `models/en-indic-1b-ct2`, `device=cpu`, `compute_type=int8`; SentencePiece `model.SRC`/`model.TGT`; `IndicProcessor(inference=True)`.
  * FLORES tags (`eng_Latn`/`kan_Knda`) prepended as raw tokens AFTER SentencePiece (tags shatter into garbage subwords if passed through SP — verified).
  * `translate_paragraphs()` splits paragraphs into sentences first (English regex `(?<=[.!?])\s+`; non-English passed through pending a Kannada segmenter), translates sentence batch, rejoins — matches IndicTrans2 sentence-level training; whole-paragraph blobs degrade quality and truncate content.
  * Any sentence encoding still exceeding `max_content_tokens` (default 200, +8 tag/margin headroom as CTranslate2 backstop) is chunked and merged post-translation — never silently truncated (verified: at 160 tokens unchunked, long paras lost final sentences silently).
  * Roman-numeral headings (`I.`, `III. THE RED-HEADED LEAGUE`) bypass translation verbatim and reattach afterward (verified: otherwise `I.` transliterates as Kannada vowel `ಐ.`).
  * `IndicProcessor.preprocess_batch/postprocess_batch` are stateful and paired 1:1 with nothing interleaved; engine instance is not thread-safe.

#### FR-3.2 Smart Table Processing Engine
* **Structure Preservation:** Extract `<table>` structures independently. Isolate cell text while preserving `rowspan`, `colspan`, `class`, `style`, and `align` attributes.
* **Cell-by-Cell Translation:** Process text within `<th>` and `<td>` elements individually or in structured arrays to prevent model structural confusion.
* **Layout Adjustment:** Automatically adjust cell line heights and font sizes to accommodate Kannada script expansion.
* **Implementation status (v2.0): DEFERRED.** No table code exists; `epub_io` extracts only block-tag text. Design from v1.0 (cell-map JSON → translate → re-inject) still holds when scheduled.

### FR-4: Automated Quality Assurance & Redundancy Loop
* (Unchanged from v1.0 — retained as requirement, still deferred.)

#### FR-4.1 Back-Translation Execution
* Translate the output Kannada text back to English ($T_{back}$) using an independent local translation path or model prompt configuration.

#### FR-4.2 Vector Semantic Comparison
* Pass the Original English text ($T_{orig}$) and Back-Translated English text ($T_{back}$) into a local sentence transformer model (e.g., `all-MiniLM-L6-v2` or `bge-small-en-v1.5`).
* Calculate the Cosine Similarity metric $S$:
  $$\text{Cosine Similarity } S = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}$$
  where $\mathbf{A} = \text{Embed}(T_{orig})$ and $\mathbf{B} = \text{Embed}(T_{back})$.

#### FR-4.3 Thresholding & Self-Correction Logic
* **Pass Condition ($S \ge 0.85$):** Accept translation and write to Kannada AST.
* **Retry Condition ($0.70 \le S < 0.85$):** Re-translate using alternative parameters (e.g., reduced beam search width, adjusted temperature, or alternative sentence segmentation).
* **Flag Condition ($S < 0.70$):** Accept output but tag node in HTML metadata (`class="qa-review-flag"`) and log detailed breakdown to `qa_report.json`.

```json
{
  "chapter": "chapter_03.xhtml",
  "node_id": "p_104",
  "original_en": "The thread scheduler preempts running processes.",
  "kannada_target": "ಥ್ರೆಡ್ ಶೆಡ್ಯೂಲರ್ ಚಾಲನೆಯಲ್ಲಿರುವ ಪ್ರಕ್ರಿಯೆಗಳನ್ನು ತಡೆಯುತ್ತದೆ.",
  "back_translated_en": "The thread scheduler stops running processes.",
  "similarity_score": 0.78,
  "status": "FLAGGED_FOR_REVIEW"
}
```
* **Implementation status (v2.0): DEFERRED.** No back-translation or embedding code exists. Current quality gates are: sentence/chunk loss prevention (FR-3.1), glossary enforcement (FR-6), LLM consistency edit with paragraph-count refusal (FR-7), and human proofing (FR-10).

### FR-5: EPUB Generation & Rendering Enhancements
* **FR-5.1 Font Injection:** Embed standard open-source Kannada typefaces (e.g., *Noto Sans Kannada*, *Tiro Kannada*) directly into the output `.epub` font directory.
* **FR-5.2 Style Auto-Tuning:** Inject CSS overrides to handle script rendering specifics:
  * Increase default `line-height` to `1.45em` - `1.6em` to prevent vertical overlap of Kannada diacritics (*kagunita*).
  * Configure `font-family` fallbacks across global stylesheets.
* **FR-5.3 Manifest Verification:** Generate updated `content.opf` manifests and ensure compliance using `epubcheck`.
* **Implementation status (v2.0): DEFERRED.** No packaging code exists. Requirement retained unchanged.

### FR-6: Persistent Glossary / Translation Memory (NEW — implemented)
* **FR-6.1 Store:** `src/kannada_epub/glossary/store.py::GlossaryStore` — one SQLite file per book/series (`data/project.glossary.db`). Schema `glossary_terms(source_term, source_term_key UNIQUE, target_term, term_type, status pending|approved|rejected, frequency, first_seen_chapter, notes, timestamps)`. `add_candidates` inserts pending / bumps frequency on repeats; `upsert_review` applies human decisions; `get_approved_dict()` returns `{en: kn}` lock map; `get_relevant_glossary(text)` scopes the map to terms occurring in the current batch (keeps editor prompts small).
* **FR-6.2 Candidate mining:** `glossary/extraction.py::extract_candidates` — dependency-free regex/frequency heuristics tuned for recall (human reviews all): all-caps acronyms, multi-word Cap phrases (leading sentence-stopwords stripped so `The Thread Scheduler` == `Thread Scheduler`), single repeated non-sentence-initial capitals (character names). `min_frequency` default 3. Explicitly swappable for spaCy NER later; downstream depends only on `list[Candidate]`.
* **FR-6.3 Human review loop:** `glossary/review.py::export_pending_for_review` → CSV editable in any spreadsheet → `import_reviewed` applies decisions; `approved`-without-`target_term` rows are refused with reasons, never silently dropped.
* **FR-6.4 Forced substitution:** `glossary/substitution.py::apply_forced_substitutions` — longest-terms-first, case-insensitive replacement of every approved source term with its locked target rendering.

### FR-7: LLM Consistency Editing (NEW — implemented)
* **FR-7.1 Scope:** `src/kannada_epub/consistency_editor.py::ConsistencyEditor.edit_chapter` + `src/kannada_epub/book_translator.py::BookTranslator`. Draft Kannada batch (default 20 paragraphs) + relevant glossary + rolling prior-chapter context (tail 2 paras of same chapter, English source) + register string (default `neutral, standard written Kannada`) → edited Kannada. Fixes ONLY: (1) glossary term enforcement, (2) pronoun/referent consistency, (3) register. No rewriting of correct sentences; no meaning change/addition/removal.
* **FR-7.2 Paragraph-integrity guarantee:** `[P<n>]` numbering; model must return exactly the same count/order/tags. `_parse_numbered_output` raises on missing/unexpected numbers; `BookTranslator` raises on batch-length mismatch — never misalign edited text/emotions against wrong source paras. Empty output raises (known cause: reasoning model burning `max_tokens` on hidden reasoning).
* **FR-7.3 Provider abstraction:** `src/kannada_epub/providers/` — `ConsistencyEditorProvider.complete(system, user)` interface; `factory.build_provider` selects `ollama` (default, `http://localhost:11434`, `think: false` default), `openai_compatible` (any OpenAI-style API, requires `base_url` + `api_key_env`), or `anthropic`. Prompt logic is provider-agnostic. Config: `config/consistency_editor.example.yaml` → copy to `consistency_editor.yaml`; keys via `.env` (`OPENAI_API_KEY` / `ANTHROPIC_API_KEY`) — see `.env.example`.
* **FR-7.4 Orchestration:** `BookTranslator.translate_chapters(chapters)` — per chapter: batch → `engine.translate_paragraphs(en, eng_Latn, kan_Knda)` → `glossary.get_relevant_glossary` → `editor.edit_chapter` → `TranslatedBatch(chapter_id, title, span, source_english, draft_kannada, edited_kannada, edited_emotions, prior_context_used)`; rolling context advances from English source tail of the batch just processed.

### FR-8: Per-Paragraph Emotion Tagging (NEW — implemented)
* Same `edit_chapter` call returns one emotion tag per paragraph from the fixed `EMOTIONS` set matching `ai4bharat/indic-parler-tts` supported tags: Command, Anger, Narration, Conversation, Disgust, Fear, Happy, Neutral, Proper Noun, News, Sad, Surprise. Off-list tags normalize to `Narration`. Output shape per para: `[Pn]` + `EMOTION: <tag>` + edited text. No second classification pass.

### FR-9: Local TTS Audiobook Synthesis (NEW — implemented)
* **FR-9.1 Engine:** `src/kannada_epub/tts/engine.py::IndicParlerTTSEngine` wraps `ai4bharat/indic-parler-tts` (`models/indic-parler-tts`), device `mps` if available else `cpu`. Description prompt per chunk: `{voice} speaks in {emotion_phrase} at a moderate pace, close-sounding and high quality, with no background noise.`
**v2.1 adds cloud alternatives** behind `tts/base.py::TTSProvider` + `tts/factory.py`: `tts/cloud.py::SarvamTTSProvider` (Bulbul `bulbul:v3`, native `kn-IN`, recommended) and `OpenAICompatibleTTSProvider` (`/audio/speech`). Cloud voices ignore the emotion tag (no equivalent parameter). Selected via `book.tts` in `config/book.yaml`.
* **FR-9.2 Chunking (normative):** Kannada sentence-split on `. ! ? । ॥`, grouped into ≤~200-char chunks — a single `generate()` tops out at ~30s audio (`max_length=2610` codec tokens) and silently truncates beyond it. Intra-paragraph chunks joined with 0.25s gaps; per-chunk peak normalization prevents inter-chunk volume jumps.
* **FR-9.3 Audiobook assembly:** `src/kannada_epub/audiobook_builder.py::build_audiobook` narrates every `edited_kannada` para with its emotion tag + chosen voice (e.g. `Chetan`), concatenates with 0.6s paragraph gaps into one WAV (`data/scandal_in_bohemia_audiobook.wav`). Scripts: `scripts/build_audiobook.py` (20-para end-to-end), `scripts/download_tts_model.py`, `scripts/test_tts*.py`.

### FR-10: Human Proofing Artifacts (NEW — implemented)
* `scripts/translate_sample_chapter.py` → `data/scandal_in_bohemia_sample.txt` (raw IndicTrans2 EN/KN pairs for `item4`, first 20 paras — glossary/editor deliberately excluded to judge raw quality).
* `scripts/build_proof_pdf.py` + `data/scandal_in_bohemia_proof.html` → print-ready proof HTML/PDF (`data/scandal_in_bohemia_proof.pdf`) with EN row, KN row, and blank reviewer-correction lines per paragraph.
* `scripts/test_*.py` (`test_translation`, `test_glossary`, `test_consistency_editor`, `test_book_translator`, `test_tts`, `test_tts_voices`) are the regression/voice-selection harness.

---

## 4. Non-Functional Requirements (NFRs)

| Metric | Specification |
|---|---|
| **Privacy & Security** | Local-first. Translation (IndicTrans2/CTranslate2), glossary, TTS, and EPUB parsing run 100% offline with zero data transmission. The consistency-editing pass (FR-7) is provider-pluggable: default `ollama` stays offline; `openai_compatible`/`anthropic` options send draft Kannada + glossary + rolling context to the configured API — operators must opt in explicitly via config + API key. v2.1 extends the same opt-in model to translation (`book.translation.provider: openai_compatible`) and TTS (`book.tts.provider: sarvam`/`openai_compatible`) for machines that cannot hold local models. |
| **Performance** | v1.0 target retained: standard 80,000-word book in < 20 minutes on consumer GPUs (Apple Silicon M-series or NVIDIA RTX 3060+) for the translation path. TTS synthesis is slower than translation and scales with audio length — budget separately; current scripts synthesize sample spans, not full books. |
| **Memory Footprint** | Peak VRAM usage ≤ 6 GB for NMT (quantized 4-bit / 8-bit or CTranslate2 `int8` binaries). TTS (`parler-tts`) runs on MPS/CPU in current implementation. |
| **Output Integrity** | Zero valid HTML structural errors; valid EPUB schema compliant with standard e-readers (Kindle, Kobo, Apple Books). |
| **Determinism & Safety** | Never silently drop content (translation chunking FR-3.1, TTS chunking FR-9.2). Never misalign edited output against source (paragraph-count refusal FR-7.2). Approved-without-target glossary rows are refused, not dropped (FR-6.3). |
| **Reproducibility** | Pinned local model dirs (`models/en-indic-1b-ct2`, `models/indic-parler-tts`), sample chapter + proof artifacts checked into `data/`, provider config versioned as `config/consistency_editor.example.yaml`. |

---

## 5. Technical Stack Recommendations

* **Programming Language:** Python 3.11+
* **EPUB Parsing / DOM Manipulation:** `EbookLib`, `BeautifulSoup4`, `lxml`
* **Local Translation Models:** `IndicTrans2` (AI4Bharat) via `ctranslate2` (+ `sentencepiece`, `IndicTransToolkit`, `huggingface_hub`, `torch`, `transformers`) — see `pyproject.toml`
* **Named Entity Recognition / Masking:** heuristic regex/frequency (`glossary/extraction.py`) now; `spaCy` (`en_core_web_sm`) + custom regex remains the future option per FR-2
* **Consistency Editing LLM:** Ollama (`gemma4:26b` default) / any OpenAI-compatible API / Anthropic native — via `providers/` abstraction; `pyyaml`, `pydantic`, `httpx`, `openai`, `anthropic` clients
* **Embedding Model (QA Loop, deferred):** `sentence-transformers` (`all-MiniLM-L6-v2`)
* **TTS / Audio:** `ai4bharat/indic-parler-tts` via `parler_tts`, `soundfile`, `numpy`, `torch`
* **Font Packages:** Google Noto Fonts (*Noto Sans Kannada*)
* **Glossary Store:** stdlib `sqlite3` (no ORM)

---

## 6. Risk Matrix & Mitigation Strategies

| Risk Factor | Impact | Mitigation Strategy |
|---|---|---|
| **Table Grid Distortion** | High | Extract table content into strict JSON schemas prior to translation. Re-assemble cell-by-cell matching exact AST node structure. (Still deferred — FR-3.2.) |
| **Diacritics Overlap (Rendering)** | Medium | Automatically inject global CSS line-height patches (`line-height: 1.55 !important`) into all XHTML documents. (Still deferred — FR-5.2.) |
| **Model Translation Hallucinations** | High | Enforce strict sentence-length ratio checks and semantic similarity verification via the QA back-translation loop. (Deferred — current mitigations: sentence-level input, chunking, glossary lock, consistency edit, human proof.) |
| **Missing Fonts on Target E-Readers** | Medium | Hard-embed font files (`.ttf` / `.woff2`) inside the `.epub` container and declare explicit `@font-face` rules. (Still deferred — FR-5.1.) |
| **Silent content truncation (NMT)** | High | Sentence-split + token-chunk + merge (FR-3.1). Verified fix for lost trailing sentences. |
| **Silent content truncation (TTS)** | High | ≤200-char sentence-group chunks per generate() call (FR-9.2). Verified ~30s ceiling. |
| **Wrong-context pronoun fixes** | Medium | Rolling context scoped to same chapter only; reset at chapter boundary (FR-7.4). Correct for story collections; revisit for continuous novels. |
| **Markup destruction by NMT** | Medium | Sentinel `⟦DNTn⟧` tokens instead of inline HTML tags (FR-2.4). |
| **Reasoning-model empty output** | Medium | `think: false` default for Ollama reasoning models; empty-output refusal with actionable error (FR-7.2). |
| **Paragraph misalignment** | High | Numbered `[Pn]` contract + count-mismatch refusal at parse and orchestration layers (FR-7.2). |
| **Roman-numeral mistranslation** | Low | Verbatim bypass + reattach (FR-3.1). `I.` no longer becomes `ಐ.` |
| **Cloud data egress (editor)** | Medium | Default provider is local Ollama; cloud providers require explicit config + key (Section 4). |

---

## 7. Operational Roadmap

1. **Phase 1: Parsing & Masking Engine (Weeks 1–2)** — *Partially done.* EPUB unzipper + minimal chapter/paragraph parser + sentinel/DNT masking + heuristic glossary mining done. Still to do: full DOM/AST preservation (FR-1.3), spaCy NER upgrade.
2. **Phase 2: Translation & Table Engine Integration (Weeks 3–4)** — *Translation done, tables deferred.* CTranslate2 IndicTrans2 with sentence/chunk hardening + Roman-numeral handling done. Still to do: cell-by-cell table pipeline (FR-3.2).
3. **Phase 3: QA & Back-Translation Loop (Weeks 5–6)** — *Deferred, interim mitigations in place.* Still to do: `kn→en` back-translation path (needs Indic sentence splitter), embedding similarity + `qa_report.json` (FR-4).
4. **Phase 4: CSS/Font Injection & EPUB Packaging (Weeks 7–8)** — *Deferred.* Still to do: font embedding, line-height auto-tuning, `content.opf` + `epubcheck` (FR-5).
5. **Phase 5: Glossary TM + Consistency Editing (NEW — implemented).** SQLite store, CSV review loop, forced substitution, provider-abstracted LLM edit with paragraph-integrity guarantees, chapter-scoped rolling context (FR-6/FR-7).
6. **Phase 6: Emotion Tagging + TTS Audiobook (NEW — implemented).** Fixed emotion set, chunked Parler-TTS synthesis, peak normalization, single-WAV assembly (FR-8/FR-9).
7. **Phase 7: Proofing & Regression Harness (NEW — implemented).** EN/KN sample export, proof HTML/PDF, `test_*.py` scripts (FR-10).
8. **Next:** FR-4 QA loop → FR-3.2 tables → FR-5 packaging, in that order (packaging last so it wraps verified content).

---

## Appendix A — Current Module Map (informative, v2.0)

```
src/kannada_epub/
  epub_io.py            FR-1 minimal (spine-order chapters -> paragraphs)
  translation/engine.py FR-3.1 (IndicTrans2 CTranslate2, sentence/chunk hardened)
  glossary/             FR-6 + FR-2.4 (extraction, store.sqlite, review CSV, substitution/masking)
  consistency_editor.py FR-7 + FR-8 (numbered-para edit + emotion tag)
  providers/            FR-7.3 (base, factory, ollama, openai_compatible, anthropic)
  book_translator.py    FR-7.4 (batch 20, context tail 2, chapter-boundary reset)
  tts/engine.py         FR-9.1/9.2 (chunked Parler-TTS, Kannada splitter, peak norm)
  audiobook_builder.py  FR-9.3 (paragraph-gap concatenation to WAV)
  config.py             ProviderConfig loader (FR-7.3)
scripts/                translate_sample_chapter, build_audiobook, build_proof_pdf,
                        download_tts_model, test_{translation,glossary,consistency_editor,
                        book_translator,tts,tts_voices}
config/consistency_editor.example.yaml   provider/model/temperature/max_tokens/think
data/                   sherlock_holmes.epub, scandal_in_bohemia_sample.txt,
                        scandal_in_bohemia_proof.{html,pdf}, project.glossary.db,
                        scandal_in_bohemia_audiobook.wav, tts_voices/
models/                 en-indic-1b-ct2/, indic-parler-tts/
```
