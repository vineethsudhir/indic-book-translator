import logging
import re
from dataclasses import dataclass
from typing import Literal, Optional

import httpx

from .consistency_editor import ConsistencyEditor, EditedParagraph, EditorOutputError
from .epub_io import Chapter
from .glossary import GlossaryStore
from .providers.base import OutputTruncatedError
from .translation import TranslationProvider

_log = logging.getLogger(__name__)

# ``Chapter.title`` values that identify front matter rather than a story or
# chapter. Such chapters are ignored when deciding the continuity mode.
_FRONT_MATTER_RE = re.compile(
    r"^(?:the\s+)?(?:preface|introduction|table\s+of\s+contents|contents|"
    r"dedication|front\s+matter)\b",
    re.IGNORECASE,
)
# "Chapter 5", "CHAPTER XLII THE DRAWERS OF WATER", "Ch. 7 …" — a chapter
# marker followed by a number or roman numeral. Text may follow on the same
# line ("… THE DRAWERS OF WATER"), which named-story titles ("I. A SCANDAL IN
# BOHEMIA") do not have: there the roman numeral is separated by a period and
# is not the word "chapter".
_NUMBERED_CHAPTER_RE = re.compile(
    r"^ch(?:apter|\.)\s*(?:\d+|[ivxlcdm]+)\b", re.IGNORECASE
)
# A bare heading that is only a number or roman numeral, e.g. "IV" or "3.".
_BARE_NUMBER_RE = re.compile(r"^(?:\d+|[ivxlcdm]+)\.?$", re.IGNORECASE)

# A strictly valid roman numeral, so words made only of numeral letters
# ("CIVIL", "DID") are not mistaken for one.
_ROMAN_RE = r"M{0,4}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})"
# Paragraphs with nothing to translate: digits, punctuation and at most one
# roman numeral ("7.", "II", "— 12 —", "XIV."). Models invent content for
# these ("7." came back as "happened on the 7th"), so they skip the engine
# and the editor and are copied verbatim.
_NOTHING_TO_TRANSLATE_RE = re.compile(
    rf"[\W\d_]*(?:(?:{_ROMAN_RE})|(?:{_ROMAN_RE.lower()}))?[\W\d_]*"
)


def needs_translation(text: str) -> bool:
    """False for paragraphs that are only numbers, numerals or punctuation."""
    stripped = text.strip()
    return bool(stripped) and _NOTHING_TO_TRANSLATE_RE.fullmatch(stripped) is None


# Errors after which re-running the editor on a smaller slice can succeed:
# the model ran out of output tokens, the HTTP call timed out, or the reply
# came back unusable (empty / wrong paragraph count). Unrelated RuntimeErrors
# are deliberately not caught and propagate unchanged.
_RETRYABLE_EDITOR_ERRORS = (OutputTruncatedError, httpx.TimeoutException, EditorOutputError)


@dataclass
class TranslatedBatch:
    chapter_id: str
    chapter_title: Optional[str]
    paragraph_start: int
    paragraph_end: int
    source_english: list[str]
    draft_kannada: list[str]
    edited_kannada: list[str]
    edited_emotions: list[str]
    prior_context_used: str


def chapter_context_tail(chapter: Chapter, context_tail_paragraphs: int) -> str:
    """The English tail of ``chapter`` used as the next chapter's context.

    Mirrors the rolling context within a chapter (the last
    ``context_tail_paragraphs`` English source paragraphs, joined by spaces),
    but taken from the whole chapter so a resumed run computes the same value
    as an uninterrupted one.
    """
    if context_tail_paragraphs <= 0:
        return ""
    return " ".join(p.text for p in chapter.paragraphs[-context_tail_paragraphs:])


def _looks_like_chapter_heading(text: str) -> bool:
    """True for "Chapter N"/"Ch. N"/"IV"-shaped headings, not story titles."""
    text = text.strip()
    if not text:
        return False
    return bool(_NUMBERED_CHAPTER_RE.match(text) or _BARE_NUMBER_RE.match(text))


def detect_chapter_context(chapters: list[Chapter]) -> Literal["carry", "reset"]:
    """Guess whether ``chapters`` are parts of one story or separate stories.

    A continuous novel carries context across chapter boundaries; a
    short-story collection resets it at each one. The guess looks at the
    chapter headings, so it stays cheap and explainable:

    - Front matter (a title starting with "preface", "introduction", "table of
      contents", "contents", "dedication" or "front matter") is ignored.
    - A chapter counts as numbered when its title looks like "Chapter 5",
      "Ch. 7", "CHAPTER XLII …", or is only a number/roman numeral ("IV").
      A descriptive title whose document has no heading of its own (the loader
      falls back to the TOC label, e.g. "The Drawers of Water") may instead
      start with its own leading heading paragraph, so that paragraph is
      checked too. An untitled chapter has neither, so it never counts.
    - With fewer than 3 remaining chapters there isn't enough signal: reset.
    - Otherwise the book carries when at least 60% of the remaining chapters
      are numbered, and resets when most are named stories (e.g. "I. A SCANDAL
      IN BOHEMIA").
    """
    remaining = 0
    numbered = 0
    for chapter in chapters:
        title = (chapter.title or "").strip()
        if title and _FRONT_MATTER_RE.match(title):
            continue
        remaining += 1
        if _looks_like_chapter_heading(title):
            numbered += 1
        elif title and chapter.paragraphs and _looks_like_chapter_heading(
            chapter.paragraphs[0].text
        ):
            numbered += 1
    if remaining < 3:
        return "reset"
    return "carry" if numbered / remaining >= 0.6 else "reset"


class BookTranslator:
    """Orchestrates EPUB -> IndicTrans2 -> consistency-edit across a whole book.

    Within a chapter, batches of `batch_size` paragraphs are translated
    together, and a rolling context (the tail of the previous batch's English
    source) carries forward to the next batch of the SAME chapter.

    The boundary *between* chapters is decided by the caller, not by this
    class. ``translate_chapters`` accepts an ``incoming_context`` that the
    first batch of the first chapter in the call sees; every later chapter in
    the same call resets to empty. Passing one chapter at a time (as the
    pipeline does) lets the caller choose per book whether context carries
    across chapter boundaries — carrying it for a continuous novel, and
    resetting it for a short-story collection, where crossing a story boundary
    would leak one plot's pronouns into an unrelated one and cause the
    consistency editor to "fix" a reference using context that doesn't apply.
    See ``detect_chapter_context`` for the automatic choice.

    The consistency editor also assigns each paragraph an emotion tag (see
    consistency_editor.EMOTIONS) for downstream TTS narration — one model
    call does both jobs, no separate classification pass needed.

    If the editor can't handle a whole batch (truncated output, HTTP timeout,
    empty/unparseable reply), the batch is edited in halves, recursively down
    to single paragraphs, without redoing the draft translation. The final
    result is still one `TranslatedBatch` for the original span.
    """

    def __init__(
        self,
        translation_engine: TranslationProvider,
        glossary_store: GlossaryStore,
        consistency_editor: ConsistencyEditor,
        batch_size: int = 20,
        context_tail_paragraphs: int = 2,
        register: str = "neutral, standard written Kannada",
    ):
        self._engine = translation_engine
        self._glossary = glossary_store
        self._editor = consistency_editor
        self._batch_size = batch_size
        self._context_tail = context_tail_paragraphs
        self._register = register

    def _edit_with_split(
        self,
        english_texts: list[str],
        draft_kn: list[str],
        glossary: dict[str, str],
        incoming_context: str,
        chapter_id: str,
        span_start: int,
    ) -> list[EditedParagraph]:
        """Consistency-edit one slice of a batch, halving it on retryable errors.

        Returns one `EditedParagraph` per input paragraph, concatenated in
        order across any splits. If a single-paragraph edit still fails with a
        retryable error, that error propagates: the book fails rather than
        risking a misalignment.
        """
        draft_text = "\n\n".join(draft_kn)
        try:
            return self._editor.edit_chapter(
                draft_kannada_text=draft_text,
                glossary=glossary,
                prior_chapter_context=incoming_context,
                register=self._register,
            )
        except _RETRYABLE_EDITOR_ERRORS as exc:
            if len(english_texts) <= 1:
                raise
            mid = len(english_texts) // 2
            _log.warning(
                "Consistency editor failed for chapter %r paragraphs [%d:%d] (%s: %s); "
                "splitting into [%d:%d] and [%d:%d] and retrying.",
                chapter_id,
                span_start,
                span_start + len(english_texts),
                type(exc).__name__,
                exc,
                span_start,
                span_start + mid,
                span_start + mid,
                span_start + len(english_texts),
            )
            left = self._edit_with_split(
                english_texts[:mid],
                draft_kn[:mid],
                glossary,
                incoming_context,
                chapter_id,
                span_start,
            )
            # The second half continues the first half's English source, the
            # same rolling-context rule that carries between whole batches.
            right_context = " ".join(english_texts[:mid][-self._context_tail :])
            right = self._edit_with_split(
                english_texts[mid:],
                draft_kn[mid:],
                glossary,
                right_context,
                chapter_id,
                span_start + mid,
            )
            return left + right

    def translate_chapters(
        self, chapters: list[Chapter], *, incoming_context: str = ""
    ) -> list[TranslatedBatch]:
        """Translate ``chapters`` in order, returning one batch per span.

        ``incoming_context`` is the English tail of whatever came before the
        first chapter (the pipeline's carried context); it seeds the rolling
        context of the first batch only. Every later chapter in this call
        resets to empty, preserving the collection behaviour for callers that
        pass several chapters at once. Callers that want cross-chapter
        continuity should pass one chapter per call and thread the context
        themselves (see ``chapter_context_tail``).
        """
        results: list[TranslatedBatch] = []
        for index, chapter in enumerate(chapters):
            # A later chapter is a fresh boundary; the first chapter may
            # continue context supplied by the caller.
            rolling_context = incoming_context if index == 0 else ""
            paragraphs = chapter.paragraphs
            for start in range(0, len(paragraphs), self._batch_size):
                batch = paragraphs[start : start + self._batch_size]
                english_texts = [p.text for p in batch]

                # Only paragraphs with words go to the engine and the editor;
                # the rest are copied verbatim and merged back by position.
                wanted = [i for i, text in enumerate(english_texts) if needs_translation(text)]
                draft_kn = list(english_texts)
                edited_paragraphs = [EditedParagraph(emotion="Narration", text=text) for text in english_texts]
                if wanted:
                    wanted_texts = [english_texts[i] for i in wanted]
                    wanted_draft = self._engine.translate_paragraphs(wanted_texts, "eng_Latn", "kan_Knda")
                    if len(wanted_draft) != len(wanted_texts):
                        raise RuntimeError(
                            f"Translation engine returned {len(wanted_draft)} paragraphs for "
                            f"{len(wanted_texts)} (chapter {chapter.id!r}, paragraphs "
                            f"[{start}:{start + len(batch)}]) — refusing to misalign."
                        )
                    relevant_glossary = self._glossary.get_relevant_glossary(" ".join(wanted_texts))
                    wanted_edited = self._edit_with_split(
                        english_texts=wanted_texts,
                        draft_kn=wanted_draft,
                        glossary=relevant_glossary,
                        incoming_context=rolling_context,
                        chapter_id=chapter.id,
                        span_start=start,
                    )
                    if len(wanted_edited) != len(wanted_texts):
                        raise RuntimeError(
                            f"Consistency editor returned {len(wanted_edited)} paragraphs for "
                            f"{len(wanted_texts)} (chapter {chapter.id!r}, paragraphs "
                            f"[{start}:{start + len(batch)}]) — refusing to misalign."
                        )
                    for position, index in enumerate(wanted):
                        draft_kn[index] = wanted_draft[position]
                        edited_paragraphs[index] = wanted_edited[position]

                if len(edited_paragraphs) != len(batch):
                    raise RuntimeError(
                        f"Consistency editor returned {len(edited_paragraphs)} paragraphs for a "
                        f"{len(batch)}-paragraph batch (chapter {chapter.id!r}, paragraphs "
                        f"[{start}:{start + len(batch)}]) — refusing to misalign edited text/emotions "
                        f"against the wrong source paragraphs."
                    )

                results.append(
                    TranslatedBatch(
                        chapter_id=chapter.id,
                        chapter_title=chapter.title,
                        paragraph_start=start,
                        paragraph_end=start + len(batch),
                        source_english=english_texts,
                        draft_kannada=draft_kn,
                        edited_kannada=[p.text for p in edited_paragraphs],
                        edited_emotions=[p.emotion for p in edited_paragraphs],
                        prior_context_used=rolling_context,
                    )
                )

                rolling_context = " ".join(t.text for t in batch[-self._context_tail :])

        return results
