import logging
from dataclasses import dataclass
from typing import Optional

import httpx

from .consistency_editor import ConsistencyEditor, EditedParagraph, EditorOutputError
from .epub_io import Chapter
from .glossary import GlossaryStore
from .providers.base import OutputTruncatedError
from .translation import TranslationProvider

_log = logging.getLogger(__name__)

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


class BookTranslator:
    """Orchestrates EPUB -> IndicTrans2 -> consistency-edit across a whole book.

    Continuity boundary = EPUB chapter (spine item). Within a chapter, batches
    of `batch_size` paragraphs are translated together, and a rolling context
    (the tail of the previous batch's English source) carries forward to the
    next batch of the SAME chapter. That rolling context resets to empty at
    every chapter boundary.

    This is the right call for a short-story collection, where each EPUB
    chapter is an independent story: carrying context across chapters would
    leak one story's plot into an unrelated one and could cause the
    consistency editor to "fix" a pronoun or reference using context that
    doesn't actually apply — a wrong-context bug, not just a missing-context
    one. A single continuous novel split into chapters would want the
    opposite (context carried across chapters); that's a different book
    shape and not what this class assumes — chapter boundaries here are
    treated as story boundaries because that's what they are in this book.

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

    def translate_chapters(self, chapters: list[Chapter]) -> list[TranslatedBatch]:
        results: list[TranslatedBatch] = []
        for chapter in chapters:
            rolling_context = ""  # reset at every chapter (= story) boundary
            paragraphs = chapter.paragraphs
            for start in range(0, len(paragraphs), self._batch_size):
                batch = paragraphs[start : start + self._batch_size]
                english_texts = [p.text for p in batch]

                draft_kn = self._engine.translate_paragraphs(english_texts, "eng_Latn", "kan_Knda")

                relevant_glossary = self._glossary.get_relevant_glossary(" ".join(english_texts))
                edited_paragraphs = self._edit_with_split(
                    english_texts=english_texts,
                    draft_kn=draft_kn,
                    glossary=relevant_glossary,
                    incoming_context=rolling_context,
                    chapter_id=chapter.id,
                    span_start=start,
                )

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
