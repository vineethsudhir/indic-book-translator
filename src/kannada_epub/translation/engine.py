import re
from pathlib import Path

import ctranslate2
import sentencepiece as spm
from IndicTransToolkit.processor import IndicProcessor

# Best-effort — English only. Indic-language sentence splitting (needed for the
# future kn->en back-translation direction) isn't implemented yet; the model
# was trained/evaluated on sentence-level input, and translating whole
# multi-sentence paragraphs as a single unit is what caused real content loss
# in testing (see translate_paragraphs docstring).
_ENGLISH_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str, lang: str) -> list[str]:
    if lang != "eng_Latn":
        return [text]
    sentences = [s.strip() for s in _ENGLISH_SENTENCE_SPLIT_RE.split(text.strip())]
    return [s for s in sentences if s]


# Matches a leading, syntactically-valid Roman numeral immediately followed by
# a period — chapter/section markers like "I." or "III. THE RED-HEADED LEAGUE".
# Deliberately NOT a blanket "mask every I/V/X/L/C/D/M" rule: the pronoun "I"
# is constant in first-person prose and must translate normally. What makes
# this pattern safe is requiring the numeral to be immediately followed by a
# period as the very first token — ordinary prose never has "I" directly
# followed by a period with no verb in between ("I. have seen..." doesn't
# happen), so the false-positive rate on real sentences is effectively zero.
_ROMAN_NUMERAL_HEADING_RE = re.compile(
    r"^(M{0,4}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3}))\.(?:\s+(.*))?$"
)


def _split_roman_numeral_heading(text: str) -> tuple[str, str] | None:
    """If `text` is a Roman-numeral chapter/section marker, return
    (numeral_with_period, remainder) so the numeral can bypass translation
    entirely and the remainder (if any) is translated normally. Returns None
    for ordinary prose."""
    m = _ROMAN_NUMERAL_HEADING_RE.match(text.strip())
    if not m or not m.group(1):
        return None
    return f"{m.group(1)}.", (m.group(2) or "").strip()


class IndicTrans2Engine:
    """Batch translator over a CTranslate2-converted IndicTrans2 checkpoint.

    IndicTrans2's vocabulary registers FLORES-style language tags ("eng_Latn",
    "kan_Knda", ...) as atomic entries, separate from the SentencePiece
    subword vocabulary — running the tagged string through SentencePiece
    directly shatters the tags into garbage subword pieces (verified: "kan_Knda"
    -> ['▁k', 'an', '_', 'K', 'nda']). So SentencePiece is applied to sentence
    content only, and the tags are prepended afterward as separate raw tokens,
    matching how the model's own vocabulary and AI4Bharat's reference
    ctranslate2 inference engine do it.

    IndicProcessor.preprocess_batch()/postprocess_batch() are stateful — each
    preprocess call queues per-sentence placeholder maps that the matching
    postprocess call consumes in order. translate() always pairs them 1:1
    with nothing interleaved, but this class is not safe to call concurrently
    from multiple threads on the same instance.

    Any single sentence whose SentencePiece encoding still exceeds
    max_content_tokens (e.g. one pathological run-on) is chunked and
    translated in pieces rather than silently truncated by CTranslate2 —
    verified in testing: at max_input_length=160 with no chunking, two long
    paragraphs lost their final 1-2 sentences with no error or warning.
    """

    def __init__(
        self,
        ct2_model_dir: str | Path,
        spm_src_path: str | Path,
        spm_tgt_path: str | Path,
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 5,
        max_content_tokens: int = 200,
        max_decoding_length: int = 256,
    ):
        self._processor = IndicProcessor(inference=True)
        self._sp_src = spm.SentencePieceProcessor(model_file=str(spm_src_path))
        self._sp_tgt = spm.SentencePieceProcessor(model_file=str(spm_tgt_path))
        self._translator = ctranslate2.Translator(str(ct2_model_dir), device=device, compute_type=compute_type)
        self._beam_size = beam_size
        self._max_content_tokens = max_content_tokens
        # +8 tag/margin headroom over max_content_tokens: our own chunking is
        # the real limit, this is just a backstop so CTranslate2 never has to
        # truncate a chunk we already sized correctly.
        self._max_input_length = max_content_tokens + 8
        self._max_decoding_length = max_decoding_length

    def translate(self, sentences: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        """Translate a batch of already-segmented sentences. `src_lang`/`tgt_lang`
        are FLORES-200 codes, e.g. "eng_Latn" / "kan_Knda"."""
        if not sentences:
            return []

        # is_target=True suppresses tag prepending so we get plain normalized/
        # tokenized/transliterated text to feed through SentencePiece ourselves.
        processed = self._processor.preprocess_batch(sentences, src_lang, tgt_lang, is_target=True)
        encoded = [self._sp_src.encode(s, out_type=str) for s in processed]

        # Chunk any sentence whose encoding is still too long, tracking which
        # original sentence each chunk belongs to so translated chunks can be
        # merged back before postprocessing.
        chunk_owner: list[int] = []
        chunks: list[list[str]] = []
        for i, pieces in enumerate(encoded):
            for start in range(0, max(len(pieces), 1), self._max_content_tokens):
                chunks.append(pieces[start : start + self._max_content_tokens])
                chunk_owner.append(i)

        tokenized = [[src_lang, tgt_lang, *chunk] for chunk in chunks]
        results = self._translator.translate_batch(
            tokenized,
            max_batch_size=2048,
            batch_type="tokens",
            max_input_length=self._max_input_length,
            max_decoding_length=self._max_decoding_length,
            beam_size=self._beam_size,
        )
        translated_pieces = [" ".join(r.hypotheses[0]) for r in results]
        detokenized_chunks = [p.replace(" ", "").replace("▁", " ").strip() for p in translated_pieces]

        merged = [""] * len(sentences)
        for owner, text in zip(chunk_owner, detokenized_chunks):
            merged[owner] = f"{merged[owner]} {text}".strip() if merged[owner] else text

        return self._processor.postprocess_batch(merged, lang=tgt_lang)

    def translate_paragraphs(self, paragraphs: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        """Translate paragraph-level text by splitting into sentences first —
        matching how IndicTrans2 was trained/evaluated — and rejoining after
        translation, rather than feeding a whole multi-sentence paragraph
        through as one unit (which also degrades quality independent of the
        truncation risk: the model attends over one long blob instead of
        clean per-sentence input).

        Roman-numeral chapter/section markers ("I.", "III. THE RED-HEADED
        LEAGUE") are stripped before translation and reattached verbatim
        afterward — verified in testing: fed through untouched, the model
        has no way to distinguish the numeral "I" from the pronoun "I" and
        transliterates it as if it were the word ("I." -> "ಐ.", a Kannada
        vowel sound, instead of staying "I.").
        """
        heading_numerals: dict[int, str] = {}
        texts_to_translate: list[str] = []
        for para in paragraphs:
            split = _split_roman_numeral_heading(para) if src_lang == "eng_Latn" else None
            if split is not None:
                numeral, remainder = split
                heading_numerals[len(texts_to_translate)] = numeral
                texts_to_translate.append(remainder)
            else:
                texts_to_translate.append(para)

        all_sentences: list[str] = []
        owner: list[int] = []
        for i, text in enumerate(texts_to_translate):
            for sent in _split_sentences(text, src_lang):
                all_sentences.append(sent)
                owner.append(i)

        translated = self.translate(all_sentences, src_lang, tgt_lang)

        merged = [""] * len(paragraphs)
        for idx, text in zip(owner, translated):
            merged[idx] = f"{merged[idx]} {text}".strip() if merged[idx] else text

        for idx, numeral in heading_numerals.items():
            merged[idx] = f"{numeral} {merged[idx]}".strip() if merged[idx] else numeral

        return merged