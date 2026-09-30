from abc import ABC, abstractmethod


class TranslationProvider(ABC):
    """A paragraph-level EN->KN translation backend.

    Implementations wrap a specific engine (local CTranslate2 IndicTrans2,
    OpenAI-compatible cloud chat, ...). BookTranslator only depends on this
    interface, so swapping engines never touches pipeline code.
    """

    #: Human-readable summary of how a retry differs from a normal call. Shown
    #: in progress output so a user can tell why a retried paragraph changed.
    retry_description: str = "same settings"

    @abstractmethod
    def translate_paragraphs(
        self, paragraphs: list[str], src_lang: str, tgt_lang: str
    ) -> list[str]:
        """Translate paragraph-level text. `src_lang`/`tgt_lang` are
        FLORES-200 codes, e.g. "eng_Latn" / "kan_Knda". Must return exactly
        one output per input paragraph, in order."""
        raise NotImplementedError

    def translate_paragraphs_retry(
        self, paragraphs: list[str], src_lang: str, tgt_lang: str
    ) -> list[str]:
        """Re-translate paragraphs that scored low in QA, varying the decoding
        setup so a deterministic engine does not just reproduce the same
        output. Same contract as :meth:`translate_paragraphs`: exactly one
        output per input, in order. The default repeats the normal call for
        engines that cannot vary anything.
        """
        return self.translate_paragraphs(paragraphs, src_lang, tgt_lang)
