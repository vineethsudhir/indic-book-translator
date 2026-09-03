from abc import ABC, abstractmethod


class TranslationProvider(ABC):
    """A paragraph-level EN->KN translation backend.

    Implementations wrap a specific engine (local CTranslate2 IndicTrans2,
    OpenAI-compatible cloud chat, ...). BookTranslator only depends on this
    interface, so swapping engines never touches pipeline code.
    """

    @abstractmethod
    def translate_paragraphs(
        self, paragraphs: list[str], src_lang: str, tgt_lang: str
    ) -> list[str]:
        """Translate paragraph-level text. `src_lang`/`tgt_lang` are
        FLORES-200 codes, e.g. "eng_Latn" / "kan_Knda". Must return exactly
        one output per input paragraph, in order."""
        raise NotImplementedError
