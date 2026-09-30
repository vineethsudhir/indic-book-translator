"""Target languages the translator can translate into.

IndicTrans2 covers many Indic languages; Kannada is the default and the only
one with a bundled font today. Each :class:`TargetLanguage` carries the values
the rest of the codebase hard-coded for Kannada: the FLORES-200 code the
translation engine and QA back-translator use, the BCP-47 code TTS uses, the
book's HTML/XML ``lang`` and the output file-name suffix, the bundled font
(family and files), a regular expression matching the script, and the cover's
machine-translation label.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TargetLanguage:
    key: str  # "kn" — also the html/xml lang and the file-name suffix
    name: str  # "Kannada"
    flores: str  # "kan_Knda"
    bcp47: str  # "kn-IN" (TTS)
    font_family: str  # "Noto Sans Kannada"
    font_regular: str  # "NotoSansKannada-Regular.ttf"
    font_bold: str  # "NotoSansKannada-Bold.ttf"
    script_re: str  # character class for the script, e.g. "[ಀ-೿]"
    machine_translation_label: str  # "ಯಂತ್ರ ಅನುವಾದ"


LANGUAGES: dict[str, TargetLanguage] = {
    "kn": TargetLanguage(
        key="kn",
        name="Kannada",
        flores="kan_Knda",
        bcp47="kn-IN",
        font_family="Noto Sans Kannada",
        font_regular="NotoSansKannada-Regular.ttf",
        font_bold="NotoSansKannada-Bold.ttf",
        script_re=r"[\u0C80-\u0CFF]",
        machine_translation_label="ಯಂತ್ರ ಅನುವಾದ",
    ),
    "ta": TargetLanguage(
        key="ta",
        name="Tamil",
        flores="tam_Taml",
        bcp47="ta-IN",
        font_family="Noto Sans Tamil",
        font_regular="NotoSansTamil-Regular.ttf",
        font_bold="NotoSansTamil-Bold.ttf",
        script_re=r"[\u0B80-\u0BFF]",
        machine_translation_label="இயந்திர மொழிபெயர்ப்பு",
    ),
    "te": TargetLanguage(
        key="te",
        name="Telugu",
        flores="tel_Telu",
        bcp47="te-IN",
        font_family="Noto Sans Telugu",
        font_regular="NotoSansTelugu-Regular.ttf",
        font_bold="NotoSansTelugu-Bold.ttf",
        script_re=r"[\u0C00-\u0C7F]",
        machine_translation_label="యంత్ర అనువాదం",
    ),
    "ml": TargetLanguage(
        key="ml",
        name="Malayalam",
        flores="mal_Mlym",
        bcp47="ml-IN",
        font_family="Noto Sans Malayalam",
        font_regular="NotoSansMalayalam-Regular.ttf",
        font_bold="NotoSansMalayalam-Bold.ttf",
        script_re=r"[\u0D00-\u0D7F]",
        machine_translation_label="യന്ത്ര വിവർത്തനം",
    ),
    "hi": TargetLanguage(
        key="hi",
        name="Hindi",
        flores="hin_Deva",
        bcp47="hi-IN",
        font_family="Noto Sans Devanagari",
        font_regular="NotoSansDevanagari-Regular.ttf",
        font_bold="NotoSansDevanagari-Bold.ttf",
        script_re=r"[\u0900-\u097F]",
        machine_translation_label="मशीनी अनुवाद",
    ),
}


def get_language(key: str) -> TargetLanguage:
    """The :class:`TargetLanguage` for ``key``; ``ValueError`` if unknown."""
    try:
        return LANGUAGES[key]
    except KeyError:
        raise ValueError(
            f"Unknown target language {key!r}; choose one of "
            f"{', '.join(sorted(LANGUAGES))}."
        ) from None
