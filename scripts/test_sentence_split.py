"""Smoke test: exercise _split_sentences for the Kannada path (with an
English regression case) directly, no model or network needed."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kannada_epub.translation.engine import _split_sentences

KAN = "kan_Knda"
ENG = "eng_Latn"


def check(text: str, lang: str, expected: list[str]) -> None:
    got = _split_sentences(text, lang)
    assert got == expected, f"{lang} {text!r}\n  expected {expected!r}\n  got      {got!r}"


# Multi-sentence Kannada with periods.
check(
    "ಅವನು ಬಂದನು. ಅವಳು ಹೋದಳು. ನಾವು ಉಳಿದೆವು.",
    KAN,
    ["ಅವನು ಬಂದನು.", "ಅವಳು ಹೋದಳು.", "ನಾವು ಉಳಿದೆವು."],
)

# Question and exclamation marks.
check(
    "ನೀವು ಹೇಗಿದ್ದೀರಿ? ಚೆನ್ನಾಗಿದ್ದೇನೆ! ಧನ್ಯವಾದಗಳು.",
    KAN,
    ["ನೀವು ಹೇಗಿದ್ದೀರಿ?", "ಚೆನ್ನಾಗಿದ್ದೇನೆ!", "ಧನ್ಯವಾದಗಳು."],
)

# Danda and double danda.
check(
    "ಇದು ಒಂದು ವಾಕ್ಯ। ಅದು ಇನ್ನೊಂದು ವಾಕ್ಯ॥ ಮುಂದಿನದು.",
    KAN,
    ["ಇದು ಒಂದು ವಾಕ್ಯ।", "ಅದು ಇನ್ನೊಂದು ವಾಕ್ಯ॥", "ಮುಂದಿನದು."],
)

# Punctuation immediately followed by a closing quote is not a split point:
# the quote closes quoted speech and the reporting verb that follows it must
# stay with the sentence rather than being stranded as its own fragment.
check(
    'ಅವನು "ಹೌದು." ಎಂದನು. ಮುಂದೆ.',
    KAN,
    ['ಅವನು "ಹೌದು." ಎಂದನು.', "ಮುಂದೆ."],
)
check(
    "ಅವನು ಹೋದನು.” ಅವಳು ನಕ್ಕಳು.",
    KAN,
    ["ಅವನು ಹೋದನು.” ಅವಳು ನಕ್ಕಳು."],
)

# No terminal punctuation -> a single sentence.
check("ಇದು ವಿರಾಮ ಚಿಹ್ನೆ ಇಲ್ಲದ ವಾಕ್ಯ", KAN, ["ಇದು ವಿರಾಮ ಚಿಹ್ನೆ ಇಲ್ಲದ ವಾಕ್ಯ"])

# Extra / leading / trailing whitespace is stripped and empty parts dropped.
check("  ಮೊದಲನೇ ವಾಕ್ಯ.   ಎರಡನೇ ವಾಕ್ಯ.  ", KAN, ["ಮೊದಲನೇ ವಾಕ್ಯ.", "ಎರಡನೇ ವಾಕ್ಯ."])
check("   \n  ", KAN, [])

# Other languages still pass through unchanged.
check("ಅವನು ಬಂದನು. ಅವಳು ಹೋದಳು.", "hin_Deva", ["ಅವನು ಬಂದನು. ಅವಳು ಹೋದಳು."])

# English behavior unchanged (regression).
check(
    "Hello there. How are you? Fine!",
    ENG,
    ["Hello there.", "How are you?", "Fine!"],
)

# English abbreviations must not split: the abbreviation merges forward.
check(
    "Dr. Watson arrived. He left.",
    ENG,
    ["Dr. Watson arrived.", "He left."],
)
check(
    "Mr. Sherlock Holmes sat down. Then he left.",
    ENG,
    ["Mr. Sherlock Holmes sat down.", "Then he left."],
)
check(
    "He went to the Church of St. Monica. It was old.",
    ENG,
    ["He went to the Church of St. Monica.", "It was old."],
)
# Initials chain fully rejoins.
check(
    "J. H. Watson wrote this. Read it.",
    ENG,
    ["J. H. Watson wrote this.", "Read it."],
)
# The pronoun "I." is not an abbreviation; the sentence still splits.
check(
    "You and I. Then we left.",
    ENG,
    ["You and I.", "Then we left."],
)
# Ordinary sentence boundaries still split.
check("He left. She stayed.", ENG, ["He left.", "She stayed."])
# "? / !" are not abbreviation terminators and stay split points.
check("Who came? Dr. Watson did.", ENG, ["Who came?", "Dr. Watson did."])
check("He ran fast! She walked.", ENG, ["He ran fast!", "She walked."])

# Kannada abbreviations merge forward too.
check(
    "ಡಾ. ವಾಟ್ಸನ್ ಬಂದರು. ಅವನು ಹೋದನು.",
    KAN,
    ["ಡಾ. ವಾಟ್ಸನ್ ಬಂದರು.", "ಅವನು ಹೋದನು."],
)
check(
    "ಶ್ರೀ. ರಾಮನ್ ಬಂದರು. ಮುಂದೆ.",
    KAN,
    ["ಶ್ರೀ. ರಾಮನ್ ಬಂದರು.", "ಮುಂದೆ."],
)
# Initials chain fully rejoins.
check(
    "ಡಿ. ವಿ. ಗುಂಡಪ್ಪ ಬರೆದರು. ಮುಂದೆ.",
    KAN,
    ["ಡಿ. ವಿ. ಗುಂಡಪ್ಪ ಬರೆದರು.", "ಮುಂದೆ."],
)
# Multi-akshara abbreviation tokens merge too.
check(
    "ಕ್ರಿ.ಶ. 1900ರಲ್ಲಿ ಬಂದನು. ಮುಂದೆ.",
    KAN,
    ["ಕ್ರಿ.ಶ. 1900ರಲ್ಲಿ ಬಂದನು.", "ಮುಂದೆ."],
)
# A normal multi-syllable word ending a sentence still splits.
check(
    "ಅವನು ಬಂದನು. ಅವಳು ಹೋದಳು.",
    KAN,
    ["ಅವನು ಬಂದನು.", "ಅವಳು ಹೋದಳು."],
)

print("test_sentence_split: all assertions passed")
