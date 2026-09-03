import re
from collections import Counter

from .models import Candidate

ACRONYM_RE = re.compile(r"\b[A-Z]{2,6}\b")
CAP_PHRASE_RE = re.compile(r"\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)+\b")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
WORD_RE = re.compile(r"[A-Za-z]+")

_ACRONYM_STOPWORDS = {"I", "A"}

# Capitalized only because they can start a sentence, not because they're part
# of a name/term — stripped from the front of a matched phrase before it's
# counted, so "The Thread Scheduler" (sentence-initial) and "Thread Scheduler"
# (mid-sentence) collapse into the same key instead of splitting the count.
_LEADING_STOPWORDS = {
    "The", "A", "An", "This", "That", "These", "Those", "His", "Her", "Its",
    "Their", "He", "She", "It", "They", "We", "You", "I", "But", "And", "So",
    "Every", "All", "Some", "My", "Your", "Our", "There", "Then", "When",
    "While", "If", "As",
}


def _strip_leading_stopword(phrase: str) -> str:
    first, _, rest = phrase.partition(" ")
    return rest if first in _LEADING_STOPWORDS and rest else phrase


def extract_candidates(text: str, chapter_id: str | None = None, min_frequency: int = 3) -> list[Candidate]:
    """Heuristic (regex/frequency) candidate mining for glossary review — no ML
    dependency, tuned for recall over precision since a human reviews every
    candidate before it's locked in (see review.py). Swap this out for spaCy
    NER once the masking stage (FR-2) is built; downstream code only depends
    on the list[Candidate] shape, not how it was produced.
    """
    candidates: list[Candidate] = []

    acronyms = Counter(m for m in ACRONYM_RE.findall(text) if m not in _ACRONYM_STOPWORDS)
    for term, freq in acronyms.items():
        if freq >= min_frequency:
            candidates.append(Candidate(term, "acronym", freq, chapter_id))

    phrases = Counter(_strip_leading_stopword(m) for m in CAP_PHRASE_RE.findall(text))
    accepted_phrase_words: set[str] = set()
    for term, freq in phrases.items():
        if freq >= min_frequency:
            candidates.append(Candidate(term, "proper_noun_phrase", freq, chapter_id))
            accepted_phrase_words.update(term.split())

    # Single capitalized words repeated in non-sentence-initial position — the
    # common shape of a character name ("Meera") that the phrase regex above
    # would miss on its own, without flagging every ordinary sentence-initial
    # capital. Words already covered by an accepted phrase are skipped so
    # "Thread Scheduler" doesn't also surface "Thread" and "Scheduler" alone.
    single_words = Counter()
    for sentence in SENTENCE_SPLIT_RE.split(text):
        words = WORD_RE.findall(sentence)
        for word in words[1:]:
            if word[0].isupper() and not word.isupper() and len(word) > 1 and word not in accepted_phrase_words:
                single_words[word] += 1
    for term, freq in single_words.items():
        if freq >= min_frequency:
            candidates.append(Candidate(term, "repeated_capitalized", freq, chapter_id))

    return candidates