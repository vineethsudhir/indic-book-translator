import re
from dataclasses import dataclass

from .providers.base import ConsistencyEditorProvider


class EditorOutputError(RuntimeError):
    """The consistency editor produced output that can't be used as-is.

    Raised for an empty reply or a paragraph-number mismatch. Callers can catch
    this precisely to re-run the batch in smaller pieces, while unrelated
    ``RuntimeError``s still propagate unchanged.
    """

# ai4bharat/indic-parler-tts's officially supported emotion tags (see its
# model card). Constraining to this exact set means the tag can be dropped
# straight into a TTS description prompt downstream with no translation step.
EMOTIONS = [
    "Command", "Anger", "Narration", "Conversation", "Disgust", "Fear",
    "Happy", "Neutral", "Proper Noun", "News", "Sad", "Surprise",
]

SYSTEM_PROMPT = """You are a Kannada-language consistency editor working on one chapter of a book \
that has already been machine-translated from English into Kannada.

You are NOT translating from scratch. You are given a draft Kannada chapter and must return an \
edited version that only fixes:
1. Glossary terms — every term in GLOSSARY must appear exactly as given wherever its English \
   source term occurs, replacing whatever the draft used instead.
2. Pronoun / referent consistency — resolve ambiguous or inconsistent pronouns and character \
   references using PRIOR_CHAPTER_CONTEXT.
3. Register — keep the tone consistent with REGISTER across the whole chapter.

Do not rewrite sentences that are already correct. Do not change meaning, add content, or remove \
content.

DRAFT_CHAPTER's paragraphs are each numbered with a [P<n>] tag, e.g. "[P1]", "[P2]". Your output \
MUST have exactly the same number of paragraphs, in the same order, each carrying the same [P<n>] \
tag it had in the input. This holds even when DRAFT_CHAPTER has only one paragraph, and even when \
a paragraph is long or contains many sentences: one input paragraph always produces exactly one \
output entry. Never split one input paragraph into several output entries and never merge several \
input paragraphs into one, regardless of how much the tone or subject varies within a paragraph.

Additionally, classify the emotional tone each paragraph should be narrated in for an audiobook \
reading. Choose exactly one tag per paragraph from this fixed set: """ + ", ".join(EMOTIONS) + """. \
If a paragraph's tone varies internally, pick the single tag that best represents it as a whole — \
do not split it to give different parts different tags. Use "Narration" for ordinary \
descriptive/narrative prose; use a more specific tag only when the paragraph's content clearly \
signals it (e.g. dialogue expressing anger -> "Anger").

Return each paragraph as its [P<n>] tag, then "EMOTION: <tag>" on its own line, then the edited \
paragraph text — no commentary, preamble, or markdown anywhere in your answer. Example shape for \
two paragraphs:

[P1]
EMOTION: Narration
<edited paragraph 1 text>

[P2]
EMOTION: Happy
<edited paragraph 2 text>"""


@dataclass
class EditedParagraph:
    emotion: str
    text: str


_PARAGRAPH_BLOCK_RE = re.compile(
    r"\[P(\d+)\]\s*\n\s*EMOTION:\s*(\S.*?)\s*\n(.*?)(?=\n\s*\[P\d+\]|\Z)", re.DOTALL
)


def _normalize_emotion(tag: str) -> str:
    tag_clean = tag.strip()
    for known in EMOTIONS:
        if tag_clean.lower() == known.lower():
            return known
    return "Narration"  # safe default if the model returns something off-list


def _number_paragraphs(draft_kannada_text: str) -> str:
    paragraphs = [p for p in draft_kannada_text.split("\n\n") if p.strip()]
    return "\n\n".join(f"[P{i}]\n{p.strip()}" for i, p in enumerate(paragraphs, start=1))


def _parse_numbered_output(raw: str, expected_count: int) -> list[EditedParagraph]:
    blocks = _PARAGRAPH_BLOCK_RE.findall(raw.strip())
    by_number: dict[int, EditedParagraph] = {}
    for num_str, tag, text in blocks:
        if text.strip():
            by_number[int(num_str)] = EditedParagraph(emotion=_normalize_emotion(tag), text=text.strip())

    expected_numbers = set(range(1, expected_count + 1))
    got_numbers = set(by_number.keys())
    if got_numbers != expected_numbers:
        missing = sorted(expected_numbers - got_numbers)
        unexpected = sorted(got_numbers - expected_numbers)
        raise EditorOutputError(
            f"Consistency editor output paragraph numbers don't match input — expected "
            f"[P1..P{expected_count}], missing {missing or 'none'}, unexpected {unexpected or 'none'}. "
            f"Raw output started with: {raw[:200]!r}"
        )
    return [by_number[i] for i in range(1, expected_count + 1)]


def _build_user_prompt(
    draft_kannada_text: str,
    glossary: dict[str, str],
    prior_chapter_context: str,
    register: str,
) -> str:
    glossary_block = "\n".join(f"- {en} -> {kn}" for en, kn in glossary.items()) or "(none)"
    return (
        f"GLOSSARY:\n{glossary_block}\n\n"
        f"PRIOR_CHAPTER_CONTEXT:\n{prior_chapter_context or '(none — this is the first chapter)'}\n\n"
        f"REGISTER:\n{register}\n\n"
        f"DRAFT_CHAPTER:\n{_number_paragraphs(draft_kannada_text)}"
    )


class ConsistencyEditor:
    def __init__(self, provider: ConsistencyEditorProvider):
        self._provider = provider

    def edit_chapter(
        self,
        draft_kannada_text: str,
        glossary: dict[str, str],
        prior_chapter_context: str = "",
        register: str = "neutral, standard written Kannada",
    ) -> list[EditedParagraph]:
        expected_count = len([p for p in draft_kannada_text.split("\n\n") if p.strip()])
        user_prompt = _build_user_prompt(draft_kannada_text, glossary, prior_chapter_context, register)
        result = self._provider.complete(SYSTEM_PROMPT, user_prompt)
        if not result.strip():
            raise EditorOutputError(
                "Consistency editor returned empty output — refusing to overwrite the draft chapter. "
                "This can happen if a reasoning-capable model exhausts max_tokens before answering; "
                "if using Ollama, keep 'think: false' or raise max_tokens."
            )
        return _parse_numbered_output(result, expected_count)