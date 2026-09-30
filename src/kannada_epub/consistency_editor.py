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

SYSTEM_PROMPT_TEMPLATE = """You are a {name}-language consistency editor working on one chapter of a book \
that has already been machine-translated from English into {name}.

You are NOT translating from scratch. You are given a draft {name} chapter and must return an \
edited version that only fixes:
1. Glossary terms — every term in GLOSSARY must appear exactly as given wherever its English \
   source term occurs, replacing whatever the draft used instead.
2. Pronoun / referent consistency — resolve ambiguous or inconsistent pronouns and character \
   references using PRIOR_CHAPTER_CONTEXT.
3. Register — keep the tone consistent with REGISTER across the whole chapter.
4. Headings and titles — paragraphs listed under HEADINGS are headings or titles. Translate their \
   meaning into natural {name} instead of transliterating English words into {name} script. For \
   example, "A Scandal in Bohemia" should become a {name} phrase meaning "a scandal in Bohemia", \
   with only the place name "Bohemia" transliterated. Names of people and places may stay \
   transliterated.

Do not rewrite sentences that are already correct. Do not change meaning, add content, or remove \
content.

DRAFT_CHAPTER's paragraphs are each numbered with a [P<n>] tag, e.g. "[P1]", "[P2]". Your output \
MUST have exactly the same number of paragraphs, in the same order, each carrying the same [P<n>] \
tag it had in the input. This holds even when DRAFT_CHAPTER has only one paragraph, and even when \
a paragraph is long or contains many sentences: one input paragraph always produces exactly one \
output entry. Never split one input paragraph into several output entries and never merge several \
input paragraphs into one, regardless of how much the tone or subject varies within a paragraph.

Additionally, classify the emotional tone each paragraph should be narrated in for an audiobook \
reading. Choose exactly one tag per paragraph from this fixed set: {emotions}. \
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


def system_prompt_for(language_name: str) -> str:
    """The consistency editor's system prompt for a target language name."""
    return SYSTEM_PROMPT_TEMPLATE.format(
        name=language_name, emotions=", ".join(EMOTIONS)
    )


SYSTEM_PROMPT = system_prompt_for("Kannada")


# Appended to SYSTEM_PROMPT only when the caller preserves inline markup
# (FR-1.3), so prompts for plain runs stay byte-identical to before.
INLINE_MARKUP_TEMPLATE = """

This chapter was translated from a source with inline formatting. Words that were bold, italic, a \
link or a footnote reference in the source are wrapped in numbered markers: ⟦1⟧ … ⟦/1⟧, ⟦2⟧ … \
⟦/2⟧, and so on. Keep every marker pair around the {name} words that translate the marked English \
words. Never add, remove or renumber markers, never move a marker onto different words, and never \
let one marker pair cross another."""


def inline_markup_rule_for(language_name: str) -> str:
    """The inline-markup rule for a target language name."""
    return INLINE_MARKUP_TEMPLATE.format(name=language_name)


INLINE_MARKUP_RULE = inline_markup_rule_for("Kannada")


@dataclass
class EditedParagraph:
    emotion: str
    text: str


# The EMOTION line is optional: local models sometimes drop it for a single
# paragraph while keeping the [P<n>] tag and text. Alignment comes from the
# tag, so such a paragraph is kept and its emotion defaults to Narration.
_PARAGRAPH_BLOCK_RE = re.compile(
    r"\[P(\d+)\][ \t]*\n(?:\s*EMOTION:[ \t]*([^\n]*?)[ \t]*(?:\n|\Z))?(.*?)(?=\n\s*\[P\d+\]|\Z)",
    re.DOTALL,
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
    heading_numbers: set[int] | None = None,
) -> str:
    glossary_block = "\n".join(f"- {en} -> {kn}" for en, kn in glossary.items()) or "(none)"
    # Omitted entirely when there are no headings, so prompts for such chapters
    # are byte-identical to before this argument existed.
    headings_block = ""
    if heading_numbers:
        numbers = ", ".join(f"P{n}" for n in sorted(heading_numbers))
        headings_block = f"HEADINGS:\n{numbers}\n\n"
    return (
        f"GLOSSARY:\n{glossary_block}\n\n"
        f"{headings_block}"
        f"PRIOR_CHAPTER_CONTEXT:\n{prior_chapter_context or '(none — this is the first chapter)'}\n\n"
        f"REGISTER:\n{register}\n\n"
        f"DRAFT_CHAPTER:\n{_number_paragraphs(draft_kannada_text)}"
    )


class ConsistencyEditor:
    def __init__(
        self, provider: ConsistencyEditorProvider, language_name: str = "Kannada"
    ):
        self._provider = provider
        self._system_prompt = system_prompt_for(language_name)
        self._inline_markup_rule = inline_markup_rule_for(language_name)

    def edit_chapter(
        self,
        draft_kannada_text: str,
        glossary: dict[str, str],
        prior_chapter_context: str = "",
        register: str = "neutral, standard written Kannada",
        heading_numbers: set[int] | None = None,
        preserve_inline_markup: bool = False,
    ) -> list[EditedParagraph]:
        expected_count = len([p for p in draft_kannada_text.split("\n\n") if p.strip()])
        user_prompt = _build_user_prompt(
            draft_kannada_text, glossary, prior_chapter_context, register, heading_numbers
        )
        system_prompt = self._system_prompt + (
            self._inline_markup_rule if preserve_inline_markup else ""
        )
        result = self._provider.complete(system_prompt, user_prompt)
        if not result.strip():
            raise EditorOutputError(
                "Consistency editor returned empty output — refusing to overwrite the draft chapter. "
                "This can happen if a reasoning-capable model exhausts max_tokens before answering; "
                "if using Ollama, keep 'think: false' or raise max_tokens."
            )
        return _parse_numbered_output(result, expected_count)