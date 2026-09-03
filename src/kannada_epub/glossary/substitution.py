import re

# A sentinel token, not an HTML tag: NMT models are trained on plain text and
# can reflow, duplicate, or drop inline markup they weren't trained to treat
# as inviolate. A plain bracketed token survives translation intact and gets
# swapped back afterward.
_PLACEHOLDER_TEMPLATE = "⟦DNT{index}⟧"


def apply_forced_substitutions(text: str, glossary: dict[str, str]) -> str:
    """Replace every occurrence of an approved source term with its locked
    target-language rendering. Longest terms first, so a multi-word phrase is
    matched before any single-word term nested inside it.
    """
    result = text
    for source_term in sorted(glossary, key=len, reverse=True):
        pattern = re.compile(re.escape(source_term), re.IGNORECASE)
        result = pattern.sub(glossary[source_term], result)
    return result


def mask_dnt_terms(text: str, dnt_terms: list[str]) -> tuple[str, dict[str, str]]:
    """Replace each do-not-translate term with a plain sentinel token before
    the text goes to the NMT model. Returns (masked_text, restore_map); pass
    restore_map to unmask_dnt_terms() after translation.
    """
    masked = text
    restore_map: dict[str, str] = {}
    for i, term in enumerate(sorted(set(dnt_terms), key=len, reverse=True)):
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        if pattern.search(masked):
            placeholder = _PLACEHOLDER_TEMPLATE.format(index=i)
            masked = pattern.sub(placeholder, masked)
            restore_map[placeholder] = term
    return masked, restore_map


def unmask_dnt_terms(text: str, restore_map: dict[str, str]) -> str:
    result = text
    for placeholder, original in restore_map.items():
        result = result.replace(placeholder, original)
    return result