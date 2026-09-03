from dataclasses import dataclass
from typing import Optional


@dataclass
class Candidate:
    """A term surfaced by extraction, not yet reviewed by a human."""

    source_term: str
    term_type: str  # 'acronym' | 'proper_noun_phrase' | 'repeated_capitalized'
    frequency: int
    first_seen_chapter: Optional[str] = None


@dataclass
class GlossaryEntry:
    id: int
    source_term: str
    source_term_key: str
    target_term: Optional[str]
    term_type: str
    status: str  # 'pending' | 'approved' | 'rejected'
    frequency: int
    first_seen_chapter: Optional[str]
    notes: Optional[str]