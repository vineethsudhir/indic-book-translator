import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .models import Candidate, GlossaryEntry

SCHEMA = """
CREATE TABLE IF NOT EXISTS glossary_terms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_term TEXT NOT NULL,
    source_term_key TEXT NOT NULL UNIQUE,
    target_term TEXT,
    term_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    frequency INTEGER NOT NULL DEFAULT 0,
    first_seen_chapter TEXT,
    notes TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_glossary_status ON glossary_terms(status);
"""


def _normalize_key(term: str) -> str:
    return " ".join(term.strip().lower().split())


class GlossaryStore:
    """Persistent, project-scoped Translation Memory: one SQLite file per book
    (or series, if reused across books) holding source->target term mappings
    at every stage from raw candidate to human-approved.
    """

    def __init__(self, db_path: str | Path):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def add_candidates(self, candidates: list[Candidate]) -> None:
        """Insert new candidates as 'pending'; bump frequency on repeats
        (e.g. the same term reappearing in a later chapter's scan)."""
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            for c in candidates:
                key = _normalize_key(c.source_term)
                existing = conn.execute(
                    "SELECT id FROM glossary_terms WHERE source_term_key = ?", (key,)
                ).fetchone()
                if existing:
                    conn.execute(
                        "UPDATE glossary_terms SET frequency = frequency + ?, updated_at = ? WHERE id = ?",
                        (c.frequency, now, existing["id"]),
                    )
                else:
                    conn.execute(
                        """INSERT INTO glossary_terms
                           (source_term, source_term_key, term_type, status, frequency,
                            first_seen_chapter, created_at, updated_at)
                           VALUES (?, ?, ?, 'pending', ?, ?, ?, ?)""",
                        (c.source_term, key, c.term_type, c.frequency, c.first_seen_chapter, now, now),
                    )

    def get_pending(self, min_frequency: int = 1) -> list[GlossaryEntry]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM glossary_terms WHERE status = 'pending' AND frequency >= ? "
                "ORDER BY frequency DESC",
                (min_frequency,),
            ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def get_approved(self) -> list[GlossaryEntry]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM glossary_terms WHERE status = 'approved' ORDER BY frequency DESC"
            ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def get_approved_dict(self) -> dict[str, str]:
        """English source term -> locked target term. Feeds directly into
        ConsistencyEditor.edit_chapter(glossary=...) and forced substitution."""
        return {e.source_term: e.target_term for e in self.get_approved() if e.target_term}

    def get_relevant_glossary(self, text: str) -> dict[str, str]:
        """Approved entries whose source term actually occurs in `text` — keeps
        prompts/substitution passes scoped instead of dragging in the whole
        book-length glossary for every chapter."""
        text_lower = text.lower()
        return {term: target for term, target in self.get_approved_dict().items() if term.lower() in text_lower}

    def upsert_review(
        self,
        source_term_key: str,
        status: str,
        target_term: Optional[str] = None,
        term_type: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> None:
        if status not in ("pending", "approved", "rejected"):
            raise ValueError(f"Invalid status: {status!r}")
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            fields = ["status = ?", "updated_at = ?"]
            params: list = [status, now]
            if target_term is not None:
                fields.append("target_term = ?")
                params.append(target_term)
            if term_type is not None:
                fields.append("term_type = ?")
                params.append(term_type)
            if notes is not None:
                fields.append("notes = ?")
                params.append(notes)
            params.append(source_term_key)
            conn.execute(
                f"UPDATE glossary_terms SET {', '.join(fields)} WHERE source_term_key = ?", params
            )

    def all_entries(self) -> list[GlossaryEntry]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM glossary_terms ORDER BY frequency DESC").fetchall()
        return [self._row_to_entry(r) for r in rows]

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> GlossaryEntry:
        return GlossaryEntry(
            id=row["id"],
            source_term=row["source_term"],
            source_term_key=row["source_term_key"],
            target_term=row["target_term"],
            term_type=row["term_type"],
            status=row["status"],
            frequency=row["frequency"],
            first_seen_chapter=row["first_seen_chapter"],
            notes=row["notes"],
        )