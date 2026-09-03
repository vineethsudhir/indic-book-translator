import csv
from pathlib import Path

from .store import GlossaryStore

FIELDNAMES = [
    "source_term_key",
    "source_term",
    "term_type",
    "target_term",
    "status",
    "frequency",
    "first_seen_chapter",
    "notes",
]


def export_pending_for_review(store: GlossaryStore, out_path: str | Path, min_frequency: int = 1) -> int:
    """Write pending candidates to a CSV a human can open in any spreadsheet
    app, fill in target_term, and flip status to 'approved' or 'rejected'.
    Defaults status to 'pending' — nothing gets locked in without an explicit
    human decision.
    """
    entries = store.get_pending(min_frequency=min_frequency)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for e in entries:
            writer.writerow(
                {
                    "source_term_key": e.source_term_key,
                    "source_term": e.source_term,
                    "term_type": e.term_type,
                    "target_term": e.target_term or "",
                    "status": "pending",
                    "frequency": e.frequency,
                    "first_seen_chapter": e.first_seen_chapter or "",
                    "notes": e.notes or "",
                }
            )
    return len(entries)


def import_reviewed(store: GlossaryStore, in_path: str | Path) -> dict:
    """Read a reviewed CSV back and apply the human's decisions. Rows marked
    'approved' with no target_term are refused (with a reason), not silently
    dropped, so the reviewer can see and fix them.
    """
    imported = 0
    skipped: list[str] = []
    with open(in_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row["source_term_key"]
            status = row["status"].strip().lower()
            if status not in ("pending", "approved", "rejected"):
                skipped.append(f"{key}: invalid status {row['status']!r}")
                continue
            if status == "approved" and not row["target_term"].strip():
                skipped.append(f"{key}: approved but target_term is empty")
                continue
            store.upsert_review(
                source_term_key=key,
                status=status,
                target_term=row["target_term"].strip() or None,
                term_type=row["term_type"].strip() or None,
                notes=row["notes"].strip() or None,
            )
            imported += 1
    return {"imported": imported, "skipped": skipped}