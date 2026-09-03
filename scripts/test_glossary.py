"""End-to-end smoke test: extract candidates from sample chapter text, run
them through the CSV review round-trip, then feed the resulting locked
glossary into the live ConsistencyEditor (Ollama/gemma4:26b)."""

import csv
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kannada_epub.config import load_provider_config
from kannada_epub.consistency_editor import ConsistencyEditor
from kannada_epub.glossary import (
    GlossaryStore,
    export_pending_for_review,
    extract_candidates,
    import_reviewed,
)
from kannada_epub.providers.factory import build_provider

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "consistency_editor.example.yaml"

CHAPTER_1_TEXT = """
Meera opened her laptop and stared at the kernel logs. The Thread Scheduler was misbehaving
again, and Meera knew exactly why.

Every time the HTTP server received a burst of requests, the Thread Scheduler would stall.
The engineers assigned to Meera showed her the traces, and Meera had seen this pattern before.

She restarted the service and watched the Thread Scheduler recover. The HTTP API logs looked
clean now, and Meera smiled before closing her laptop. The HTTP endpoint kept responding
correctly all night.
"""

# What a human reviewer would decide after opening the exported CSV.
REVIEWER_DECISIONS = {
    "thread scheduler": ("ಥ್ರೆಡ್ ಶೆಡ್ಯೂಲರ್", "approved"),
    "http": ("HTTP", "approved"),  # keep untranslated acronym, but lock it explicitly
    "meera": ("ಮೀರಾ", "approved"),
}

CHAPTER_2_DRAFT_KANNADA = (
    "ಥ್ರೆಡ್ ಶೆಡ್ಯೂಲರ್ ಚಾಲನೆಯಲ್ಲಿರುವ ಪ್ರಕ್ರಿಯೆಗಳನ್ನು ತಡೆಯುತ್ತದೆ. "
    "ಅವಳು ಅದನ್ನು ಮತ್ತೆ ಪ್ರಾರಂಭಿಸಿದಳು."
)
CHAPTER_2_SOURCE_TEXT_FOR_SCOPING = (
    "The Thread Scheduler stalled again. Meera restarted it once more."
)


def simulate_human_review(csv_path: Path) -> None:
    rows = list(csv.DictReader(open(csv_path, newline="", encoding="utf-8")))
    for row in rows:
        if row["source_term_key"] in REVIEWER_DECISIONS:
            target, status = REVIEWER_DECISIONS[row["source_term_key"]]
            row["target_term"] = target
            row["status"] = status
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "project.glossary.db"
        csv_path = Path(tmp) / "review.csv"

        store = GlossaryStore(db_path)

        candidates = extract_candidates(CHAPTER_1_TEXT, chapter_id="chapter_01", min_frequency=3)
        print(f"Extracted {len(candidates)} candidates from chapter 1:")
        for c in candidates:
            print(f"  {c.term_type:20s} {c.source_term!r:20s} freq={c.frequency}")
        store.add_candidates(candidates)

        n = export_pending_for_review(store, csv_path)
        print(f"\nExported {n} pending candidates to {csv_path}")

        simulate_human_review(csv_path)
        result = import_reviewed(store, csv_path)
        print(f"Imported review: {result}")

        relevant = store.get_relevant_glossary(CHAPTER_2_SOURCE_TEXT_FOR_SCOPING)
        print(f"\nRelevant glossary for chapter 2 (scoped): {relevant}")

        config = load_provider_config(CONFIG_PATH)
        provider = build_provider(config)
        editor = ConsistencyEditor(provider)

        edited = editor.edit_chapter(
            draft_kannada_text=CHAPTER_2_DRAFT_KANNADA,
            glossary=relevant,
            prior_chapter_context="Chapter 1 introduced Meera, a systems engineer debugging a kernel scheduler.",
            register="neutral, technical, third-person narration",
        )
        print("\n--- Edited chapter (glossary sourced from persistent store) ---")
        for p in edited:
            print(f"[{p.emotion}] {p.text}")