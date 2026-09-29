import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .base import BackTranslator, Embedder
from .scoring import FLAGGED_FOR_REVIEW, PASS, RETRY, classify, cosine_similarity


@dataclass
class QAResult:
    """One scored paragraph. Field names match the FR-4 JSON example
    (`chapter`, `original_en`, `kannada_target`, `back_translated_en`,
    `similarity_score`, `status`) plus `paragraph_index` for stable lookup.
    """

    chapter: str
    paragraph_index: int
    original_en: str
    kannada_target: str
    back_translated_en: str
    similarity_score: float
    status: str


def evaluate(
    original_en: list[str],
    kannada: list[str],
    *,
    chapter: str,
    paragraph_indices: list[int],
    back_translator: BackTranslator,
    embedder: Embedder,
    pass_threshold: float,
    flag_threshold: float,
) -> list[QAResult]:
    """Back-translate `kannada`, embed originals and back-translations, score
    and classify every paragraph.

    Exactly one back-translation call and one embedding call are made:
    originals and back-translations are embedded together in a single batch.
    Any count mismatch raises rather than producing misaligned results.
    """
    if not (len(original_en) == len(kannada) == len(paragraph_indices)):
        raise ValueError(
            "evaluate: original_en, kannada and paragraph_indices must be the same length "
            f"({len(original_en)}, {len(kannada)}, {len(paragraph_indices)})"
        )
    if not original_en:
        return []

    back_translated = back_translator.back_translate(kannada)
    if len(back_translated) != len(kannada):
        raise ValueError(
            f"evaluate: back-translator returned {len(back_translated)} paragraphs for "
            f"{len(kannada)} inputs"
        )

    # One embedding call: originals first, then back-translations.
    embeddings = embedder.embed(list(original_en) + list(back_translated))
    if len(embeddings) != 2 * len(original_en):
        raise ValueError(
            f"evaluate: embedder returned {len(embeddings)} vectors for "
            f"{2 * len(original_en)} texts"
        )

    half = len(original_en)
    original_vectors = embeddings[:half]
    back_vectors = embeddings[half:]

    results: list[QAResult] = []
    for i in range(half):
        score = cosine_similarity(original_vectors[i], back_vectors[i])
        results.append(
            QAResult(
                chapter=chapter,
                paragraph_index=paragraph_indices[i],
                original_en=original_en[i],
                kannada_target=kannada[i],
                back_translated_en=back_translated[i],
                similarity_score=score,
                status=classify(score, pass_threshold, flag_threshold),
            )
        )
    return results


def write_qa_report(results: list[QAResult], path: str | Path) -> None:
    """Write the QA report as UTF-8 JSON (Kannada kept readable, scores rounded
    to 4 decimals)."""
    counts = {PASS: 0, RETRY: 0, FLAGGED_FOR_REVIEW: 0}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    mean_score = sum(r.similarity_score for r in results) / len(results) if results else 0.0

    payload = {
        "summary": {
            "total": len(results),
            "pass": counts[PASS],
            "retry": counts[RETRY],
            "flagged": counts[FLAGGED_FOR_REVIEW],
            "mean_score": round(mean_score, 4),
        },
        "results": [
            {**asdict(r), "similarity_score": round(r.similarity_score, 4)} for r in results
        ],
    }

    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
