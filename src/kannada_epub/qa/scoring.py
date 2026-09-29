import math

PASS = "PASS"
RETRY = "RETRY"
FLAGGED_FOR_REVIEW = "FLAGGED_FOR_REVIEW"


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors, pure Python.

    Returns 0.0 if either vector is all zeros (an undefined angle is treated
    as no similarity rather than raising or dividing by zero).
    """
    if len(a) != len(b):
        raise ValueError(f"cosine_similarity: vector length mismatch ({len(a)} vs {len(b)})")

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def classify(score: float, pass_threshold: float, flag_threshold: float) -> str:
    """Map a similarity score to a QA status.

    PASS                : score >= pass_threshold
    RETRY               : flag_threshold <= score < pass_threshold
    FLAGGED_FOR_REVIEW  : score < flag_threshold
    """
    if not (0.0 <= flag_threshold <= pass_threshold <= 1.0):
        raise ValueError(
            "Thresholds must satisfy 0 <= flag_threshold <= pass_threshold <= 1, "
            f"got flag_threshold={flag_threshold}, pass_threshold={pass_threshold}"
        )

    if score >= pass_threshold:
        return PASS
    if score >= flag_threshold:
        return RETRY
    return FLAGGED_FOR_REVIEW
