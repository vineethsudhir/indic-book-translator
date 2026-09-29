"""Smoke test for the FR-4 QA module (back-translation + embedding scoring).

Everything here uses fake back-translators/embedders defined below: no models,
no network. Run with: .venv/bin/python scripts/test_qa.py
"""

import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kannada_epub.config import BookConfig, QAConfig
from kannada_epub.qa import (
    FLAGGED_FOR_REVIEW,
    PASS,
    RETRY,
    LLMBackTranslator,
    LocalBackTranslator,
    QAResult,
    build_qa,
    classify,
    cosine_similarity,
    evaluate,
    write_qa_report,
)


def assert_raises(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type as e:
        return e
    raise AssertionError(f"expected {exc_type.__name__}, but no exception was raised")


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------
class FakeBackTranslator:
    def __init__(self, outputs=None):
        self.calls: list[list[str]] = []
        self._outputs = outputs

    def back_translate(self, kannada: list[str]) -> list[str]:
        self.calls.append(list(kannada))
        if self._outputs is not None:
            return list(self._outputs)
        return [f"EN({k})" for k in kannada]


class FakeEmbedder:
    def __init__(self, vectors: dict[str, list[float]]):
        self.calls: list[list[str]] = []
        self._vectors = vectors

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._vectors[t] for t in texts]


class FakeProvider:
    def __init__(self, response: str):
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self.response


# --------------------------------------------------------------------------
# classify: boundaries + threshold validation
# --------------------------------------------------------------------------
assert classify(0.85, 0.85, 0.70) == PASS
assert classify(0.84999, 0.85, 0.70) == RETRY
assert classify(0.70, 0.85, 0.70) == RETRY
assert classify(0.69, 0.85, 0.70) == FLAGGED_FOR_REVIEW
assert classify(1.0, 0.85, 0.70) == PASS
assert classify(0.999999, 0.85, 0.70) == PASS

assert_raises(ValueError, classify, 0.5, 0.6, 0.7)  # flag > pass
assert_raises(ValueError, classify, 0.5, 1.1, 0.7)  # pass > 1
assert_raises(ValueError, classify, 0.5, 0.8, -0.1)  # flag < 0

# --------------------------------------------------------------------------
# cosine_similarity: known vectors, zero vectors
# --------------------------------------------------------------------------
assert math.isclose(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0, rel_tol=1e-12)
assert math.isclose(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0, abs_tol=1e-12)
assert math.isclose(cosine_similarity([3.0, 4.0], [3.0, 4.0]), 1.0, rel_tol=1e-12)
assert math.isclose(cosine_similarity([3.0, 4.0], [4.0, 3.0]), 0.96, rel_tol=1e-12)
assert math.isclose(cosine_similarity([2.0, 0.0], [4.0, 0.0]), 1.0, rel_tol=1e-12)
# Zero vectors -> 0.0, not a division error.
assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0
assert cosine_similarity([1.0, 2.0], [0.0, 0.0]) == 0.0
assert cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0
# Length mismatch is refused.
assert_raises(ValueError, cosine_similarity, [1.0, 2.0], [1.0])

# --------------------------------------------------------------------------
# evaluate: one result per paragraph, right indices/statuses, one call each
# --------------------------------------------------------------------------
originals = ["orig1", "orig2", "orig3"]
kannada = ["kn1", "kn2", "kn3"]
indices = [10, 11, 12]
back = ["back1", "back2", "back3"]
vectors = {
    "orig1": [1.0, 0.0],
    "back1": [1.0, 0.0],  # cos 1.0 -> PASS
    "orig2": [1.0, 0.0],
    "back2": [0.8, 0.6],  # cos 0.8 -> RETRY
    "orig3": [1.0, 0.0],
    "back3": [0.6, 0.8],  # cos 0.6 -> FLAGGED_FOR_REVIEW
}
bt = FakeBackTranslator(outputs=back)
em = FakeEmbedder(vectors)

results = evaluate(
    originals,
    kannada,
    chapter="chapter_03.xhtml",
    paragraph_indices=indices,
    back_translator=bt,
    embedder=em,
    pass_threshold=0.85,
    flag_threshold=0.70,
)

assert len(results) == 3
assert [r.status for r in results] == [PASS, RETRY, FLAGGED_FOR_REVIEW]
assert [r.paragraph_index for r in results] == indices
assert [r.original_en for r in results] == originals
assert [r.kannada_target for r in results] == kannada
assert [r.back_translated_en for r in results] == back
assert math.isclose(results[1].similarity_score, 0.8, rel_tol=1e-9)
# One back-translation call, one embedding call (originals + backs together).
assert len(bt.calls) == 1
assert bt.calls[0] == kannada
assert len(em.calls) == 1
assert em.calls[0] == originals + back

# Empty input -> no calls, no results.
empty_bt = FakeBackTranslator()
empty_em = FakeEmbedder({})
assert evaluate(
    [],
    [],
    chapter="c",
    paragraph_indices=[],
    back_translator=empty_bt,
    embedder=empty_em,
    pass_threshold=0.85,
    flag_threshold=0.70,
) == []
assert empty_bt.calls == []
assert empty_em.calls == []

# Wrong back-translator count -> raise, never misalign.
bad_bt = FakeBackTranslator(outputs=["only one"])
assert_raises(
    ValueError,
    evaluate,
    ["a", "b"],
    ["k1", "k2"],
    chapter="c",
    paragraph_indices=[0, 1],
    back_translator=bad_bt,
    embedder=FakeEmbedder({}),
    pass_threshold=0.85,
    flag_threshold=0.70,
)

# Wrong embedder count -> raise.
class ShortEmbedder:
    def embed(self, texts):
        return [[1.0, 0.0]]

assert_raises(
    ValueError,
    evaluate,
    ["a"],
    ["k"],
    chapter="c",
    paragraph_indices=[0],
    back_translator=FakeBackTranslator(outputs=["b"]),
    embedder=ShortEmbedder(),
    pass_threshold=0.85,
    flag_threshold=0.70,
)

# --------------------------------------------------------------------------
# write_qa_report: summary counts + Kannada stays readable (ensure_ascii=False)
# --------------------------------------------------------------------------
report_results = [
    QAResult("chapter_03.xhtml", 104, "The scheduler preempts.", "ಥ್ರೆಡ್ ಶೆಡ್ಯೂಲರ್", "The scheduler stops.", 0.9, PASS),
    QAResult("chapter_03.xhtml", 105, "A second line.", "ಎರಡನೇ ಸಾಲು", "A second line.", 0.75, RETRY),
    QAResult("chapter_03.xhtml", 106, "A third line.", "ಮೂರನೇ ಸಾಲು", "Different.", 0.5, FLAGGED_FOR_REVIEW),
]

with tempfile.TemporaryDirectory() as tmp:
    report_path = Path(tmp) / "qa_report.json"
    write_qa_report(report_results, report_path)

    raw_text = report_path.read_text(encoding="utf-8")
    assert "ಥ್ರೆಡ್" in raw_text, "Kannada must be written readable, not \\u-escaped"
    assert "\\u" not in raw_text, "ensure_ascii=False expected"

    payload = json.loads(raw_text)
    assert payload["summary"] == {
        "total": 3,
        "pass": 1,
        "retry": 1,
        "flagged": 1,
        "mean_score": round((0.9 + 0.75 + 0.5) / 3, 4),
    }
    assert len(payload["results"]) == 3
    assert payload["results"][0]["paragraph_index"] == 104
    assert payload["results"][2]["similarity_score"] == 0.5

    # Empty report is valid too.
    empty_report = Path(tmp) / "empty_qa_report.json"
    write_qa_report([], empty_report)
    empty_payload = json.loads(empty_report.read_text(encoding="utf-8"))
    assert empty_payload["summary"] == {
        "total": 0,
        "pass": 0,
        "retry": 0,
        "flagged": 0,
        "mean_score": 0.0,
    }

# --------------------------------------------------------------------------
# LLMBackTranslator: JSON array, embedded array, count mismatch
# --------------------------------------------------------------------------
# Plain JSON array.
provider = FakeProvider('["one", "two"]')
llm = LLMBackTranslator(provider)
assert llm.back_translate(["ಕ1", "ಕ2"]) == ["one", "two"]
assert len(provider.calls) == 1
assert "[1]" in provider.calls[0][1] and "[2]" in provider.calls[0][1]

# Array embedded in extra prose (fallback to first [...] block).
provider = FakeProvider('Sure, here you go:\n["one", "two"]\nHope that helps!')
llm = LLMBackTranslator(provider)
assert llm.back_translate(["ಕ1", "ಕ2"]) == ["one", "two"]

# Count mismatch -> raise.
provider = FakeProvider('["only one"]')
llm = LLMBackTranslator(provider)
assert_raises(RuntimeError, llm.back_translate, ["ಕ1", "ಕ2"])

# Non-JSON -> raise.
provider = FakeProvider("no array here at all")
llm = LLMBackTranslator(provider)
assert_raises(RuntimeError, llm.back_translate, ["ಕ1"])

# Empty input makes no provider call.
provider = FakeProvider("[]")
llm = LLMBackTranslator(provider)
assert llm.back_translate([]) == []
assert provider.calls == []

# --------------------------------------------------------------------------
# LocalBackTranslator: missing dir -> FileNotFoundError naming the script
# --------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    missing = Path(tmp) / "does-not-exist"
    err = assert_raises(FileNotFoundError, LocalBackTranslator, missing)
    assert "download_qa_models.py" in str(err), str(err)
    assert str(missing) in str(err), str(err)

# --------------------------------------------------------------------------
# build_qa: None when disabled
# --------------------------------------------------------------------------
assert build_qa(
    QAConfig(),
    default_provider_config_path="config/consistency_editor.example.yaml",
    resolve_path=lambda p: Path(p),
) is None

# --------------------------------------------------------------------------
# QAConfig defaults load from a BookConfig YAML dict with `qa` absent
# --------------------------------------------------------------------------
cfg = BookConfig.model_validate(
    {"epub_path": "book.epub", "output_dir": "out", "glossary_db": "g.db"}
)
assert cfg.qa == QAConfig()
assert cfg.qa.enabled is False
assert cfg.qa.back_translation == "llm"
assert cfg.qa.indic_en_model_dir == "models/indic-en-200m-ct2/indic-en-200m-ct2/ctranslate2_model"
assert cfg.qa.embedding == "local_minilm"
assert cfg.qa.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
assert cfg.qa.pass_threshold == 0.85
assert cfg.qa.flag_threshold == 0.70

print("test_qa: all assertions passed")
