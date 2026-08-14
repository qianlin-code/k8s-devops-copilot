from dataclasses import replace

import pytest

from scripts.eval_retrieval import (
    FIXED_HARD_CASE_IDS,
    ConfigResult,
    _release_gates,
    _validate_hard_subset,
)


def _result(**overrides: object) -> ConfigResult:
    base = ConfigResult(
        label="config",
        hit_at_1=0.80,
        hit_at_3=0.88,
        hit_at_5=0.90,
        mrr=0.81,
        avg_latency_ms=1.0,
        misses=[],
        hard_mrr=0.50,
        hard_hit_at_3=0.70,
        hard_count=len(FIXED_HARD_CASE_IDS),
    )
    return replace(base, **overrides)


def test_hard_subset_contract_is_ordered_and_fixed() -> None:
    cases = [
        {"id": case_id, "difficulty": "hard"}
        for case_id in FIXED_HARD_CASE_IDS
    ]

    assert _validate_hard_subset(cases) == list(FIXED_HARD_CASE_IDS)


def test_hard_subset_contract_rejects_relabeling() -> None:
    cases = [
        {"id": case_id, "difficulty": "hard"}
        for case_id in FIXED_HARD_CASE_IDS[:-1]
    ]

    with pytest.raises(RuntimeError, match="hard evaluation subset drifted"):
        _validate_hard_subset(cases)


def test_release_gate_uses_hard_delta_and_reports_full_delta() -> None:
    vector = _result(mrr=0.78, hit_at_5=0.88, hard_mrr=0.50)
    hybrid = _result(mrr=0.80, hit_at_5=0.88, hard_mrr=0.60)
    reranked = _result(mrr=0.84, hit_at_5=0.90, hard_mrr=0.66)

    gate = _release_gates([vector, hybrid, reranked])

    assert gate["passed"] is True
    assert gate["observed"]["full_mrr_improvement"] == pytest.approx(0.06)
    assert gate["observed"]["hard_mrr_improvement"] == pytest.approx(0.16)


def test_release_gate_rejects_hard_delta_below_threshold() -> None:
    vector = _result(mrr=0.78, hit_at_5=0.88, hard_mrr=0.50)
    hybrid = _result(mrr=0.80, hit_at_5=0.88, hard_mrr=0.58)
    reranked = _result(mrr=0.84, hit_at_5=0.90, hard_mrr=0.64)

    gate = _release_gates([vector, hybrid, reranked])

    assert gate["passed"] is False
    assert gate["checks"]["hard_mrr_improvement"] is False
