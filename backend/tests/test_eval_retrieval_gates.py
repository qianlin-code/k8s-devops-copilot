from dataclasses import replace

import pytest

from scripts.eval_retrieval import (
    FIXED_HARD_CASE_IDS,
    ConfigResult,
    _case_rank_comparison,
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


def test_release_gate_accepts_saturated_vector_baseline() -> None:
    vector = _result(
        mrr=0.919444,
        hit_at_5=1.0,
        hard_mrr=0.887255,
        hard_hit_at_3=0.941176,
    )
    hybrid = _result(
        mrr=0.927778,
        hit_at_5=1.0,
        hard_mrr=0.901961,
        hard_hit_at_3=1.0,
    )
    reranked = _result(
        mrr=0.944444,
        hit_at_5=1.0,
        hard_mrr=0.931373,
        hard_hit_at_3=1.0,
    )

    gate = _release_gates([vector, hybrid, reranked])

    assert gate["passed"] is True
    assert gate["observed"]["hard_mrr_improvement"] == pytest.approx(0.044118)
    assert gate["observed"]["hard_mrr_improvement_vs_hybrid"] == pytest.approx(
        0.029412
    )
    assert gate["legacy_diagnostic"]["required_hard_mrr"] == pytest.approx(
        1.037255
    )
    assert gate["legacy_diagnostic"]["max_possible_hard_mrr_improvement"] == (
        pytest.approx(0.112745)
    )
    assert gate["legacy_diagnostic"]["feasible"] is False


@pytest.mark.parametrize(
    ("override", "failed_check"),
    [
        ({"hard_mrr": 0.89}, "hard_mrr"),
        ({"hard_hit_at_3": 0.99}, "hard_hit_at_3"),
        ({"rerank_verified": False}, "rerank_verified"),
        ({"hit_at_5": 0.84}, "full_hit_at_5"),
        ({"mrr": 0.79}, "full_mrr"),
    ],
)
def test_release_gate_rejects_failed_absolute_or_verification_checks(
    override: dict[str, object], failed_check: str
) -> None:
    vector = _result(mrr=0.88, hit_at_5=0.90, hard_mrr=0.88)
    hybrid = _result(mrr=0.90, hit_at_5=0.92, hard_mrr=0.90)
    reranked = _result(
        **{
            "mrr": 0.93,
            "hit_at_5": 0.95,
            "hard_mrr": 0.93,
            "hard_hit_at_3": 1.0,
            **override,
        }
    )

    gate = _release_gates([vector, hybrid, reranked])

    assert gate["passed"] is False
    assert gate["checks"][failed_check] is False


def test_release_gate_requires_rerank_hard_gain_over_hybrid() -> None:
    vector = _result(mrr=0.88, hit_at_5=0.90, hard_mrr=0.88)
    hybrid = _result(mrr=0.92, hit_at_5=0.95, hard_mrr=0.93)
    reranked = _result(
        mrr=0.93,
        hit_at_5=0.95,
        hard_mrr=0.93,
        hard_hit_at_3=1.0,
    )

    gate = _release_gates([vector, hybrid, reranked])

    assert gate["passed"] is False
    assert gate["checks"]["hard_mrr_improvement_vs_hybrid"] is False


def test_release_gate_rejects_hard_regression_vs_vector() -> None:
    vector = _result(mrr=0.88, hit_at_5=0.90, hard_mrr=0.95)
    hybrid = _result(mrr=0.90, hit_at_5=0.92, hard_mrr=0.90)
    reranked = _result(
        mrr=0.93,
        hit_at_5=0.95,
        hard_mrr=0.93,
        hard_hit_at_3=1.0,
    )

    gate = _release_gates([vector, hybrid, reranked])

    assert gate["passed"] is False
    assert gate["checks"]["hard_mrr_non_regression_vs_vector"] is False


def test_case_rank_comparison_reports_all_three_stages() -> None:
    vector = _result(case_ranks={"q11": 4, "q13": 1})
    hybrid = _result(case_ranks={"q11": 3, "q13": 1})
    reranked = _result(case_ranks={"q11": 1, "q13": 2})

    comparison = _case_rank_comparison(
        [
            {"id": "q11", "difficulty": "hard"},
            {"id": "q13", "difficulty": "hard"},
        ],
        [vector, hybrid, reranked],
    )

    assert comparison == [
        {
            "case_id": "q11",
            "difficulty": "hard",
            "vector_rank": 4,
            "hybrid_rank": 3,
            "reranked_rank": 1,
            "rank_gain_vs_vector": 3,
            "rank_gain_vs_hybrid": 2,
        },
        {
            "case_id": "q13",
            "difficulty": "hard",
            "vector_rank": 1,
            "hybrid_rank": 1,
            "reranked_rank": 2,
            "rank_gain_vs_vector": -1,
            "rank_gain_vs_hybrid": -1,
        },
    ]
