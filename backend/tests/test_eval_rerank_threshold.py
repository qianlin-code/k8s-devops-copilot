from pathlib import Path

import pytest

from scripts import eval_rerank_threshold as threshold_eval


def test_fixed_eval_set_contract_accepts_current_39_cases(tmp_path: Path) -> None:
    payload = {
        "cases": [
            {"id": case_id}
            for case_id in threshold_eval.FIXED_CASE_IDS
        ]
    }
    path = tmp_path / "eval_set.json"
    path.write_text(__import__("json").dumps(payload), encoding="utf-8")

    cases = threshold_eval._load_fixed_cases(path)

    assert len(cases) == 39
    assert [case["id"] for case in cases] == list(threshold_eval.FIXED_CASE_IDS)


def test_fixed_eval_set_contract_rejects_case_drift(tmp_path: Path) -> None:
    path = tmp_path / "eval_set.json"
    path.write_text('{"cases": [{"id": "q01"}]}', encoding="utf-8")

    with pytest.raises(RuntimeError, match="fixed evaluation set"):
        threshold_eval._load_fixed_cases(path)


def test_threshold_simulation_reports_hit_empty_and_noise() -> None:
    scored = [
        threshold_eval.CaseScore(
            case_id="q01",
            difficulty="hard",
            query="query",
            chunks=[(1, 0.20, True), (2, 0.10, False)],
        ),
        threshold_eval.CaseScore(
            case_id="q02",
            difficulty="easy",
            query="query",
            chunks=[(1, 0.08, False)],
        ),
    ]

    row = threshold_eval._simulate(scored, 0.12)

    assert row.hit_rate == pytest.approx(0.5)
    assert row.hard_hit_rate == pytest.approx(1.0)
    assert row.empty_context_rate == pytest.approx(0.5)
    assert row.avg_kept == pytest.approx(0.5)
    assert row.avg_noise_kept == pytest.approx(0.0)


def test_threshold_candidates_include_knowledge_query_policy_value() -> None:
    assert 0.03 in threshold_eval.THRESHOLDS
