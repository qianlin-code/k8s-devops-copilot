import argparse
import json
import importlib.util
from pathlib import Path

from app.rag.fusion import FusedChunk
from app.rag.reranker import RerankedChunk
from app.rag.vector_store import ScoredChunk


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_ragas.py"
SPEC = importlib.util.spec_from_file_location("eval_ragas", SCRIPT)
assert SPEC and SPEC.loader
eval_ragas = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(eval_ragas)


def _case(**overrides):
    case = {
        "id": "q01",
        "expected_outcome": "direct_answer",
        "expected_tool": None,
        "expected_keywords": ["Pending"],
    }
    case.update(overrides)
    return case


def _result(**overrides):
    result = eval_ragas.CaseResult(
        case_id="q01", query="question", difficulty="easy", gold_answer="gold",
        expected_outcome="direct_answer", expected_tool=None,
        actual_outcome="direct_answer", actual_tool=None, answer="answer",
        answer_generation={"status": "verified", "attempts": 1, "fallback_reason": None},
        citation_records=[{
            "chunk_id": "chunk-1", "document_id": "doc-1", "document_title": "pod",
            "heading_path": ["pod", "pending"], "citation_label": "pod > pending [chunk-1]",
            "rendered_text": "Pending resource", "exact_quote": "Pending resource",
        }],
        verified_evidence_records=[{
            "item_index": 1, "section": "conclusion", "evidence_kind": "knowledge",
            "source_id": "K1", "rendered_text": "Pending resource",
            "chunk_id": "chunk-1", "document_id": "doc-1", "document_title": "pod",
            "heading_path": ["pod", "pending"], "citation_label": "pod > pending [chunk-1]",
            "exact_quote": "Pending resource", "tool_name": None,
            "invocation_index": None, "json_pointer": None, "serialized_value": None,
        }],
    )
    for key, value in overrides.items():
        setattr(result, key, value)
    return result


def test_knowledge_case_requires_exact_outcome_and_no_tool():
    result = _result(actual_outcome="insufficient_information")
    verdict = eval_ragas.evaluate_strict_case(case=_case(), result=result, allowed_titles={"pod"}, write_executed=False)
    assert not verdict.passed
    assert "outcome_expected_direct_answer_got_insufficient_information" in verdict.failures


def test_tool_case_requires_expected_tool_and_outcome():
    case = _case(expected_outcome="tool_assisted_answer", expected_tool="get_pod_status")
    result = _result(actual_outcome="tool_assisted_answer", actual_tool="list_deployments")
    verdict = eval_ragas.evaluate_strict_case(case=case, result=result, allowed_titles={"pod"}, write_executed=False)
    assert not verdict.passed
    assert "tool_expected_get_pod_status_got_list_deployments" in verdict.failures
    assert "missing_verified_tool_evidence" in verdict.failures


def test_write_confirmation_must_not_execute_write_tool():
    case = _case(expected_outcome="write_confirmation_required", expected_tool="restart_deployment")
    result = _result(actual_outcome="write_confirmation_required", actual_tool="restart_deployment")
    verdict = eval_ragas.evaluate_strict_case(case=case, result=result, allowed_titles={"pod"}, write_executed=True)
    assert not verdict.passed
    assert "write_tool_executed_during_evaluation" in verdict.failures


def test_incomplete_write_case_still_requires_zero_execution():
    case = _case(
        expected_outcome="insufficient_information",
        expected_tool="create_incident",
        expected_write_executed=False,
    )
    result = _result(
        actual_outcome="insufficient_information", actual_tool="create_incident"
    )
    verdict = eval_ragas.evaluate_strict_case(
        case=case, result=result, allowed_titles={"pod"}, write_executed=True
    )
    assert not verdict.passed
    assert "write_tool_executed_during_evaluation" in verdict.failures


def test_retrieval_diagnostics_do_not_change_citation_contract():
    result = _result(
        retrieval_diagnostics=[
            {"chunk_id": "chunk-1", "kept": False, "filter_reason": "below_rerank_threshold"}
        ]
    )
    payload = eval_ragas._detail_payload(result)
    assert payload["retrieval_diagnostics"][0]["filter_reason"] == "below_rerank_threshold"


def test_layer_diagnostics_distinguish_threshold_filtering_from_recall_failure():
    relevant = ScoredChunk(
        chunk_id="chunk-1", document_id="doc-1", document_title="pod",
        text="CrashLoopBackOff uses exponential backoff", heading_path=["pod", "CrashLoopBackOff"],
        chunk_index=0, score=0.8,
    )
    retrieval = type("Retrieval", (), {
        "vector_hits": [relevant], "bm25_hits": [relevant],
        "fused_chunks": [FusedChunk(relevant, rrf_score=0.03, vector_rank=1, bm25_rank=1)],
        "reranked_chunks": [RerankedChunk(relevant, rerank_score=0.0528, rank_before=1, rank_after=1)],
        "chunks": [],
    })()

    layers = eval_ragas._retrieval_layers(retrieval, ["CrashLoopBackOff"])

    assert layers["vector"][0]["keyword_hit"] is True
    assert layers["bm25"][0]["keyword_hit"] is True
    assert layers["rrf"][0]["sources"] == ["vector", "bm25"]
    assert layers["rerank"][0]["keyword_hit"] is True
    assert layers["post_threshold"] == []


def test_layer_diagnostics_keep_rerank_rank_after_reranking():
    first = ScoredChunk(
        chunk_id="chunk-first", document_id="doc-1", document_title="pod",
        text="irrelevant", heading_path=["pod"], chunk_index=0, score=0.9,
    )
    relevant = ScoredChunk(
        chunk_id="chunk-relevant", document_id="doc-1", document_title="pod",
        text="CrashLoopBackOff backoff restart", heading_path=["pod", "CrashLoopBackOff"],
        chunk_index=1, score=0.7,
    )
    retrieval = type("Retrieval", (), {
        "vector_hits": [first, relevant], "bm25_hits": [relevant],
        "fused_chunks": [FusedChunk(first, 0.03, 1, None), FusedChunk(relevant, 0.02, 2, 1)],
        "reranked_chunks": [
            RerankedChunk(relevant, rerank_score=0.2, rank_before=2, rank_after=1),
            RerankedChunk(first, rerank_score=0.1, rank_before=1, rank_after=2),
        ],
        "chunks": [RerankedChunk(relevant, rerank_score=0.2, rank_before=2, rank_after=1)],
    })()

    layers = eval_ragas._retrieval_layers(retrieval, ["CrashLoopBackOff"])

    assert layers["rrf"][1]["keyword_hit"] is True
    assert layers["rerank"][0]["chunk_id"] == "chunk-relevant"
    assert layers["rerank"][0]["rank"] == 1
    assert layers["rerank"][0]["rank_before"] == 2
    assert layers["rerank"][0]["rank_after"] == 1


def test_subset_aggregate_uses_null_for_absent_tool_categories():
    result = _result()
    result.strict = eval_ragas.evaluate_strict_case(
        case=_case(), result=result, allowed_titles={"pod"}, write_executed=False
    )

    aggregate = eval_ragas._aggregate("local", [result], judge_enabled=False)

    assert aggregate.knowledge_routing_accuracy == 1.0
    assert aggregate.readonly_tool_routing_accuracy is None
    assert aggregate.write_confirmation_routing_accuracy is None


def test_knowledge_citation_must_be_parseable_and_keyword_supported():
    result = _result(citation_records=[{
        "chunk_id": "", "document_id": "doc-1", "document_title": "unknown",
        "heading_path": [], "rendered_text": "unrelated", "exact_quote": "unrelated",
    }])
    verdict = eval_ragas.evaluate_strict_case(case=_case(), result=result, allowed_titles={"pod"}, write_executed=False)
    assert not verdict.passed
    assert not verdict.citation.parseable
    assert not verdict.citation.keyword_supported
    assert "citation_unknown_document" in verdict.failures


def test_retrieved_keyword_cannot_replace_final_verified_quote():
    result = _result(
        retrieval_citation_records=[{
            "chunk_id": "retrieved", "document_id": "doc-1", "document_title": "pod",
            "heading_path": ["pending"], "contextual_text": "Pending is present here",
        }],
        citation_records=[{
            "chunk_id": "selected", "document_id": "doc-1", "document_title": "pod",
            "heading_path": ["other"], "citation_label": "pod > other [selected]",
            "rendered_text": "Unrelated selected text", "exact_quote": "Unrelated selected text",
        }],
    )

    verdict = eval_ragas.evaluate_strict_case(
        case=_case(), result=result, allowed_titles={"pod"}, write_executed=False
    )

    assert not verdict.passed
    assert "citation_missing_expected_keyword_support" in verdict.failures


def test_generated_answer_fallback_fails_strict_gate():
    result = _result(
        actual_outcome="insufficient_information",
        answer_generation={"status": "fallback", "attempts": 2, "fallback_reason": "exact_quote_mismatch"},
        citation_records=[],
        verified_evidence_records=[],
    )

    verdict = eval_ragas.evaluate_strict_case(
        case=_case(), result=result, allowed_titles={"pod"}, write_executed=False
    )

    assert not verdict.passed
    assert "answer_generation_not_verified" in verdict.failures


def test_tool_assisted_answer_requires_matching_verified_tool_mapping():
    case = _case(expected_outcome="tool_assisted_answer", expected_tool="get_pod_status")
    result = _result(
        expected_outcome="tool_assisted_answer",
        expected_tool="get_pod_status",
        actual_outcome="tool_assisted_answer",
        actual_tool="get_pod_status",
        citation_records=[],
        verified_evidence_records=[{
            "item_index": 1, "section": "conclusion", "evidence_kind": "tool",
            "source_id": "T1", "rendered_text": "Running", "tool_name": "get_pod_status",
            "invocation_index": 1, "json_pointer": "/status", "serialized_value": "Running",
        }],
    )

    verdict = eval_ragas.evaluate_strict_case(
        case=case, result=result, allowed_titles={"pod"}, write_executed=False
    )

    assert verdict.passed


def test_no_judge_payload_uses_explicit_not_run_markers():
    payload = eval_ragas._detail_payload(_result())
    assert payload["faithfulness"] == "not_run"
    assert payload["answer_relevancy"] == "not_run"


def test_judge_context_uses_final_tool_evidence_instead_of_unselected_retrieval():
    result = _result(
        retrieved_texts=["unselected retrieval"],
        verified_evidence_records=[{
            "item_index": 1, "section": "conclusion", "evidence_kind": "tool",
            "source_id": "T1.A2", "rendered_text": "Running",
            "tool_name": "get_pod_status", "invocation_index": 1,
            "json_pointer": "/phase", "serialized_value": "Running",
        }],
    )

    context = eval_ragas._judge_context_records(result, pending_write=None)

    assert context == [{
        "kind": "tool",
        "source_id": "T1.A2",
        "tool_name": "get_pod_status",
        "json_pointer": "/phase",
        "text": "Running",
    }]


def test_pending_write_judge_context_is_deterministic_and_redacted():
    context = eval_ragas._judge_context_records(
        _result(verified_evidence_records=[]),
        pending_write={
            "tool_name": "restart_deployment",
            "description": "滚动重启",
            "arguments": {
                "namespace": "ops-demo",
                "name": "worker-queue",
                "request_id": "secret-request-id",
                "confirmation_token": "secret-token",
            },
        },
    )

    serialized = json.dumps(context, ensure_ascii=False)
    assert context[0]["kind"] == "pending_write"
    assert context[0]["arguments"] == {
        "name": "worker-queue",
        "namespace": "ops-demo",
    }
    assert "secret-request-id" not in serialized
    assert "secret-token" not in serialized


def test_no_judge_output_creates_a_new_evidence_directory(tmp_path):
    result = _result()
    target = tmp_path / "new-evidence" / "no-judge-results.json"

    eval_ragas._write_outputs(
        target,
        args=argparse.Namespace(no_judge=True),
        aggregate=eval_ragas._aggregate("local", [result], judge_enabled=False),
        results=[result],
        metadata={},
        exit_code=0,
    )

    assert target.is_file()
    assert target.with_name("no-judge-results-citation-review.md").is_file()


def test_cost_estimate_is_conservative_and_model_bound():
    tokens, cost = eval_ragas.estimate_judge_cost_cny([_result()], "qwen-max")
    assert tokens > 1500
    assert cost is not None and cost > 0
    assert eval_ragas.estimate_judge_cost_cny([_result()], "unknown")[1] is None


def test_unknown_generation_model_price_is_rejected():
    try:
        eval_ragas.generation_price_per_1k_tokens_cny("unknown")
    except ValueError as exc:
        assert "unknown" in str(exc)
    else:
        raise AssertionError("unknown generation pricing must fail closed")


def test_deepseek_generation_prices_are_separated_and_exact():
    assert eval_ragas.generation_prices_per_1k_tokens_cny("deepseek-v4-pro") == {
        "prompt_cache_miss": 0.003,
        "prompt_cache_hit": 0.000025,
        "completion": 0.006,
    }


def test_deepseek_bootstrap_selects_chat_only(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only-key")
    changed_keys = {
        "JWT_SECRET_KEY", "STARTUP_PROBE_EXTERNAL", "DATABASE_URL", "QDRANT_PATH",
        "LLM_PROVIDER", "EMBEDDING_PROVIDER", "ENABLE_QUERY_REWRITE", "CHUNK_STRATEGY",
        "RETRIEVE_TOP_K", "RERANK_TOP_N", "AGENT_MAX_STEPS", "TOOL_CACHE_TTL_SECONDS",
    }
    original = {key: eval_ragas.os.environ.get(key) for key in changed_keys}
    try:
        eval_ragas._bootstrap_env("deepseek")

        assert eval_ragas.os.environ["LLM_PROVIDER"] == "deepseek"
        assert eval_ragas.os.environ["EMBEDDING_PROVIDER"] == "ollama"
    finally:
        for key, value in original.items():
            if value is None:
                eval_ragas.os.environ.pop(key, None)
            else:
                eval_ragas.os.environ[key] = value


def _generation_payload(result=None):
    result = result or _result()
    current = next(
        case
        for case in json.loads(eval_ragas.EVAL_SET.read_text(encoding="utf-8"))["cases"]
        if case["id"] == result.case_id
    )
    result.query = current["query"]
    result.gold_answer = current["gold_answer"]
    result.expected_outcome = current["expected_outcome"]
    result.expected_tool = current.get("expected_tool")
    result.strict = eval_ragas.evaluate_strict_case(
        case=_case(), result=result, allowed_titles={"pod"}, write_executed=False
    )
    return {
        "schema_version": 3,
        "script_sha256": eval_ragas.hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
        "eval_set_sha256": eval_ragas.hashlib.sha256(eval_ragas.EVAL_SET.read_bytes()).hexdigest(),
        "document_sha256": eval_ragas._source_hashes(),
        "case_count": 1,
        "judge_mode": "not_run",
        "exit_code": 0,
        "metadata": {
            "generation_provider": "qwen",
            "generation_model": "qwen-plus",
            "generation_usage": {"cost_cny": 0.01},
        },
        "aggregate": {
            **argparse.Namespace().__dict__,
            **eval_ragas.asdict(eval_ragas._aggregate("cloud", [result], False)),
        },
        "cases": [eval_ragas._detail_payload(result)],
    }


def test_generation_report_hash_mismatch_is_rejected():
    payload = _generation_payload()
    payload["eval_set_sha256"] = "wrong"

    try:
        eval_ragas._validate_generation_report(payload)
    except ValueError as exc:
        assert "eval_set_sha256_mismatch" in str(exc)
    else:
        raise AssertionError("hash mismatch should fail")


def test_deepseek_generation_report_is_accepted_for_independent_judging():
    payload = _generation_payload()
    payload["metadata"]["generation_provider"] = "deepseek"
    payload["metadata"]["generation_model"] = "deepseek-v4-pro"

    results = eval_ragas._validate_generation_report(payload)

    assert len(results) == 1


def test_zero_budget_judge_is_preflight_only(tmp_path, monkeypatch):
    generation = tmp_path / "generation.json"
    output = tmp_path / "preflight.json"
    generation.write_text(json.dumps(_generation_payload(), ensure_ascii=False), encoding="utf-8")

    def fail_if_called():
        raise AssertionError("judge client must not initialize during zero-budget preflight")

    import app.llm.factory

    monkeypatch.setattr(app.llm.factory, "get_judge_client", fail_if_called)
    exit_code = eval_ragas._judge_existing_report(
        generation, output, max_total_cost_cny=0
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["judge_mode"] == "preflight"
    assert payload["metadata"]["judge_mode"] == "preflight"
    assert payload["metadata"]["judge_enabled"] is False
    assert payload["metadata"]["judge_cost_preflight"]["estimated_tokens_upper_bound"] > 0
    assert payload["cases"][0]["faithfulness"] == "not_run"


def test_output_judge_mode_reflects_actual_execution(tmp_path):
    result = _result()
    target = tmp_path / "result.json"
    metadata = {"judge_mode": "not_run"}

    eval_ragas._write_outputs(
        target,
        args=argparse.Namespace(no_judge=False),
        aggregate=eval_ragas._aggregate("cloud", [result], judge_enabled=False),
        results=[result],
        metadata=metadata,
        exit_code=2,
    )

    assert json.loads(target.read_text(encoding="utf-8"))["judge_mode"] == "not_run"


def test_formal_judge_consumes_stored_answers_without_generation(tmp_path, monkeypatch):
    generation = tmp_path / "generation.json"
    output = tmp_path / "judged.json"
    generation.write_text(json.dumps(_generation_payload(), ensure_ascii=False), encoding="utf-8")

    class FakeJudge:
        model = "qwen-max"
        total_prompt_tokens = 10
        total_completion_tokens = 5
        total_calls = 1
        estimated_cost_cny = 0.0003
        cost_policy = None

    fake_judge = FakeJudge()
    judged_answers = []

    def fake_judge_call(_client, **kwargs):
        judged_answers.append(kwargs["candidate_answer"])
        return eval_ragas.JudgeVerdict(
            faithfulness=0.9, answer_relevancy=0.8, reasoning="stored answer"
        )

    import app.llm.factory

    monkeypatch.setattr(app.llm.factory, "get_judge_client", lambda: fake_judge)
    monkeypatch.setattr(eval_ragas, "_judge", fake_judge_call)

    exit_code = eval_ragas._judge_existing_report(
        generation, output, max_total_cost_cny=20
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert judged_answers == ["answer"]
    assert payload["judge_mode"] == "qwen"
    assert payload["metadata"]["judge_mode"] == "qwen"
    assert payload["metadata"]["judge_enabled"] is True
    assert payload["cases"][0]["answer"] == "answer"
    assert payload["cases"][0]["faithfulness"] == 0.9
