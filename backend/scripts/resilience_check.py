"""Run deterministic offline resilience gates and emit a redacted JSON report.

The checks use the test doubles configured by ``tests/conftest.py``. They do
not contact Ollama, mutate the acceptance Docker project, or treat offline
results as proof that a real dependency outage was exercised.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CHECK_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "agent_convergence",
        (
            "tests/contract/test_contract_chat.py::test_max_steps_exceeded_branch",
            "tests/contract/test_contract_chat.py::test_tool_failure_is_reported_not_crashed",
            "tests/contract/test_contract_chat.py::test_confirmed_write_does_not_loop_back_to_confirmation",
            "tests/contract/test_contract_chat.py::test_repeated_failed_call_is_skipped_not_reexecuted",
            "tests/contract/test_contract_chat.py::test_write_rejected_executes_nothing",
        ),
    ),
    (
        "dependency_error_mapping",
        (
            "tests/test_error_mapping.py",
            "tests/test_retriever_degradation.py",
        ),
    ),
    (
        "sqlite_lock_handling",
        (
            "tests/test_db_concurrency.py",
            "tests/test_write_lock_duration.py",
        ),
    ),
    (
        "trace_contract_and_persistence",
        (
            "tests/contract/test_contract_basics.py::test_trace_id_header_echoed",
            "tests/contract/test_contract_history.py::test_conversation_detail_carries_trace",
            "tests/contract/test_chat_stream_contract.py::test_stream_emits_progress_then_done",
        ),
    ),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _run_group(
    *, label: str, node_ids: tuple[str, ...], python: str, evidence_dir: Path, timeout_seconds: int
) -> dict[str, object]:
    started = time.monotonic()
    log_file = evidence_dir / f"30-resilience-{label}.log"
    command = [python, "-m", "pytest", *node_ids, "-q"]
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
        output = (completed.stdout or "") + (completed.stderr or "")
        returncode: int | None = completed.returncode
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + (exc.stderr or "")
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        returncode = None
        timed_out = True
    log_file.write_text(
        "label=" + label + "\n"
        f"timeout_seconds={timeout_seconds}\n"
        "runtime_credentials=not-used\n\n"
        + output
        + f"\nexit={returncode if returncode is not None else 'timeout'}\n",
        encoding="utf-8",
    )
    return {
        "label": label,
        "node_ids": list(node_ids),
        "log": log_file.name,
        "returncode": returncode,
        "timed_out": timed_out,
        "duration_seconds": round(time.monotonic() - started, 2),
        "passed": returncode == 0 and not timed_out,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="离线鲁棒性验收门禁")
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")

    evidence_dir = args.evidence_dir.resolve()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    started_at = _utc_now()
    results = [
        _run_group(
            label=label,
            node_ids=node_ids,
            python=sys.executable,
            evidence_dir=evidence_dir,
            timeout_seconds=args.timeout_seconds,
        )
        for label, node_ids in CHECK_GROUPS
    ]
    payload = {
        "schema_version": 1,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "layer": "offline-controlled-injection",
        "real_dependency_outage": "not-exercised",
        "checks": results,
        "passed": all(item["passed"] for item in results),
    }
    report = evidence_dir / "30-resilience-report.json"
    report.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report), "passed": payload["passed"]}, ensure_ascii=False))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
