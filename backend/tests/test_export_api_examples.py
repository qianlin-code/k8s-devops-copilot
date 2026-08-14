from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from scripts import export_api_examples


BACKEND = Path(__file__).resolve().parents[1]
SCRIPT = BACKEND / "scripts" / "export_api_examples.py"
EXAMPLES = BACKEND / "api_examples"


def _write_json(directory: Path, name: str, payload: object) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _run_export(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    for key in ("QWEN_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY"):
        env.pop(key, None)
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=BACKEND,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
        check=False,
    )


def _hash_examples() -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(EXAMPLES.glob("*.json"))
    }


def test_semantic_snapshot_accepts_equivalent_volatile_values(tmp_path: Path) -> None:
    expected = tmp_path / "expected"
    actual = tmp_path / "actual"
    first_uuid = "11111111-1111-4111-8111-111111111111"
    second_uuid = "22222222-2222-4222-8222-222222222222"
    _write_json(
        expected,
        "a.json",
        {
            "conversation_id": first_uuid,
            "trace_id": "a" * 32,
            "created_at": "2026-08-14T07:20:15.123456Z",
            "incident_id": "INC-AAAAAAAAAA",
            "elapsed_ms": 3,
            "namespace": "ops-demo",
        },
    )
    _write_json(expected, "b.json", {"conversation_id": first_uuid})
    _write_json(
        actual,
        "a.json",
        {
            "conversation_id": second_uuid,
            "trace_id": "b" * 32,
            "created_at": "2026-08-15T08:21:16.654321Z",
            "incident_id": "INC-BBBBBBBBBB",
            "elapsed_ms": 19,
            "namespace": "ops-demo",
        },
    )
    _write_json(
        actual,
        "b.json",
        {
            "conversation_id": second_uuid,
            "message": f"Conversation '{second_uuid}' not found",
        },
    )
    expected_payload = json.loads((expected / "b.json").read_text(encoding="utf-8"))
    expected_payload["message"] = f"Conversation '{first_uuid}' not found"
    _write_json(expected, "b.json", expected_payload)

    assert export_api_examples.compare_example_directories(expected, actual) == []


def test_semantic_snapshot_rejects_broken_cross_file_reference(tmp_path: Path) -> None:
    expected = tmp_path / "expected"
    actual = tmp_path / "actual"
    first_uuid = "11111111-1111-4111-8111-111111111111"
    second_uuid = "22222222-2222-4222-8222-222222222222"
    _write_json(expected, "a.json", {"conversation_id": first_uuid})
    _write_json(expected, "b.json", {"conversation_id": first_uuid})
    _write_json(actual, "a.json", {"conversation_id": first_uuid})
    _write_json(actual, "b.json", {"conversation_id": second_uuid})

    differences = export_api_examples.compare_example_directories(expected, actual)

    assert differences == ["changed:b.json"]
    assert first_uuid not in "".join(differences)
    assert second_uuid not in "".join(differences)


def test_semantic_snapshot_rejects_stable_business_change(tmp_path: Path) -> None:
    expected = tmp_path / "expected"
    actual = tmp_path / "actual"
    _write_json(expected, "a.json", {"namespace": "ops-demo", "outcome": "answer"})
    _write_json(actual, "a.json", {"namespace": "default", "outcome": "answer"})

    assert export_api_examples.compare_example_directories(expected, actual) == [
        "changed:a.json"
    ]


def test_semantic_snapshot_rejects_file_set_drift(tmp_path: Path) -> None:
    expected = tmp_path / "expected"
    actual = tmp_path / "actual"
    _write_json(expected, "kept.json", {"ok": True})
    _write_json(expected, "missing.json", {"ok": True})
    _write_json(actual, "kept.json", {"ok": True})
    _write_json(actual, "extra.json", {"ok": True})

    assert export_api_examples.compare_example_directories(expected, actual) == [
        "missing:missing.json",
        "extra:extra.json",
    ]


def test_export_covers_confirmation_approval_and_rejection(tmp_path: Path) -> None:
    output = tmp_path / "examples"
    completed = _run_export("--output-dir", str(output))
    assert completed.returncode == 0, completed.stdout + completed.stderr

    pending = json.loads(
        (output / "chat__write_confirmation_required.json").read_text(encoding="utf-8")
    )
    approved = json.loads(
        (output / "chat_confirm__approved.json").read_text(encoding="utf-8")
    )
    rejected = json.loads(
        (output / "chat_confirm__rejected.json").read_text(encoding="utf-8")
    )

    assert pending["response"]["outcome"] == "write_confirmation_required"
    assert pending["response"]["pending_write"]["tool_name"] == "restart_deployment"
    sedimentation = json.loads(
        (output / "knowledge_sedimentations__marked_pending.json").read_text(
            encoding="utf-8"
        )
    )
    assert sedimentation["response"]["status"] == "pending"
    assert sedimentation["response"]["quality_score"] == export_api_examples.QUALITY_SCORE
    assert (
        sedimentation["response"]["quality_reasoning"]
        == export_api_examples.QUALITY_REASONING
    )
    assert approved["http_status"] == 200
    assert any(
        call["success"] is True
        for call in approved["response"]["trace"]["tool_calls"]
    )
    assert rejected["http_status"] == 200
    assert rejected["response"]["trace"] is None

    for value in ("ops-demo", "worker-queue", "配置修复后重启生效"):
        assert value in export_api_examples.RESTART_QUESTION
    for value in (
        "ops-demo",
        "api-gateway Pod 长期 Pending 需人工介入",
        "资源已确认充足但仍无法调度，需要工程师排查调度器配置。",
        "high",
    ):
        assert value in export_api_examples.INCIDENT_QUESTION


def test_check_mode_preserves_tracked_examples() -> None:
    before = _hash_examples()

    completed = _run_export("--check")

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert _hash_examples() == before
