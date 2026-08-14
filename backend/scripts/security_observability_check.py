"""Create redacted configuration, trace, and evidence hygiene checks."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# These markers are intentionally high-confidence. Generic names such as
# "password" are not secrets by themselves and would make a report misleading.
SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("api_key", re.compile(r"(?<![A-Za-z0-9._-])sk-(?:ws-)?[A-Za-z0-9._-]{20,}(?![A-Za-z0-9._-])")),
    ("github_pat", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b")),
    ("github", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("private_key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
)

CONFIG_TESTS = (
    "tests/test_prod_guardrails.py",
    "tests/test_acceptance_evidence.py",
    "tests/contract/test_contract_basics.py::test_trace_id_header_echoed",
    "tests/contract/test_contract_history.py::test_conversation_detail_carries_trace",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _tracked_files() -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    return [ROOT / line for line in completed.stdout.splitlines() if line]


def _scan(paths: list[Path]) -> tuple[int, list[dict[str, object]]]:
    files_scanned = 0
    findings: list[dict[str, object]] = []
    for path in paths:
        if not path.is_file() or path.stat().st_size > 2_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        files_scanned += 1
        for line_number, line in enumerate(text.splitlines(), start=1):
            for kind, pattern in SECRET_PATTERNS:
                if pattern.search(line):
                    # Never retain the matching value; location and category are enough to remediate.
                    try:
                        reported_path = path.relative_to(ROOT)
                    except ValueError:
                        reported_path = path
                    findings.append(
                        {
                            "path": str(reported_path).replace("\\", "/"),
                            "line": line_number,
                            "kind": kind,
                            "value": "[REDACTED]",
                        }
                    )
    return files_scanned, findings


def _run_tests(evidence_dir: Path, timeout_seconds: int) -> dict[str, object]:
    started = time.monotonic()
    log = evidence_dir / "31-security-observability-tests.log"
    command = [sys.executable, "-m", "pytest", *CONFIG_TESTS, "-q"]
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
    log.write_text(
        f"timeout_seconds={timeout_seconds}\nruntime_credentials=not-used\n\n{output}"
        f"\nexit={returncode if returncode is not None else 'timeout'}\n",
        encoding="utf-8",
    )
    return {
        "node_ids": list(CONFIG_TESTS),
        "log": log.name,
        "returncode": returncode,
        "timed_out": timed_out,
        "duration_seconds": round(time.monotonic() - started, 2),
        "passed": returncode == 0 and not timed_out,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="配置安全与可观测性证据检查")
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")

    evidence_dir = args.evidence_dir.resolve()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    started_at = _utc_now()
    source_count, source_findings = _scan(_tracked_files())
    evidence_paths = [path for path in evidence_dir.rglob("*") if path.is_file()]
    evidence_count, evidence_findings = _scan(evidence_paths)
    tests = _run_tests(evidence_dir, args.timeout_seconds)
    payload = {
        "schema_version": 1,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "scope": {
            "source": "git tracked and unignored working-tree files",
            "evidence": str(evidence_dir),
            "history": "not-scanned-in-this-stage",
        },
        "source_secret_scan": {
            "files_scanned": source_count,
            "findings": source_findings,
            "passed": not source_findings,
        },
        "evidence_secret_scan": {
            "files_scanned": evidence_count,
            "findings": evidence_findings,
            "passed": not evidence_findings,
        },
        "production_guardrails_and_trace": tests,
        "passed": not source_findings and not evidence_findings and tests["passed"],
    }
    report = evidence_dir / "31-security-observability-report.json"
    report.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report), "passed": payload["passed"]}, ensure_ascii=False))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
