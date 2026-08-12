"""本地 Docker + Ollama 真实验收编排器。

每次运行创建独立 Compose project、端口、volume 与证据目录；失败时保留现场，
不自动 down 或删除任何资源。长时间 Docker 构建会将实时纯文本进度同时写入
控制台和证据文件，避免因捕获输出而失去诊断信息。
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import queue
import secrets
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"


@dataclass(frozen=True)
class Command:
    label: str
    args: list[str]
    cwd: Path
    env: dict[str, str]
    log: Path
    timeout_seconds: int


@dataclass(frozen=True)
class CommandResult:
    label: str
    passed: bool
    returncode: int | None
    duration_seconds: float
    timed_out: bool
    process_id: int | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _redacted_command(args: list[str]) -> str:
    """Render commands for evidence without ever serializing credential values."""
    return subprocess.list2cmdline(args)


def _pump_output(stream: object, lines: queue.Queue[str]) -> None:
    assert hasattr(stream, "readline")
    while True:
        line = stream.readline()
        if not line:
            return
        lines.put(line)


def _echo(line: str) -> None:
    """Mirror child output without allowing a Windows console codepage to abort acceptance."""
    try:
        sys.stdout.write(line)
        sys.stdout.flush()
    except UnicodeEncodeError:
        # Evidence is already durably written as UTF-8. Console mirroring is
        # best-effort only; writing UTF-8 bytes avoids losing the whole run.
        sys.stdout.buffer.write(line.encode("utf-8", errors="replace"))
        sys.stdout.buffer.flush()


def _run(command: Command) -> CommandResult:
    """Run one command with streamed output, a deadline, and a durable log."""
    command.log.parent.mkdir(parents=True, exist_ok=True)
    started_at = _utc_now()
    started = time.monotonic()
    header = (
        f"label={command.label}\n"
        f"started_at={started_at}\n"
        f"timeout_seconds={command.timeout_seconds}\n"
        f"command={_redacted_command(command.args)}\n"
        "runtime_credentials=environment-only\n\n"
    )
    print(f"\n== {command.label} ==")
    try:
        process = subprocess.Popen(
            command.args,
            cwd=command.cwd,
            env=command.env,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
    except OSError as exc:
        command.log.write_text(header + f"launch_error={type(exc).__name__}: {exc}\n", encoding="utf-8")
        print(f"无法启动命令：{type(exc).__name__}: {exc}", file=sys.stderr)
        return CommandResult(command.label, False, None, 0.0, False)

    lines: queue.Queue[str] = queue.Queue()
    reader = threading.Thread(target=_pump_output, args=(process.stdout, lines), daemon=True)
    reader.start()
    timed_out = False
    with command.log.open("w", encoding="utf-8", newline="") as handle:
        handle.write(header)
        while process.poll() is None:
            try:
                line = lines.get(timeout=0.5)
            except queue.Empty:
                if time.monotonic() - started >= command.timeout_seconds:
                    timed_out = True
                    timeout_message = (
                        f"\nTIMED_OUT after {command.timeout_seconds} seconds; "
                        f"client process PID {process.pid} was intentionally left running.\n"
                    )
                    handle.write(timeout_message)
                    handle.flush()
                    print(timeout_message, file=sys.stderr, end="")
                    break
                continue
            handle.write(line)
            handle.flush()
            _echo(line)

        while not lines.empty():
            line = lines.get_nowait()
            handle.write(line)
            _echo(line)
        if timed_out:
            duration = round(time.monotonic() - started, 2)
            handle.write(f"\nfinished_at={_utc_now()}\nexit=unknown\nduration_seconds={duration}\n")
            return CommandResult(command.label, False, None, duration, True, process.pid)

        returncode = process.wait(timeout=5)
        duration = round(time.monotonic() - started, 2)
        handle.write(f"\nfinished_at={_utc_now()}\nexit={returncode}\nduration_seconds={duration}\n")

    passed = returncode == 0 and not timed_out
    return CommandResult(command.label, passed, returncode, duration, timed_out, process.pid)


def _wait_http_ok(url: str, timeout_seconds: int, log: Path) -> bool:
    deadline = time.monotonic() + timeout_seconds
    attempts = 0
    last_error = "not-attempted"
    while time.monotonic() < deadline:
        attempts += 1
        try:
            with urlopen(url, timeout=5) as response:  # noqa: S310 - fixed localhost URL
                if response.status == 200:
                    log.write_text(f"url={url}\nattempts={attempts}\nstatus=200\n", encoding="utf-8")
                    return True
                last_error = f"status={response.status}"
        except (URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(3)
    log.write_text(f"url={url}\nattempts={attempts}\nstatus=timeout\nlast_error={last_error}\n", encoding="utf-8")
    return False


def _check_port_available(label: str, port: int, evidence_dir: Path) -> CommandResult:
    """Fail before the Docker build when an explicitly requested host port is busy."""
    started = time.monotonic()
    log = evidence_dir / f"00-{label}.log"
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("0.0.0.0", port))
    except OSError as exc:
        log.write_text(
            f"label={label}\nport={port}\navailable=false\nerror={type(exc).__name__}: {exc}\n",
            encoding="utf-8",
        )
        print(f"端口 {port} 已被占用；请使用 --{'backend' if label == 'backend-port' else 'frontend'}-port 指定空闲端口。", file=sys.stderr)
        return CommandResult(label, False, None, round(time.monotonic() - started, 2), False)
    log.write_text(f"label={label}\nport={port}\navailable=true\n", encoding="utf-8")
    return CommandResult(label, True, 0, round(time.monotonic() - started, 2), False)


def _collect_diagnostics(
    *, compose: list[str], env: dict[str, str], evidence_dir: Path, reason: str, timeout_seconds: int
) -> None:
    """Collect non-destructive diagnostics without masking the original failure."""
    print(f"\n验收中断：{reason}；保留 Compose 现场与证据目录供复查。", file=sys.stderr)
    diagnostics = [
        Command("diagnostic-compose-ps", [*compose, "ps", "--all", "--format", "json"], ROOT, env, evidence_dir / "90-compose-ps.log", timeout_seconds),
        Command("diagnostic-compose-logs", [*compose, "logs", "--no-color", "--tail", "300"], ROOT, env, evidence_dir / "91-compose-logs.log", timeout_seconds),
        Command("diagnostic-docker-version", ["docker", "version"], ROOT, env, evidence_dir / "92-docker-version.log", timeout_seconds),
        Command("diagnostic-docker-disk", ["docker", "system", "df"], ROOT, env, evidence_dir / "93-docker-disk.log", timeout_seconds),
    ]
    for diagnostic in diagnostics:
        _run(diagnostic)


def _write_run_manifest(
    evidence_dir: Path, *, args: argparse.Namespace, results: list[CommandResult], outcome: str, reason: str | None
) -> None:
    payload = {
        "project_name": args.project_name,
        "backend_port": args.backend_port,
        "frontend_port": args.frontend_port,
        "with_rerank": True,
        "qwen_judge": "not-run",
        "outcome": outcome,
        "reason": reason,
        "commands": [
            {
                "label": result.label,
                "passed": result.passed,
                "returncode": result.returncode,
                "duration_seconds": result.duration_seconds,
                "timed_out": result.timed_out,
                "process_id": result.process_id,
            }
            for result in results
        ],
    }
    (evidence_dir / "run.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _resolve_service_container(
    *, compose: list[str], env: dict[str, str], evidence_dir: Path
) -> str | None:
    """Resolve the already-running backend without creating or restarting anything."""
    log = evidence_dir / "17a-container-resolution.log"
    try:
        completed = subprocess.run(
            [*compose, "ps", "-q", "backend"],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.write_text(f"container_resolution_error={type(exc).__name__}: {exc}\n", encoding="utf-8")
        return None
    container_id = completed.stdout.strip()
    log.write_text(
        "command=" + subprocess.list2cmdline([*compose, "ps", "-q", "backend"]) + "\n"
        f"exit={completed.returncode}\ncontainer_id={container_id}\nstderr={completed.stderr}",
        encoding="utf-8",
    )
    return container_id if completed.returncode == 0 and container_id else None


def _stage_container_evaluation_sources(
    *, container_id: str, evidence_dir: Path, stamp: str, env: dict[str, str]
) -> tuple[list[CommandResult], dict[str, str]]:
    """Copy only timestamped evaluator copies into the current container filesystem.

    They live outside the persistent /app/data volume and never replace the image's
    original scripts.  The host-side manifest makes the exact evaluated source auditable.
    """
    staged: dict[str, str] = {}
    results: list[CommandResult] = []
    manifest: list[dict[str, str]] = []
    for label, source in (
        ("retrieval", BACKEND / "scripts" / "eval_retrieval.py"),
        ("rerank_threshold", BACKEND / "scripts" / "eval_rerank_threshold.py"),
    ):
        staged_name = f"_acceptance_{label}_{stamp}.py"
        destination = f"/app/scripts/{staged_name}"
        staged[label] = f"scripts/{staged_name}"
        manifest.append(
            {
                "label": label,
                "host_source": str(source.resolve()),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "container_path": destination,
            }
        )
        result = _run(
            Command(
                f"stage-container-{label}-evaluator",
                ["docker", "cp", str(source), f"{container_id}:{destination}"],
                ROOT,
                env,
                evidence_dir / f"17b-stage-{label}.log",
                60,
            )
        )
        results.append(result)
        if not result.passed:
            break
    (evidence_dir / "17b-evaluator-sources.json").write_text(
        json.dumps({"staged_at": _utc_now(), "container_id": container_id, "sources": manifest}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return results, staged


def _run_container_evaluations(
    *, compose: list[str], env: dict[str, str], evidence_dir: Path, stamp: str,
    timeout_seconds: int, skip_retrieval: bool, skip_threshold: bool,
) -> tuple[list[CommandResult], Path | None, Path | None]:
    """Run the current host evaluator code with the container's real BGE dependencies."""
    results: list[CommandResult] = []
    readiness_probe = "\n".join(
        [
            "import json, urllib.request",
            "from app.auth.jwt import create_access_token",
            "token = create_access_token('acceptance-probe', 'acceptance-probe', 'admin', 'acceptance-probe-org')",
            "request = urllib.request.Request('http://127.0.0.1:8000/api/v1/readiness', headers={'Authorization': f'Bearer {token}'})",
            "with urllib.request.urlopen(request, timeout=30) as response:",
            "    payload = json.load(response)",
            "assert response.status == 200 and payload['ready'] is True",
            "print(json.dumps({'status': response.status, 'ready': payload['ready'], 'checks': payload['checks']}, ensure_ascii=False))",
        ]
    )
    readiness_result = _run(
        Command(
            "authenticated-readiness",
            [*compose, "exec", "-T", "backend", "python", "-c", readiness_probe],
            ROOT,
            env,
            evidence_dir / "17a-authenticated-readiness.log",
            60,
        )
    )
    results.append(readiness_result)
    if not readiness_result.passed:
        return results, None, None
    container_id = _resolve_service_container(compose=compose, env=env, evidence_dir=evidence_dir)
    if container_id is None:
        results.append(CommandResult("resolve-backend-container", False, 1, 0.0, False))
        return results, None, None

    staged_results, staged = _stage_container_evaluation_sources(
        container_id=container_id, evidence_dir=evidence_dir, stamp=stamp, env=env
    )
    results.extend(staged_results)
    if not staged_results or not all(result.passed for result in staged_results):
        return results, None, None

    remote_root = f"/tmp/copilot-acceptance-eval-{stamp}"
    retrieval_report = evidence_dir / f"18-container-retrieval-report-{stamp}.json"
    threshold_report = evidence_dir / f"19-container-rerank-threshold-report-{stamp}.json"
    evaluations = [
        (
            "container-retrieval-evaluation",
            "retrieval",
            f"{remote_root}/retrieval",
            f"{remote_root}/retrieval-report.json",
            retrieval_report,
            skip_retrieval,
        ),
        (
            "container-rerank-threshold",
            "rerank_threshold",
            f"{remote_root}/rerank-threshold",
            f"{remote_root}/rerank-threshold-report.json",
            threshold_report,
            skip_threshold,
        ),
    ]
    for label, source_key, remote_workdir, remote_report, host_report, skipped in evaluations:
        if skipped:
            continue
        result = _run(
            Command(
                label,
                [
                    *compose, "exec", "-T", "backend", "python", staged[source_key],
                    "--workdir", remote_workdir, "--report-file", remote_report,
                ],
                ROOT,
                env,
                evidence_dir / f"{18 if source_key == 'retrieval' else 19}-container-{source_key}.log",
                timeout_seconds,
            )
        )
        results.append(result)
        if result.timed_out:
            # The evaluator may still be using the CPU.  Preserve that scene and do
            # not start a competing phase or attempt a partial report copy.
            return results, retrieval_report if retrieval_report.exists() else None, threshold_report if threshold_report.exists() else None
        copy_result = _run(
            Command(
                f"copy-{source_key}-report",
                ["docker", "cp", f"{container_id}:{remote_report}", str(host_report)],
                ROOT,
                env,
                evidence_dir / f"{18 if source_key == 'retrieval' else 19}-copy-{source_key}-report.log",
                60,
            )
        )
        results.append(copy_result)
        if not result.passed or not copy_result.passed:
            return results, retrieval_report if retrieval_report.exists() else None, threshold_report if threshold_report.exists() else None
    return results, retrieval_report if retrieval_report.exists() else None, threshold_report if threshold_report.exists() else None


def _check_retrieval_gate(report_file: Path, log: Path) -> CommandResult:
    """Require a complete, non-degraded BGE result before accepting retrieval metrics."""
    started = time.monotonic()
    try:
        report = json.loads(report_file.read_text(encoding="utf-8"))
        reranked = next(
            row for row in report["results"] if "Rerank" in str(row["label"])
        )
        passed = (
            report["mode"] == "real"
            and report["document_count"] == 7
            and report["chunk_count"] == 50
            and report["case_count"] == 38
            and report["models"]["reranker"] == "BGEReranker"
            and reranked["rerank_verified"] is True
            and not reranked["rerank_failure_reasons"]
            and reranked["evaluated_case_count"] == 38
        )
        detail = {
            "mode": report.get("mode"),
            "document_count": report.get("document_count"),
            "chunk_count": report.get("chunk_count"),
            "case_count": report.get("case_count"),
            "reranker": report.get("models", {}).get("reranker"),
            "rerank_verified": reranked.get("rerank_verified"),
            "evaluated_case_count": reranked.get("evaluated_case_count"),
            "passed": passed,
        }
        log.write_text(json.dumps(detail, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return CommandResult("retrieval-evidence-gate", passed, 0 if passed else 1, round(time.monotonic() - started, 2), False)
    except (KeyError, OSError, StopIteration, TypeError, ValueError, json.JSONDecodeError) as exc:
        log.write_text(f"gate_error={type(exc).__name__}: {exc}\n", encoding="utf-8")
        return CommandResult("retrieval-evidence-gate", False, 1, round(time.monotonic() - started, 2), False)


def _check_rerank_threshold_gate(report_file: Path, log: Path) -> CommandResult:
    """Reject a degraded, incomplete, or all-empty BGE threshold evaluation."""
    started = time.monotonic()
    try:
        report = json.loads(report_file.read_text(encoding="utf-8"))
        active = next(
            row for row in report["thresholds"] if abs(float(row["threshold"]) - 0.12) < 1e-9
        )
        passed = (
            report["rerank_verified"] is True
            and not report["rerank_failure_reasons"]
            and report["document_count"] == 7
            and report["chunk_count"] == 50
            and report["case_count"] == 38
            and len(report["cases"]) == 38
            and report["models"]["reranker"] == "BGEReranker"
            and float(active["hit_rate"]) > 0
            and float(active["empty_context_rate"]) < 1
        )
        detail = {
            "active_threshold": active["threshold"],
            "hit_rate": active["hit_rate"],
            "empty_context_rate": active["empty_context_rate"],
            "rerank_verified": report.get("rerank_verified"),
            "case_count": report.get("case_count"),
            "chunk_count": report.get("chunk_count"),
            "passed": passed,
        }
        log.write_text(json.dumps(detail, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return CommandResult(
            "rerank-threshold-gate",
            passed,
            0 if passed else 1,
            round(time.monotonic() - started, 2),
            False,
        )
    except (KeyError, OSError, StopIteration, TypeError, ValueError, json.JSONDecodeError) as exc:
        log.write_text(f"gate_error={type(exc).__name__}: {exc}\n", encoding="utf-8")
        return CommandResult("rerank-threshold-gate", False, 1, round(time.monotonic() - started, 2), False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-name", default="copilot-acceptance")
    parser.add_argument("--backend-port", type=int, default=18000)
    parser.add_argument("--frontend-port", type=int, default=15173)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="复用已启动的隔离 Compose 项目；跳过构建、up、端口预检和已有 HTTP 验收",
    )
    parser.add_argument("--skip-playwright", action="store_true")
    parser.add_argument("--skip-credential-rotation", action="store_true")
    parser.add_argument("--skip-sse-check", action="store_true")
    parser.add_argument("--skip-e2e-check", action="store_true")
    parser.add_argument("--skip-retrieval-evaluation", action="store_true")
    parser.add_argument("--skip-rerank-threshold", action="store_true")
    parser.add_argument(
        "--run-persistence-check",
        action="store_true",
        help="in resume mode, verify existing seeded data remains readable after recovery",
    )
    parser.add_argument(
        "--run-concurrent-check",
        action="store_true",
        help="in resume mode, run the low-concurrency SSE/health/history evidence check",
    )
    parser.add_argument("--build-timeout-seconds", type=int, default=1800)
    parser.add_argument("--command-timeout-seconds", type=int, default=900)
    parser.add_argument("--evaluation-timeout-seconds", type=int, default=10_800)
    parser.add_argument("--health-timeout-seconds", type=int, default=360)
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    evidence_dir = args.evidence_dir or BACKEND / "acceptance-evidence" / stamp
    if args.resume_existing:
        absolute_evidence_dir = evidence_dir.resolve()
        if args.evidence_dir is None:
            parser.error("--resume-existing 需要显式指定现有 --evidence-dir")
        if not evidence_dir.is_dir():
            parser.error(f"恢复证据目录不存在：{evidence_dir}")
    else:
        evidence_dir.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(
        {
            "WITH_RERANK": "true",
            "BACKEND_PORT": str(args.backend_port),
            "FRONTEND_PORT": str(args.frontend_port),
            "STARTUP_PROBE_EXTERNAL": "true",
            "COMPOSE_PROGRESS": "plain",
            "BUILDKIT_PROGRESS": "plain",
        }
    )
    compose = ["docker", "compose", "--progress", "plain", "-p", args.project_name]

    password = secrets.token_urlsafe(32)
    runtime_env = {
        **env,
        "COPILOT_SEED_PASSWORD": password,
        "COPILOT_ADMIN_USERNAME": "admin",
        "COPILOT_ADMIN_PASSWORD": password,
        "COPILOT_USER_USERNAME": "demo-user",
        "COPILOT_USER_PASSWORD": password,
        "COPILOT_BASE": f"http://127.0.0.1:{args.backend_port}",
    }
    results: list[CommandResult] = []
    retrieval_report_file: Path | None = None
    rerank_report_file: Path | None = None

    if args.resume_existing:
        resume_check = Command(
            "resume-compose-ps",
            [*compose, "ps", "--all", "--format", "json"],
            ROOT,
            env,
            evidence_dir / "14-resume-compose-ps.log",
            args.command_timeout_seconds,
        )
        result = _run(resume_check)
        results.append(result)
        if not result.passed:
            _collect_diagnostics(compose=compose, env=env, evidence_dir=evidence_dir, reason="Unable to inspect existing Compose project", timeout_seconds=args.command_timeout_seconds)
            _write_run_manifest(evidence_dir, args=args, results=results, outcome="resume-preflight-failed", reason="resume-compose-ps")
            return 2
    else:
        for label, port in (("backend-port", args.backend_port), ("frontend-port", args.frontend_port)):
            result = _check_port_available(label, port, evidence_dir)
            results.append(result)
            if not result.passed:
                _write_run_manifest(evidence_dir, args=args, results=results, outcome="preflight-failed", reason=label)
                return 2

        preflight = [
            Command("docker-engine", ["docker", "info", "--format", "{{.ServerVersion}}"], ROOT, env, evidence_dir / "01-docker-engine.log", args.command_timeout_seconds),
            Command("ollama-chat-model", ["ollama", "show", "qwen2.5:7b"], ROOT, env, evidence_dir / "02-ollama-chat-model.log", args.command_timeout_seconds),
            Command("ollama-embedding-model", ["ollama", "show", "bge-m3"], ROOT, env, evidence_dir / "03-ollama-embedding-model.log", args.command_timeout_seconds),
            Command("compose-config", [*compose, "config", "--quiet"], ROOT, env, evidence_dir / "04-compose-config.log", args.command_timeout_seconds),
        ]
        for command in preflight:
            result = _run(command)
            results.append(result)
            if not result.passed:
                _write_run_manifest(evidence_dir, args=args, results=results, outcome="preflight-failed", reason=command.label)
                return 2

        compose_up = Command("compose-up", [*compose, "up", "--build", "--detach"], ROOT, env, evidence_dir / "05-compose-up.log", args.build_timeout_seconds)
        result = _run(compose_up)
        results.append(result)
        if not result.passed:
            _collect_diagnostics(compose=compose, env=env, evidence_dir=evidence_dir, reason="Compose build/start failed", timeout_seconds=args.command_timeout_seconds)
            _write_run_manifest(evidence_dir, args=args, results=results, outcome="compose-failed", reason="compose-up")
            return 1

    health_url = f"http://127.0.0.1:{args.backend_port}/api/v1/health"
    print(f"\n等待隔离后端健康检查：{health_url}")
    if not _wait_http_ok(health_url, args.health_timeout_seconds, evidence_dir / "05a-backend-health.log"):
        _collect_diagnostics(compose=compose, env=env, evidence_dir=evidence_dir, reason="Backend health check timed out", timeout_seconds=args.command_timeout_seconds)
        _write_run_manifest(evidence_dir, args=args, results=results, outcome="health-timeout", reason="health")
        return 1

    frontend_url = f"http://127.0.0.1:{args.frontend_port}/"
    print(f"验证隔离前端可访问：{frontend_url}")
    if not _wait_http_ok(frontend_url, 60, evidence_dir / "05b-frontend-health.log"):
        _collect_diagnostics(compose=compose, env=env, evidence_dir=evidence_dir, reason="Frontend HTTP check timed out", timeout_seconds=args.command_timeout_seconds)
        _write_run_manifest(evidence_dir, args=args, results=results, outcome="frontend-timeout", reason="frontend")
        return 1

    if args.resume_existing:
        # 当前已启动镜像不可重建。凭据轮换通过容器内已安装的依赖直接更新固定
        # 演示账号；源码中新加的 --rotate-password 会在后续镜像构建时生效。
        # 明文仍仅由 docker compose exec 继承的环境变量传递。
        rotate_script = "\n".join(
            [
                "import bcrypt, os",
                "from app.storage.db import session_scope",
                "from app.storage.models import User",
                "password = os.environ['COPILOT_SEED_PASSWORD']",
                "password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=12)).decode()",
                "with session_scope() as session:",
                "    users = [session.get(User, user_id) for user_id in ('user-admin', 'user-regular')]",
                "    if any(user is None for user in users): raise RuntimeError('isolated demo users are missing')",
                "    for user in users: user.password_hash = password_hash",
                "print('Rotated passwords for fixed isolated demo accounts.')",
            ]
        )
        commands: list[Command] = []
        if not args.skip_credential_rotation:
            commands.append(
                Command(
                    "rotate-seed-users",
                    [*compose, "exec", "-T", "-e", "COPILOT_SEED_PASSWORD", "backend", "python", "-c", rotate_script],
                    ROOT,
                    runtime_env,
                    evidence_dir / "15-rotate-seed-users.log",
                    args.command_timeout_seconds,
                )
            )
        if not args.skip_sse_check:
            commands.insert(1, Command("sse-check", [*compose, "exec", "-T", "-e", "COPILOT_USER_USERNAME", "-e", "COPILOT_USER_PASSWORD", "backend", "python", "scripts/sse_check.py"], ROOT, runtime_env, evidence_dir / "16-sse-check.log", args.command_timeout_seconds))
        if not args.skip_e2e_check:
            insert_at = 2 if not args.skip_sse_check else 1
            commands.insert(insert_at, Command("e2e-check", [*compose, "exec", "-T", "-e", "COPILOT_ADMIN_USERNAME", "-e", "COPILOT_ADMIN_PASSWORD", "backend", "python", "scripts/e2e_check.py"], ROOT, runtime_env, evidence_dir / "17-e2e-check.log", args.command_timeout_seconds))
        if args.run_persistence_check:
            commands.append(
                Command(
                    "recovery-persistence-check",
                    [sys.executable, "scripts/persistence_check.py", "--report", str(evidence_dir / "22-recovery-persistence.json")],
                    BACKEND,
                    runtime_env,
                    evidence_dir / "22-recovery-persistence.log",
                    args.command_timeout_seconds,
                )
            )
        if args.run_concurrent_check:
            commands.append(
                Command(
                    "low-concurrency-check",
                    [sys.executable, "scripts/concurrent_check.py", "--report", str(evidence_dir / "32-low-concurrency-report.json")],
                    BACKEND,
                    runtime_env,
                    evidence_dir / "32-low-concurrency.log",
                    args.command_timeout_seconds,
                )
            )
    else:
        commands = [
            Command("seed-users", [*compose, "exec", "-T", "-e", "COPILOT_SEED_PASSWORD", "backend", "python", "scripts/seed_users.py"], ROOT, runtime_env, evidence_dir / "06-seed-users.log", args.command_timeout_seconds),
            Command("seed-kb", [*compose, "exec", "-T", "-e", "COPILOT_ADMIN_USERNAME", "-e", "COPILOT_ADMIN_PASSWORD", "backend", "python", "scripts/seed_kb.py", "--docs-dir", "data/docs_k8s"], ROOT, runtime_env, evidence_dir / "07-seed-kb.log", args.command_timeout_seconds),
            Command("acceptance-http", [sys.executable, "scripts/acceptance_check.py", "--evidence-file", str(evidence_dir / "acceptance.json")], BACKEND, runtime_env, evidence_dir / "08-acceptance-http.log", args.command_timeout_seconds),
            Command("e2e-check", [*compose, "exec", "-T", "-e", "COPILOT_ADMIN_USERNAME", "-e", "COPILOT_ADMIN_PASSWORD", "backend", "python", "scripts/e2e_check.py"], ROOT, runtime_env, evidence_dir / "09-e2e-check.log", args.command_timeout_seconds),
            Command("sse-check", [*compose, "exec", "-T", "-e", "COPILOT_USER_USERNAME", "-e", "COPILOT_USER_PASSWORD", "backend", "python", "scripts/sse_check.py"], ROOT, runtime_env, evidence_dir / "10-sse-check.log", args.command_timeout_seconds),
            Command("retrieval-evaluation", [*compose, "exec", "-T", "backend", "python", "scripts/eval_retrieval.py"], ROOT, env, evidence_dir / "11-retrieval-evaluation.log", args.command_timeout_seconds),
            Command("rerank-threshold", [*compose, "exec", "-T", "backend", "python", "scripts/eval_rerank_threshold.py"], ROOT, env, evidence_dir / "12-rerank-threshold.log", args.command_timeout_seconds),
        ]
    if not args.skip_playwright:
        commands.append(
            Command(
                "playwright",
                ["npm.cmd", "run", "e2e"],
                ROOT / "frontend",
                {
                    **runtime_env,
                    "COPILOT_E2E_API": f"http://127.0.0.1:{args.backend_port}",
                    "COPILOT_E2E_OUTPUT_DIR": str(evidence_dir.resolve() / "playwright-output"),
                    "COPILOT_E2E_REPORT_DIR": str(evidence_dir.resolve() / "playwright-report"),
                },
                evidence_dir / "13-playwright.log",
                args.command_timeout_seconds,
            )
        )

    passed = True
    for command in commands:
        result = _run(command)
        results.append(result)
        passed = result.passed and passed

    if args.resume_existing and (not args.skip_retrieval_evaluation or not args.skip_rerank_threshold):
        evaluation_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        evaluation_results, retrieval_report_file, rerank_report_file = _run_container_evaluations(
            compose=compose,
            env=env,
            evidence_dir=evidence_dir,
            stamp=evaluation_stamp,
            timeout_seconds=args.evaluation_timeout_seconds,
            skip_retrieval=args.skip_retrieval_evaluation,
            skip_threshold=args.skip_rerank_threshold,
        )
        results.extend(evaluation_results)
        passed = all(result.passed for result in evaluation_results) and passed

    if args.resume_existing and not args.skip_retrieval_evaluation:
        if retrieval_report_file is None:
            retrieval_gate = CommandResult("retrieval-evidence-gate", False, 1, 0.0, False)
            (evidence_dir / "18b-retrieval-evidence-gate.json").write_text(
                "gate_error=missing_container_retrieval_report\n", encoding="utf-8"
            )
        else:
            retrieval_gate = _check_retrieval_gate(
                retrieval_report_file,
                evidence_dir / "18b-retrieval-evidence-gate.json",
            )
        results.append(retrieval_gate)
        passed = retrieval_gate.passed and passed

    if args.resume_existing and not args.skip_rerank_threshold:
        if rerank_report_file is None:
            threshold_gate = CommandResult("rerank-threshold-gate", False, 1, 0.0, False)
            (evidence_dir / "19b-rerank-threshold-gate.json").write_text(
                "gate_error=missing_container_rerank_threshold_report\n", encoding="utf-8"
            )
        else:
            threshold_gate = _check_rerank_threshold_gate(
                rerank_report_file,
                evidence_dir / "19b-rerank-threshold-gate.json",
            )
        results.append(threshold_gate)
        passed = threshold_gate.passed and passed

    _collect_diagnostics(compose=compose, env=env, evidence_dir=evidence_dir, reason="final-state capture", timeout_seconds=args.command_timeout_seconds)
    outcome = "passed" if passed else "acceptance-failed"
    _write_run_manifest(evidence_dir, args=args, results=results, outcome=outcome, reason=None if passed else "one-or-more-checks-failed")
    print(f"\n{'验收通过' if passed else '验收失败'}；证据目录：{evidence_dir}")
    print(f"容器未自动停止。复查后如需清理，请手动执行：docker compose -p {args.project_name} down")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
