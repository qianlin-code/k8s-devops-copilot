"""并发压力检查：复现浏览器真实场景下的 database is locked。

浏览器里不止一个请求在跑：SSE 对话进行中，health 每 15s 轮询，
用户还可能切到历史页。原实现在这种并发下会在 INSERT conversations 处撞锁。

运行前需服务已启动: python scripts/concurrent_check.py
环境变量: COPILOT_BASE（默认 http://127.0.0.1:8000）

这是纯 HTTP 客户端脚本，本身不持有数据库连接——它会往目标服务的数据库里写入
真实的会话/消息数据。反复跑会在 dev 库里堆积测试数据（与 e2e_check.py 同理）。
若不想污染 data/app.db，启动一个指向独立 DATABASE_URL 的后端实例（例如
`$env:DATABASE_URL="sqlite:///./data/test.db"; uvicorn app.main:app --port 8001`），
再用 `$env:COPILOT_BASE="http://127.0.0.1:8001"` 指向它。
"""

import argparse
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

BASE = os.environ.get("COPILOT_BASE", "http://127.0.0.1:8000") + "/api/v1"


def _login_headers() -> dict[str, str]:
    username = os.environ.get("COPILOT_USER_USERNAME")
    password = os.environ.get("COPILOT_USER_PASSWORD")
    if not username or not password:
        raise RuntimeError("缺少 COPILOT_USER_USERNAME 或 COPILOT_USER_PASSWORD")
    with httpx.Client(timeout=10.0) as client:
        response = client.post(
            f"{BASE}/auth/login", json={"username": username, "password": password}
        )
    response.raise_for_status()
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


HEADERS: dict[str, str] = {}

QUESTIONS = [
    "ops-demo 命名空间下 api-gateway-7f9c 这个 Pod 一直是 Pending 该怎么处理",
    "api-gateway-7f9c 这个 Pod 的状态是什么",
    "worker-queue 这个 Deployment 副本数不够是什么原因",
]

_lock = threading.Lock()
_failures: list[str] = []
_results: list[tuple[str, str, float]] = []
_details: dict[str, dict[str, object]] = {}


def _record(
    label: str,
    outcome: str,
    elapsed: float,
    failure: str | None = None,
    *,
    details: dict[str, object] | None = None,
) -> None:
    with _lock:
        _results.append((label, outcome, elapsed))
        _details[label] = details or {}
        if failure:
            _failures.append(f"{label}: {failure}")


def stream_chat(index: int, question: str) -> None:
    label = f"stream#{index}"
    started = time.perf_counter()
    # read 是相邻数据块的间隔上限，不是总时长
    timeout = httpx.Timeout(connect=10.0, read=180.0, write=10.0, pool=10.0)
    try:
        with httpx.Client(timeout=timeout) as client:
            with client.stream(
                "POST",
                f"{BASE}/chat/stream",
                headers=HEADERS,
                json={"question": question, "include_trace": True},
            ) as resp:
                if resp.status_code != 200:
                    _record(
                        label,
                        f"HTTP {resp.status_code}",
                        0,
                        f"http_status={resp.status_code}",
                    )
                    return
                buffer = ""
                events: list[str] = []
                error_payload = None
                first_event_ms: int | None = None
                for chunk in resp.iter_text():
                    buffer += chunk
                    blocks = buffer.split("\n\n")
                    buffer = blocks.pop()
                    for block in blocks:
                        kind = data = None
                        for line in block.splitlines():
                            if line.startswith(":"):
                                continue
                            if line.startswith("event: "):
                                kind = line[7:]
                            elif line.startswith("data: "):
                                data = line[6:]
                        if kind:
                            if first_event_ms is None:
                                first_event_ms = round(
                                    (time.perf_counter() - started) * 1000
                                )
                            events.append(kind)
                            if kind == "error" and data:
                                error_payload = json.loads(data)
        elapsed = time.perf_counter() - started
        if error_payload is not None:
            _record(
                label,
                f"error/{error_payload['code']}",
                elapsed,
                f"sse_error={error_payload['code']}",
                details={"event_types": events, "first_event_ms": first_event_ms},
            )
        elif events and events[-1] == "done":
            _record(
                label,
                "done",
                elapsed,
                details={"event_types": events, "first_event_ms": first_event_ms},
            )
        else:
            _record(
                label,
                "incomplete",
                elapsed,
                "event_sequence_incomplete",
                details={"event_types": events, "first_event_ms": first_event_ms},
            )
    except Exception as exc:
        _record(label, type(exc).__name__, time.perf_counter() - started, type(exc).__name__)


def poll_health(stop: threading.Event) -> None:
    """模拟前端 15s 心跳，但压到 1s 以放大并发。

    超时放宽到 10s：GPU 满载时（3 路 SSE 对话并行）连接池会被挤占，
    5s 窗口下偶发客户端超时，但服务端日志无对应 5xx——记录具体异常类型
    才能区分这是客户端超时还是服务端真的挂了。
    """
    count = failures = 0
    exceptions: list[str] = []
    with httpx.Client(timeout=10.0) as client:
        while not stop.is_set():
            count += 1
            try:
                if client.get(f"{BASE}/health").status_code != 200:
                    failures += 1
            except Exception as exc:
                failures += 1
                exceptions.append(type(exc).__name__)
            stop.wait(1.0)
    detail = f"存在失败: {', '.join(exceptions)}" if exceptions else None
    _record(
        "health",
        f"{count - failures}/{count} ok",
        0,
        detail,
        details={
            "requests": count,
            "failures": failures,
            "exception_types": sorted(set(exceptions)),
        },
    )


def poll_history(stop: threading.Event) -> None:
    """模拟用户切到历史页反复刷新。"""
    count = failures = 0
    with httpx.Client(timeout=10.0) as client:
        while not stop.is_set():
            count += 1
            try:
                resp = client.get(
                    f"{BASE}/conversations",
                    headers=HEADERS,
                    params={"limit": 20},
                )
                if resp.status_code != 200:
                    failures += 1
            except Exception:
                failures += 1
            stop.wait(1.5)
    _record(
        "history",
        f"{count - failures}/{count} ok",
        0,
        "history_poll_failure" if failures else None,
        details={"requests": count, "failures": failures},
    )


def _write_report(path: Path, *, started_at: str, total_seconds: float) -> None:
    """Persist aggregate timings without serializing auth headers or responses."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "label": label,
            "outcome": outcome,
            "elapsed_ms": round(elapsed * 1000),
            "details": _details.get(label, {}),
        }
        for label, outcome, elapsed in sorted(_results)
    ]
    payload = {
        "schema_version": 1,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "base_url": BASE,
        "stream_count": len(QUESTIONS),
        "total_elapsed_ms": round(total_seconds * 1000),
        "results": rows,
        "failure_categories": sorted(set(item.split(": ", 1)[-1] for item in _failures)),
        "database_locked_detected": any("database is locked" in item.lower() for item in _failures),
        "runtime_credentials": "environment-only-not-serialized",
        "passed": not _failures,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="低并发 SSE/health/history 真实服务检查")
    parser.add_argument("--report", type=Path, help="write a redacted JSON report")
    args = parser.parse_args()
    started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    global HEADERS
    try:
        httpx.get(f"{BASE}/health", timeout=5.0).raise_for_status()
        HEADERS = _login_headers()
    except Exception as exc:
        print(f"服务未就绪: {exc}")
        if args.report:
            _write_report(args.report, started_at=started_at, total_seconds=0)
        return 1

    stop = threading.Event()
    pollers = [
        threading.Thread(target=poll_health, args=(stop,), daemon=True),
        threading.Thread(target=poll_history, args=(stop,), daemon=True),
    ]
    for t in pollers:
        t.start()

    # 三个对话请求同时发起——这是最容易撞锁的场景
    print(f"  同时发起 {len(QUESTIONS)} 个流式对话 + health/history 轮询...")
    chats = [
        threading.Thread(target=stream_chat, args=(i, q))
        for i, q in enumerate(QUESTIONS, start=1)
    ]
    started = time.perf_counter()
    for t in chats:
        t.start()
    for t in chats:
        t.join()
    stop.set()
    for t in pollers:
        t.join(timeout=5)
    total = time.perf_counter() - started

    print(f"\n  总耗时 {total:.1f}s\n")
    print(f"  {'任务':<12} {'结果':<24} {'耗时':>8}")
    print("  " + "-" * 46)
    for label, outcome, elapsed in sorted(_results):
        shown = f"{elapsed:.1f}s" if elapsed else "-"
        print(f"  {label:<12} {outcome:<24} {shown:>8}")
    print("  " + "-" * 46)

    if _failures:
        print("\n  失败明细:")
        for line in _failures:
            print(f"    {line}")
        locked = [f for f in _failures if "locked" in f.lower()]
        if locked:
            print("\n  仍存在 database is locked")
        if args.report:
            _write_report(args.report, started_at=started_at, total_seconds=total)
        return 1

    print("\n  并发场景全部通过，无 database is locked")
    if args.report:
        _write_report(args.report, started_at=started_at, total_seconds=total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
