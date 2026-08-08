"""并发压力检查：复现浏览器真实场景下的 database is locked。

浏览器里不止一个请求在跑：SSE 对话进行中，health 每 15s 轮询，
用户还可能切到历史页。原实现在这种并发下会在 INSERT conversations 处撞锁。

运行前需服务已启动: python scripts/concurrent_check.py
"""

import json
import sys
import threading
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BASE = "http://127.0.0.1:8000/api/v1"
HEADERS = {"X-API-Key": "dev-local-api-key-change-me"}
USER = "u-1001"

QUESTIONS = [
    "账号 u-1001 登录提示 403 Forbidden 该怎么处理",
    "u-1001 的账号状态是什么",
    "欠费了服务什么时候恢复",
]

_lock = threading.Lock()
_failures: list[str] = []
_results: list[tuple[str, str, float]] = []


def _record(label: str, outcome: str, elapsed: float, failure: str | None = None) -> None:
    with _lock:
        _results.append((label, outcome, elapsed))
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
                json={"question": question, "user_id": USER, "include_trace": True},
            ) as resp:
                if resp.status_code != 200:
                    body = resp.read().decode("utf-8")[:200]
                    _record(label, f"HTTP {resp.status_code}", 0, body)
                    return
                buffer = ""
                events: list[str] = []
                error_payload = None
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
                            events.append(kind)
                            if kind == "error" and data:
                                error_payload = json.loads(data)
        elapsed = time.perf_counter() - started
        if error_payload is not None:
            _record(
                label,
                f"error/{error_payload['code']}",
                elapsed,
                f"{error_payload['code']}: {error_payload['message'][:120]}",
            )
        elif events and events[-1] == "done":
            _record(label, "done", elapsed)
        else:
            _record(label, "incomplete", elapsed, f"事件序列异常: {events}")
    except Exception as exc:
        _record(label, type(exc).__name__, time.perf_counter() - started, str(exc)[:150])


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
    _record("health", f"{count - failures}/{count} ok", 0, detail)


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
                    params={"user_id": USER, "limit": 20},
                )
                if resp.status_code != 200:
                    failures += 1
            except Exception:
                failures += 1
            stop.wait(1.5)
    _record("history", f"{count - failures}/{count} ok", 0, "存在失败" if failures else None)


def main() -> int:
    try:
        httpx.get(f"{BASE}/health", timeout=5.0).raise_for_status()
    except Exception as exc:
        print(f"服务未就绪: {exc}")
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
        return 1

    print("\n  并发场景全部通过，无 database is locked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
