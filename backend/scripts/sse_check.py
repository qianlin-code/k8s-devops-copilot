"""SSE 流式接口真实联调：验证阶段事件按序到达、首个事件足够早。

运行前需先启动后端。
运行: python scripts/sse_check.py
"""

import json
import os
import sys
import time

import httpx

BASE = os.environ.get("COPILOT_BASE", "http://localhost:8000")


def _login_headers() -> dict[str, str]:
    username = os.environ.get("COPILOT_USER_USERNAME")
    password = os.environ.get("COPILOT_USER_PASSWORD")
    if not username or not password:
        print("缺少 COPILOT_USER_USERNAME 或 COPILOT_USER_PASSWORD", file=sys.stderr)
        raise SystemExit(2)
    with httpx.Client(timeout=httpx.Timeout(10.0)) as client:
        response = client.post(
            f"{BASE}/api/v1/auth/login",
            json={"username": username, "password": password},
        )
    if response.status_code != 200:
        print(f"登录失败 HTTP {response.status_code}: {response.text[:400]}", file=sys.stderr)
        raise SystemExit(1)
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def main() -> int:
    payload = {
        "question": "ops-demo 命名空间下 Pod 一直 Pending 该怎么排查？",
        "include_trace": True,
    }
    headers = {**_login_headers(), "Accept": "text/event-stream"}

    started = time.perf_counter()
    first_event_at: float | None = None
    events: list[tuple[float, str, dict]] = []

    # read 是「相邻数据块之间」的间隔上限，不是总时长。
    # 真实模型下单个 LLM 调用可能十几秒不出新事件，默认 5s 会误判超时。
    timeout = httpx.Timeout(connect=10.0, read=180.0, write=10.0, pool=10.0)
    with httpx.Client(timeout=timeout) as client:
        with client.stream(
            "POST", f"{BASE}/api/v1/chat/stream", json=payload, headers=headers
        ) as resp:
            if resp.status_code != 200:
                print(f"HTTP {resp.status_code}: {resp.read().decode('utf-8')[:400]}")
                return 1
            ctype = resp.headers.get("content-type", "")
            if "text/event-stream" not in ctype:
                print(f"content-type 不是 SSE: {ctype}")
                return 1

            buffer = ""
            for chunk in resp.iter_text():
                buffer += chunk
                blocks = buffer.split("\n\n")
                buffer = blocks.pop()
                for block in blocks:
                    event = ""
                    data = ""
                    for line in block.splitlines():
                        if line.startswith("event:"):
                            event = line[6:].strip()
                        elif line.startswith("data:"):
                            data = line[5:].strip()
                    if not event:
                        continue
                    at = time.perf_counter() - started
                    if first_event_at is None:
                        first_event_at = at
                    events.append((at, event, json.loads(data)))

    total = time.perf_counter() - started

    print(f"总耗时      : {total:.1f}s")
    print(f"首个事件    : {first_event_at:.2f}s" if first_event_at else "首个事件    : 无")
    print(f"事件总数    : {len(events)}\n")

    print(f"{'到达(s)':>8}  {'事件':<9} 内容")
    print("-" * 74)
    for at, event, data in events:
        if event == "progress":
            desc = f"[{data['phase']}] {data['label']}"
        elif event == "done":
            desc = f"outcome={data['outcome']} 回答 {len(data['answer'])} 字"
        else:
            desc = f"{data.get('code')}: {data.get('message', '')[:40]}"
        print(f"{at:>8.2f}  {event:<9} {desc}")
    print("-" * 74)

    kinds = [e for _, e, _ in events]
    problems: list[str] = []
    if not events:
        problems.append("没有收到任何事件")
    if kinds and kinds[-1] != "done":
        problems.append(f"最后一个事件应为 done，实际 {kinds[-1]}")
    if first_event_at is not None and first_event_at > 5:
        problems.append(f"首个事件 {first_event_at:.1f}s 才到，用户会误判卡死")
    times = [at for at, e, _ in events if e == "progress"]
    if times != sorted(times):
        problems.append("事件时间非单调")

    if problems:
        print("\n问题:")
        for p in problems:
            print(f"  - {p}")
        return 1

    gaps = [(events[i][0] - events[i - 1][0], events[i][2]) for i in range(1, len(events))]
    worst = max(gaps, key=lambda g: g[0]) if gaps else None
    print("\nSSE 工作正常。")
    if worst and worst[0] > 3:
        label = worst[1].get("label") or worst[1].get("outcome", "?")
        print(f"最长等待间隔 {worst[0]:.1f}s，发生在进入「{label}」之前。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
