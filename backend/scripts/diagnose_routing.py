"""路由决策细节诊断脚本。

检查指定查询在当前运行中的服务里，路由器看到的检索片段是否包含
`is_procedural` 标记、送给 Router 的 prompt 完整内容、以及 decision reasoning。

用于验证 chunk 元数据标记和 prompt 提示是否真正生效。

运行: python scripts/diagnose_routing.py "早上还好好的，现在突然连不上集群内部服务了"
"""

import argparse
import os
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
BASE = os.environ.get("COPILOT_BASE", "http://localhost:8000")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="要诊断的用户问题")
    parser.add_argument("--username", default=os.environ.get("COPILOT_USERNAME"))
    parser.add_argument("--password", default=os.environ.get("COPILOT_PASSWORD"))
    args = parser.parse_args()
    if not args.username or not args.password:
        print(
            "provide --username/--password or set COPILOT_USERNAME/COPILOT_PASSWORD",
            file=sys.stderr,
        )
        return 1

    query = args.query
    with httpx.Client(timeout=httpx.Timeout(120.0)) as client:
        login = client.post(
            f"{BASE}/api/v1/auth/login",
            json={"username": args.username, "password": args.password},
        )
        if login.status_code != 200:
            print(f"Login failed: {login.status_code} {login.text[:300]}")
            return 1
        headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
    # 走完整 chat API，拿回 trace
    payload = {"question": query}
    with httpx.Client(timeout=httpx.Timeout(120.0)) as client:
        resp = client.post(f"{BASE}/api/v1/chat", headers=headers, json=payload)
        if resp.status_code != 200:
            print(f"Chat API failed: {resp.status_code} {resp.text[:300]}")
            return 1
        body = resp.json()

    # 从 trace 找 retrieval 和 router_decision
    retrieval = None
    decision = None
    for event in body.get("trace", {}).get("events", []):
        if event["type"] == "retrieval":
            retrieval = event["data"]
        elif event["type"] == "router_decision":
            decision = event["data"]

    if retrieval is None or decision is None:
        print("No retrieval or router_decision in trace")
        return 1

    print(f"\n查询: {query}\n")
    print(f"路由决策: {decision['action']} (confidence={decision['confidence']})")
    print(f"Reasoning: {decision['reasoning']}\n")
    if decision.get("tool_name"):
        print(f"工具调用: {decision['tool_name']}({decision.get('tool_arguments', {})})\n")

    print(f"检索到 {len(retrieval['chunks'])} 个片段:\n")
    for i, chunk in enumerate(retrieval["chunks"], start=1):
        is_proc = chunk.get("is_procedural", False)
        marker = " [操作步骤]" if is_proc else ""
        print(f"[{i}]{marker} 来源: {chunk['citation_label']}")
        print(f"  Rerank 分数: {chunk.get('rerank_score', 0.0):.4f}")
        print(f"  is_procedural: {is_proc}")
        print(f"  text 前 120 字: {chunk['text'][:120]}...")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
