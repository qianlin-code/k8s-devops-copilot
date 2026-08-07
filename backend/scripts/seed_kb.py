"""把 data/docs 下的示例文档灌入运行中的服务。

比手写 curl/PowerShell 可靠：统一 UTF-8，避免 shell 的编码与对象包装问题。

运行: python scripts/seed_kb.py
环境变量: COPILOT_BASE（默认 http://localhost:8000）
"""

import os
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
BASE = os.environ.get("COPILOT_BASE", "http://localhost:8000")


def _api_key() -> str:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("API_KEY="):
                return line.split("=", 1)[1].strip()
    return "dev-local-api-key-change-me"


def main() -> int:
    docs = sorted((ROOT / "data" / "docs").glob("*.md"))
    if not docs:
        print("data/docs 下没有文档", file=sys.stderr)
        return 1

    headers = {"X-API-Key": _api_key()}
    with httpx.Client(timeout=httpx.Timeout(300.0)) as client:
        # 先清掉已有文档，保证可重复执行
        existing = client.get(f"{BASE}/api/v1/knowledge/documents", headers=headers)
        if existing.status_code != 200:
            print(f"读取文档列表失败 HTTP {existing.status_code}: {existing.text[:200]}")
            return 1
        for doc in existing.json()["documents"]:
            client.delete(
                f"{BASE}/api/v1/knowledge/documents/{doc['document_id']}",
                headers=headers,
            )

        for path in docs:
            resp = client.post(
                f"{BASE}/api/v1/knowledge/documents",
                headers=headers,
                json={
                    "title": path.stem.replace("_", " "),
                    "content": path.read_text(encoding="utf-8"),
                    "chunk_strategy": "markdown",
                },
            )
            if resp.status_code != 200:
                print(f"  {path.name} 失败 HTTP {resp.status_code}: {resp.text[:300]}")
                return 1
            body = resp.json()
            print(f"  {path.name}: {body['document']['chunk_count']} chunks")

        listing = client.get(
            f"{BASE}/api/v1/knowledge/documents", headers=headers
        ).json()
        print(f"  vectors={listing['vector_count']} bm25={listing['bm25_index_size']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
