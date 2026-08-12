"""把知识库文档灌入运行中的服务。

比手写 curl/PowerShell 可靠：统一 UTF-8，避免 shell 的编码与对象包装问题。

运行: python scripts/seed_kb.py [--docs-dir data/docs_education]
环境变量: COPILOT_BASE（默认 http://localhost:8000）

默认目录 data/docs_k8s 是当前项目场景（K8s 运维排障，CC BY 4.0 授权节选整理）。
data/docs 是迁移前的客服场景遗留文档，仅供历史参考。
"""

import argparse
import os
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
BASE = os.environ.get("COPILOT_BASE", "http://localhost:8000")


def _get_jwt_token() -> str:
    """从运行环境的 admin 凭据换取 JWT token。"""
    username = os.environ.get("COPILOT_ADMIN_USERNAME")
    password = os.environ.get("COPILOT_ADMIN_PASSWORD")
    if not username or not password:
        print("缺少 COPILOT_ADMIN_USERNAME 或 COPILOT_ADMIN_PASSWORD", file=sys.stderr)
        raise SystemExit(2)
    with httpx.Client(timeout=httpx.Timeout(10.0)) as client:
        resp = client.post(
            f"{BASE}/api/v1/auth/login",
            json={"username": username, "password": password},
        )
        if resp.status_code != 200:
            print(f"登录失败 HTTP {resp.status_code}: {resp.text[:200]}", file=sys.stderr)
            sys.exit(1)
        return resp.json()["access_token"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--docs-dir", type=Path, default=ROOT / "data" / "docs_k8s",
        help="知识库文档目录，默认 data/docs_k8s。切换行业示例时指向另一套文档，"
        "例如 data/docs_education",
    )
    args = parser.parse_args()

    docs = sorted(args.docs_dir.glob("*.md"))
    if not docs:
        print(f"{args.docs_dir} 下没有文档", file=sys.stderr)
        return 1

    token = _get_jwt_token()
    headers = {"Authorization": f"Bearer {token}"}
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
