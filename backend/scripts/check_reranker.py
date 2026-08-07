"""验证本地 BGE Reranker 是否可用，并检查它是否真的在做语义重排。

首次运行会从 HuggingFace 下载模型（bge-reranker-base 约 1.1 GB）。
若国内网络下载慢，可先设置镜像:
    $env:HF_ENDPOINT = "https://hf-mirror.com"

运行: python scripts/check_reranker.py
"""

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("API_KEY", "check")
os.environ.setdefault("STARTUP_PROBE_EXTERNAL", "false")

from app.config import get_settings  # noqa: E402
from app.errors import RerankerUnavailableError  # noqa: E402
from app.rag.reranker import BGEReranker  # noqa: E402
from app.rag.vector_store import ScoredChunk  # noqa: E402

QUERY = "登录返回 403 是权限不足吗，怎么解决"

# 故意打乱顺序：真正相关的排在后面，能重排上来才说明模型在工作
CANDIDATES = [
    "VPN 出口 IP 频繁变化会触发会话 IP 绑定校验，导致频繁掉线。",
    "Token 过期需要重新登录获取新凭证，默认有效期 8 小时。",
    "企业防火墙拦截出站端口会导致内网访问返回 504 Gateway Timeout。",
    "连续 5 次密码错误会触发风控自动锁定，锁定时长 30 分钟。",
    "账号 permission_level 为 restricted 时未被授予应用访问权限，"
    "登录会返回 403 Forbidden，需要管理员提权到 standard 并刷新权限缓存。",
    "订阅账期逾期未付款，系统会自动暂停服务，付款后 1 小时内恢复。",
]


def main() -> int:
    settings = get_settings()
    print(f"model  : {settings.rerank_model}")
    print(f"device : {settings.rerank_device}  fp16={settings.rerank_use_fp16}")

    try:
        import torch

        print(f"torch  : {torch.__version__}  cuda={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"gpu    : {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("torch  : 未安装")

    reranker = BGEReranker(
        settings.rerank_model, settings.rerank_use_fp16, settings.rerank_device
    )

    print("\n加载模型（首次会下载，请耐心等待）...")
    started = time.perf_counter()
    try:
        reranker._ensure_model()
    except RerankerUnavailableError as exc:
        print(f"\n失败: {exc.message}")
        print("\n未安装时检索链路会自动降级为 RRF 融合顺序，功能不中断，")
        print("但简历里的 Rerank 对比指标需要装上模型才能产出真实数据。")
        return 1
    print(f"加载完成 {time.perf_counter() - started:.1f}s  实际设备={reranker.resolved_device}")

    chunks = [
        ScoredChunk(f"c{i}", "d1", "手册", text, [], i, 0.0)
        for i, text in enumerate(CANDIDATES)
    ]

    started = time.perf_counter()
    ranked = reranker.rerank(QUERY, chunks, top_n=len(chunks))
    elapsed = (time.perf_counter() - started) * 1000

    print(f"\n查询: {QUERY}")
    print(f"重排 {len(chunks)} 条候选，耗时 {elapsed:.0f}ms\n")
    print(f"{'新序':>4} {'原序':>4} {'得分':>8}  片段")
    print("-" * 78)
    for item in ranked:
        moved = item.rank_before - item.rank_after
        arrow = f"↑{moved}" if moved > 0 else (f"↓{-moved}" if moved < 0 else "—")
        print(
            f"{item.rank_after:>4} {item.rank_before:>4} {item.rerank_score:>8.4f} "
            f"{arrow:>4}  {item.chunk.text[:52]}"
        )

    top = ranked[0]
    hit = "403" in top.chunk.text and "permission_level" in top.chunk.text
    print("-" * 78)
    print(f"最相关片段被排到第 1 位: {'是' if hit else '否'}")
    print(f"该片段原本在第 {top.rank_before} 位，得分 {top.rerank_score:.4f}")
    if not hit:
        print("\n警告: 重排结果不符合预期，请检查模型是否正确加载。")
        return 1
    print("\nRerank 工作正常。现在可以跑真实检索评估:")
    print("  python scripts/eval_retrieval.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
