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

os.environ.setdefault("JWT_SECRET_KEY", "check-only-jwt-secret-not-for-production")
os.environ.setdefault("STARTUP_PROBE_EXTERNAL", "false")

from app.config import get_settings  # noqa: E402
from app.errors import RerankerUnavailableError  # noqa: E402
from app.rag.reranker import BGEReranker  # noqa: E402
from app.rag.vector_store import ScoredChunk  # noqa: E402

QUERY = "Pod 一直是 Pending 是资源不足吗，怎么解决"

# 故意打乱顺序：真正相关的排在后面，能重排上来才说明模型在工作
CANDIDATES = [
    "NetworkPolicy 默认拒绝入站流量时，需要显式声明 ingress 规则才能放行。",
    "ImagePullBackOff 通常是镜像名称拼写错误或缺少 imagePullSecrets。",
    "DNS 解析失败时需要检查 CoreDNS Pod 是否正常运行以及 resolv.conf 配置。",
    "CrashLoopBackOff 是容器进程异常退出触发的反复重启，需查看上一次崩溃日志。",
    "Pod 长期停留在 Pending 状态通常是集群 CPU 或内存资源不足导致调度器"
    "无法为其找到合适节点，需要检查资源请求或为集群扩容。",
    "PVC 一直是 Pending 状态可能是 StorageClass 配置错误或后端存储容量不足。",
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
    hit = "Pending" in top.chunk.text and "调度器" in top.chunk.text
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
