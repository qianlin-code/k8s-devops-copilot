import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.config import get_settings
from app.errors import RerankerUnavailableError
from app.rag.vector_store import ScoredChunk


def _enable_hf_offline_if_cached() -> bool:
    """本地已有 bge-reranker 缓存时启用 HF 离线模式，返回是否启用。

    FlagEmbedding 加载模型时会向 HuggingFace 发 HEAD 请求校验版本。
    国内网络下每个文件连接超时 10s、再重试 5 次 —— 实测让首个请求白等 300s，
    而模型本来就在本地、最终也加载成功了。

    只在缓存已存在时启用，缓存缺失时保持在线，避免挡住首次下载。
    """
    if os.environ.get("HF_HUB_OFFLINE") or os.environ.get("TRANSFORMERS_OFFLINE"):
        return False  # 用户已显式设置，不覆盖
    cache = Path(
        os.environ.get("HF_HOME")
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
        or (Path.home() / ".cache" / "huggingface")
    )
    hub = cache / "hub" if (cache / "hub").exists() else cache
    if not any(hub.glob("models--*bge-reranker*")):
        return False
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return True


# 必须在 import huggingface_hub / transformers 之前执行：
# 那些模块在 import 时就把 HF_HUB_OFFLINE 固化成模块常量了。
_enable_hf_offline_if_cached()


@dataclass(slots=True)
class RerankedChunk:
    chunk: ScoredChunk
    rerank_score: float
    rank_before: int
    rank_after: int


class Reranker(Protocol):
    name: str

    def rerank(
        self, query: str, chunks: list[ScoredChunk], top_n: int
    ) -> list[RerankedChunk]:
        ...


class BGEReranker:
    """本地 BGE 交叉编码器。6G 显存下 bge-reranker-base + fp16 足够。

    首次调用才加载模型（懒加载），避免未用到 Rerank 的场景白占显存。
    """

    name = "bge"

    def __init__(self, model_name: str, use_fp16: bool, device: str = "auto") -> None:
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.device = device
        self.resolved_device: str | None = None
        self._model = None
        self._lock = threading.Lock()

    def _resolve_device(self) -> str | None:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return None

    @staticmethod
    def _prefer_local_files() -> None:
        """模型已在本地缓存时不要联网校验版本。

        只设环境变量不够：huggingface_hub 在 import 时就把 HF_HUB_OFFLINE
        读进了模块常量，之后改 os.environ 不生效（实测预热仍白等 294s）。
        所以这里同时改运行时常量。
        """
        if not _enable_hf_offline_if_cached():
            return
        # 已 import 过的模块要改常量，否则环境变量对它们无效
        for mod, attr in (
            ("huggingface_hub.constants", "HF_HUB_OFFLINE"),
            ("transformers.utils.hub", "_is_offline_mode"),
        ):
            module = sys.modules.get(mod)
            if module is not None and hasattr(module, attr):
                setattr(module, attr, True)

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                from FlagEmbedding import FlagReranker
            except ImportError as exc:
                raise RerankerUnavailableError(
                    "FlagEmbedding is not installed; "
                    "install with: uv pip install -r pyproject.toml --extra rerank"
                ) from exc

            self._prefer_local_files()
            device = self._resolve_device()
            # CPU 上 fp16 会显著变慢甚至不支持，强制关掉
            use_fp16 = self.use_fp16 and device != "cpu"
            kwargs: dict[str, object] = {"use_fp16": use_fp16}
            if device:
                kwargs["devices"] = device
            try:
                self._model = FlagReranker(self.model_name, **kwargs)
            except TypeError:
                # 老版本 FlagEmbedding 不接受 devices 参数
                try:
                    self._model = FlagReranker(self.model_name, use_fp16=use_fp16)
                except Exception as exc:
                    raise RerankerUnavailableError(
                        f"Failed to load reranker {self.model_name}: {type(exc).__name__}"
                    ) from exc
            except Exception as exc:
                raise RerankerUnavailableError(
                    f"Failed to load reranker {self.model_name}: {type(exc).__name__}: {exc}"[
                        :300
                    ]
                ) from exc
            self.resolved_device = device or "unknown"

    def rerank(
        self, query: str, chunks: list[ScoredChunk], top_n: int
    ) -> list[RerankedChunk]:
        if not chunks:
            return []
        self._ensure_model()
        assert self._model is not None
        # 用 contextual_text 而非裸 text：与入库嵌入保持同一文本表示，
        # 否则脱离标题上下文的段落会被误判为不相关。
        pairs = [[query, c.contextual_text] for c in chunks]
        try:
            raw = self._model.compute_score(pairs, normalize=True)
        except Exception as exc:
            raise RerankerUnavailableError(
                f"Rerank inference failed: {type(exc).__name__}"
            ) from exc
        scores = [float(raw)] if isinstance(raw, (int, float)) else [float(s) for s in raw]
        return _assemble(chunks, scores, top_n)


class IdentityReranker:
    """降级实现：保持融合顺序，仅截断。用于未装模型或评估对照组。"""

    name = "identity"

    def rerank(
        self, query: str, chunks: list[ScoredChunk], top_n: int
    ) -> list[RerankedChunk]:
        scores = [1.0 / (i + 1) for i in range(len(chunks))]
        return _assemble(chunks, scores, top_n)


def _assemble(
    chunks: list[ScoredChunk], scores: list[float], top_n: int
) -> list[RerankedChunk]:
    paired = list(enumerate(zip(chunks, scores, strict=True)))
    paired.sort(key=lambda item: -item[1][1])
    out: list[RerankedChunk] = []
    for new_rank, (old_index, (chunk, score)) in enumerate(paired[:top_n], start=1):
        out.append(
            RerankedChunk(
                chunk=chunk,
                rerank_score=score,
                rank_before=old_index + 1,
                rank_after=new_rank,
            )
        )
    return out


_reranker: Reranker | None = None


def get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        settings = get_settings()
        _reranker = BGEReranker(
            settings.rerank_model,
            settings.rerank_use_fp16,
            settings.rerank_device,
        )
    return _reranker


def preload_reranker() -> None:
    """提前把模型载入内存/显存。

    同步阻塞，调用方负责放到线程里。替身 reranker 没有 _ensure_model，直接跳过。
    """
    reranker = get_reranker()
    ensure = getattr(reranker, "_ensure_model", None)
    if callable(ensure):
        ensure()


def set_reranker(instance: Reranker) -> None:
    global _reranker
    _reranker = instance


def reset_reranker() -> None:
    global _reranker
    _reranker = None
