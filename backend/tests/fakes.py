"""契约测试用的外部依赖替身。不触碰真实 LLM / Embedding / Rerank 服务。"""

import hashlib
import json
from typing import Any

from app.rag.bm25_index import tokenize
from app.rag.reranker import RerankedChunk
from app.rag.vector_store import ScoredChunk

DIM = 64


class FakeEmbeddingClient:
    """确定性哈希向量，同词汇文本互相靠近。"""

    model = "fake-embedding"
    dim = DIM

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_one(self, text: str) -> list[float]:
        return self._vec(text)

    def _vec(self, text: str) -> list[float]:
        vec = [0.0] * DIM
        for token in text.lower().split():
            idx = int(hashlib.md5(token.encode()).hexdigest(), 16) % DIM
            vec[idx] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]


class ScriptedLLMClient:
    """按调用意图分派的 LLM 替身。

    按意图而非调用顺序取脚本，这样上下文摘要、查询改写等额外调用
    不会让路由/校验脚本错位，测试才能只声明关心的分支。
    """

    model = "fake-llm"

    _INTENTS = ("router", "sufficiency", "summary", "rewrite", "answer")

    def __init__(self) -> None:
        self._queues: dict[str, list[Any]] = {k: [] for k in self._INTENTS}
        self.calls: list[str] = []

    def queue(self, *items: Any) -> "ScriptedLLMClient":
        """按形状分派到对应意图队列，调用方不必关心实际调用顺序。"""
        for item in items:
            if isinstance(item, str):
                self._queues["answer"].append(item)
            elif "action" in item:
                self._queues["router"].append(item)
            elif "sufficient" in item:
                self._queues["sufficiency"].append(item)
            elif "rewritten" in item:
                self._queues["rewrite"].append(item)
            else:
                raise ValueError(f"cannot infer intent for scripted item: {item}")
        return self

    def queue_route(self, *items: Any) -> "ScriptedLLMClient":
        self._queues["router"].extend(items)
        return self

    def queue_sufficiency(self, *items: Any) -> "ScriptedLLMClient":
        self._queues["sufficiency"].extend(items)
        return self

    def queue_answer(self, *items: str) -> "ScriptedLLMClient":
        self._queues["answer"].extend(items)
        return self

    def queue_summary(self, *items: str) -> "ScriptedLLMClient":
        self._queues["summary"].extend(items)
        return self

    def reset(self) -> "ScriptedLLMClient":
        """丢弃未消耗的脚本，避免上一场景残留影响下一场景。"""
        for queue in self._queues.values():
            queue.clear()
        self.calls.clear()
        return self

    def turn(
        self,
        route: dict[str, Any],
        sufficiency: dict[str, Any] | None = None,
        answer: str = "这是替身生成的回答。",
    ) -> "ScriptedLLMClient":
        """声明一个完整问答轮次。"""
        self.queue_route(route)
        if sufficiency is not None:
            self.queue_sufficiency(sufficiency)
        self.queue_answer(answer)
        return self

    def chat(self, messages: list[dict[str, str]], **_: Any) -> str:
        intent = self._classify(messages)
        self.calls.append(intent)
        queue = self._queues[intent]
        if queue:
            item = queue.pop(0)
            return item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
        return self._default(intent)

    def structured(self, messages: list[dict[str, str]], schema: type, **_: Any) -> Any:
        return schema.model_validate_json(self.chat(messages))

    def _classify(self, messages: list[dict[str, str]]) -> str:
        system = messages[0].get("content", "") if messages else ""
        if "决策路由器" in system:
            return "router"
        if "充分性审核员" in system:
            return "sufficiency"
        if "对话摘要器" in system:
            return "summary"
        if "查询改写器" in system:
            return "rewrite"
        return "answer"

    def _default(self, intent: str) -> str:
        if intent == "router":
            return json.dumps(
                {
                    "action": "answer",
                    "reasoning": "替身默认直接作答",
                    "confidence": 0.5,
                },
                ensure_ascii=False,
            )
        if intent == "sufficiency":
            return json.dumps(
                {"sufficient": True, "reasoning": "替身默认判定充分"}, ensure_ascii=False
            )
        if intent == "summary":
            return "替身生成的历史摘要。"
        if intent == "rewrite":
            return json.dumps({"rewritten": "替身改写查询", "keywords": []}, ensure_ascii=False)
        return "替身默认回答。"


class KeywordReranker:
    """按查询词命中率给归一化分，模拟 BGE 的相关性判别。

    复用 BM25 的中文分词，否则 .split() 对中文只能切出极少词元，
    相关性恒为 0，会把所有片段误判成不相关。
    """

    name = "keyword-fake"

    def rerank(
        self, query: str, chunks: list[ScoredChunk], top_n: int
    ) -> list[RerankedChunk]:
        terms = {t for t in tokenize(query) if len(t) >= 2}
        scored = []
        for position, chunk in enumerate(chunks):
            body = set(tokenize(chunk.text))
            hits = len(terms & body)
            scored.append((position, chunk, hits / len(terms) if terms else 0.0))
        scored.sort(key=lambda item: -item[2])
        return [
            RerankedChunk(
                chunk=chunk,
                rerank_score=score,
                rank_before=position + 1,
                rank_after=rank,
            )
            for rank, (position, chunk, score) in enumerate(scored[:top_n], start=1)
        ]
