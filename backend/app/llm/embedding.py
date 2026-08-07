from openai import OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.errors import NonRetryableError, RetryableError
from app.llm.error_mapping import map_embedding_error

# 本地 Ollama 一次吃太多文本容易断连，分批既降压也让重试粒度更细
_DEFAULT_BATCH_SIZE = 16

_retry_policy = retry(
    retry=retry_if_exception_type(RetryableError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    reraise=True,
)


class EmbeddingClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        dim: int,
        timeout: float = 120.0,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ):
        self.model = model
        self.dim = dim
        self.batch_size = max(1, batch_size)
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._embed_batch(texts[start : start + self.batch_size]))
        return vectors

    @_retry_policy
    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        try:
            resp = self._client.embeddings.create(model=self.model, input=batch)
        except Exception as exc:
            raise map_embedding_error(exc) from exc

        vectors = [item.embedding for item in resp.data]
        if len(vectors) != len(batch):
            raise NonRetryableError(
                "Embedding endpoint returned a different number of vectors",
                details={"expected": len(batch), "actual": len(vectors)},
            )
        for vec in vectors:
            if len(vec) != self.dim:
                raise NonRetryableError(
                    "Embedding dimension mismatch between model output and config",
                    details={
                        "expected": self.dim,
                        "actual": len(vec),
                        "model": self.model,
                    },
                )
        return vectors

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]
