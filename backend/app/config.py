from enum import Enum
from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 开发默认密钥。生产环境沿用这个值会被启动校验拦下（见 _validate_production）。
DEFAULT_DEV_API_KEY = "dev-local-api-key-change-me"
_MIN_PROD_API_KEY_LENGTH = 24


class Provider(str, Enum):
    OLLAMA = "ollama"
    QWEN = "qwen"


class ChunkStrategyName(str, Enum):
    CHAR = "char"
    MARKDOWN = "markdown"


class Environment(str, Enum):
    DEV = "dev"
    PROD = "prod"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    api_key: str = DEFAULT_DEV_API_KEY
    environment: Environment = Environment.DEV
    # 允许跨域的前端来源。生产必须显式配置，否则只放开本地开发端口。
    cors_allow_origins: list[str] = Field(
        default=["http://localhost:5173", "http://127.0.0.1:5173"]
    )

    llm_provider: Provider = Provider.OLLAMA
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_api_key: str = "ollama"
    ollama_chat_model: str = "qwen2.5:7b"
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_api_key: str = ""
    qwen_chat_model: str = "qwen-plus"

    embedding_provider: Provider = Provider.QWEN
    ollama_embedding_model: str = "bge-m3"
    ollama_embedding_dim: int = 1024
    qwen_embedding_model: str = "text-embedding-v3"
    qwen_embedding_dim: int = 1024

    rerank_model: str = "BAAI/bge-reranker-base"
    rerank_use_fp16: bool = True
    rerank_top_n: int = 5
    # auto / cuda / cpu。6G 显存跑 base + fp16 够用；显存吃紧时设 cpu
    rerank_device: str = "auto"
    # 启动后在后台加载 Rerank 模型。关掉的话这十几秒开销会落在第一个用户请求上。
    warmup_reranker: bool = True

    qdrant_path: str = "./data/qdrant_storage"
    qdrant_collection_prefix: str = "kb"
    database_url: str = "sqlite:///./data/app.db"
    # SQLite 写锁等待窗口。Agent 一轮对话内有多次写入（会话/消息/审计），
    # 并发请求下写-写会短暂互斥，给足等待时间比直接抛错合理。
    sqlite_busy_timeout_ms: int = Field(default=10_000, ge=0, le=120_000)
    # WAL 自动 checkpoint 的页数阈值。0 会关掉自动回收，WAL 文件将无限增长。
    sqlite_wal_autocheckpoint_pages: int = Field(default=1_000, ge=100, le=100_000)

    chunk_strategy: ChunkStrategyName = ChunkStrategyName.MARKDOWN
    chunk_size: int = Field(default=500, ge=100, le=4000)
    chunk_overlap: int = Field(default=80, ge=0)
    retrieve_top_k: int = Field(default=10, ge=1, le=100)
    # rerank 归一化分数低于此阈值视为不相关，让"检索为空"分支可达
    min_rerank_score: float = Field(default=0.15, ge=0.0, le=1.0)
    enable_hybrid_retrieve: bool = True
    enable_query_rewrite: bool = True
    rrf_k: int = Field(default=60, ge=1)

    agent_max_steps: int = Field(default=6, ge=1, le=20)
    # 单次 LLM 调用超时。一轮对话串联多次调用，本地 7B 慢机器上要放宽；
    # 需与前端 AGENT_TIMEOUT_MS 一起看：单次 × 步数不应远超前端上限。
    llm_timeout_seconds: float = Field(default=90.0, gt=0, le=600)
    tool_cache_ttl_seconds: int = Field(default=300, ge=0)

    context_window_turns: int = Field(default=6, ge=1)
    enable_context_summary: bool = True

    max_input_length: int = Field(default=2000, ge=1)
    startup_probe_external: bool = False

    @model_validator(mode="after")
    def _validate(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.llm_provider is Provider.QWEN and not self.qwen_api_key:
            raise ValueError("QWEN_API_KEY is required when LLM_PROVIDER=qwen")
        if self.embedding_provider is Provider.QWEN and not self.qwen_api_key:
            raise ValueError("QWEN_API_KEY is required when EMBEDDING_PROVIDER=qwen")
        if self.environment is Environment.PROD:
            self._validate_production()
        return self

    def _validate_production(self) -> None:
        """生产环境的硬性要求：宁可起不来，也不要带着弱密钥或通配 CORS 上线。"""
        if self.api_key == DEFAULT_DEV_API_KEY:
            raise ValueError(
                "API_KEY still uses the development default; "
                "set a strong key before running with ENVIRONMENT=prod"
            )
        if len(self.api_key) < _MIN_PROD_API_KEY_LENGTH:
            raise ValueError(
                f"API_KEY must be at least {_MIN_PROD_API_KEY_LENGTH} characters "
                "when ENVIRONMENT=prod"
            )
        if "*" in self.cors_allow_origins:
            raise ValueError("CORS_ALLOW_ORIGINS must not contain '*' when ENVIRONMENT=prod")
        localhost = [
            o for o in self.cors_allow_origins if "localhost" in o or "127.0.0.1" in o
        ]
        if localhost:
            raise ValueError(
                f"CORS_ALLOW_ORIGINS still contains local origins {localhost}; "
                "configure the real frontend origin for production"
            )

    @property
    def embedding_model(self) -> str:
        if self.embedding_provider is Provider.QWEN:
            return self.qwen_embedding_model
        return self.ollama_embedding_model

    @property
    def embedding_dim(self) -> int:
        if self.embedding_provider is Provider.QWEN:
            return self.qwen_embedding_dim
        return self.ollama_embedding_dim

    @property
    def collection_name(self) -> str:
        """Embedding 模型与维度绑定集合名，避免切换模型时维度冲突。"""
        safe_model = self.embedding_model.replace("/", "_").replace(":", "_")
        return f"{self.qdrant_collection_prefix}_{safe_model}_{self.embedding_dim}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
