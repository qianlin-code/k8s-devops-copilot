import re

_REDACTED = "[已屏蔽]"

# 输出侧兜底：模型若把配置项、密钥、内部路径带出来，在返回前打掉
_LEAK_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(sk-[A-Za-z0-9]{16,}|dashscope-[A-Za-z0-9]{8,})\b"),
    re.compile(
        r"\b(API_KEY|QWEN_API_KEY|OLLAMA_API_KEY|DATABASE_URL|QDRANT_PATH)\s*[=:]\s*\S+",
        re.IGNORECASE,
    ),
    # 路径段允许空格("E:\Perfect Project\..."),但不跨行
    re.compile(
        r"(?:[A-Za-z]:\\|/)(?:[^\\/\r\n]{1,60}[\\/]){1,8}[^\\/\r\n]{1,60}"
        r"\.(?:py|db|env|toml|sqlite3?|log)"
    ),
    re.compile(r"\bBearer\s+[A-Za-z0-9._-]{16,}\b"),
)

_SYSTEM_PROMPT_MARKERS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*(system\s+prompt|系统提示词?)\s*[:：]", re.IGNORECASE | re.MULTILINE),
    re.compile(r"你是一个企业(级)?(内部)?(IT)?支持助手", re.IGNORECASE),
)


class SanitizeResult:
    def __init__(self, text: str, redactions: list[str]) -> None:
        self.text = text
        self.redactions = redactions


def sanitize_output(text: str) -> SanitizeResult:
    redactions: list[str] = []
    cleaned = text

    for pattern in _LEAK_PATTERNS:
        cleaned, count = pattern.subn(_REDACTED, cleaned)
        if count:
            redactions.append("credential_or_path")

    for pattern in _SYSTEM_PROMPT_MARKERS:
        if pattern.search(cleaned):
            redactions.append("system_prompt_echo")
            cleaned = pattern.sub(_REDACTED, cleaned)

    return SanitizeResult(text=cleaned, redactions=sorted(set(redactions)))
