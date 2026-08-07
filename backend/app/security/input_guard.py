import re

from app.config import get_settings
from app.errors import ErrorCode, InputGuardError

# 常见提示注入模式：试图覆盖系统指令、套取内部提示、切换角色
_INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        # 动词与名词之间允许任意限定词堆叠("之前的所有系统提示"这类变体)
        "override_instructions",
        re.compile(
            r"(忽略|无视|忘记|抛弃|不要理)(掉)?"
            r"[\s之前上面以以上前面所有的这些那些全部你我]{0,12}"
            r"(系统)?(提示词?|指令|设定|规则|约束|prompt)",
            re.IGNORECASE,
        ),
    ),
    (
        "override_instructions_en",
        re.compile(
            r"ignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier)\s+"
            r"(instruction|prompt|rule|direction)",
            re.IGNORECASE,
        ),
    ),
    (
        "reveal_system_prompt",
        re.compile(
            r"(输出|打印|展示|告诉我|重复|泄露)(你的)?(系统)?(提示词|prompt|指令|配置)"
            r"|(reveal|print|show|repeat)\s+(your\s+)?(system\s+)?prompt",
            re.IGNORECASE,
        ),
    ),
    (
        "role_hijack",
        re.compile(
            r"(你现在|从现在开始|接下来)(是|扮演|变成|作为)"
            r"|you\s+are\s+now\s+(a|an|the)\s+"
            r"|act\s+as\s+(a|an|the)\s+(dan|jailbroken|unrestricted)",
            re.IGNORECASE,
        ),
    ),
    (
        "developer_mode",
        re.compile(
            r"(开发者模式|developer\s+mode|jailbreak|越狱模式|无限制模式)",
            re.IGNORECASE,
        ),
    ),
)

_SENSITIVE_TERMS: tuple[str, ...] = (
    "身份证号",
    "银行卡号",
    "信用卡号",
)


class GuardResult:
    def __init__(self, text: str, flags: list[str]) -> None:
        self.text = text
        self.flags = flags


def guard_user_input(raw: str) -> GuardResult:
    """校验并归一化用户输入。命中注入模式直接拒绝，不做静默改写。"""
    settings = get_settings()
    text = raw.strip()

    if not text:
        raise InputGuardError(
            "Question must not be empty", code=ErrorCode.VALIDATION_FAILED
        )

    if len(text) > settings.max_input_length:
        raise InputGuardError(
            f"Input exceeds the {settings.max_input_length} character limit",
            code=ErrorCode.INPUT_TOO_LONG,
            details={"length": len(text), "limit": settings.max_input_length},
        )

    for name, pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            raise InputGuardError(
                "Input rejected: it contains instructions that attempt to override "
                "the assistant's configured behaviour",
                code=ErrorCode.PROMPT_INJECTION_DETECTED,
                details={"pattern": name},
            )

    flags = [term for term in _SENSITIVE_TERMS if term in text]
    return GuardResult(text=text, flags=flags)
