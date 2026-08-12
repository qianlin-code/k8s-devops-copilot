from app.errors import AppError
from app.llm.client import LLMClient
from app.rag.reranker import RerankedChunk

_SYSTEM = """你是 Kubernetes 运维支持 Copilot。基于给定的知识片段和工具执行结果回答用户问题。

要求:
- 只使用给定证据中的事实，不要补充未出现的信息。
- 引用知识片段时用 [1] [2] 这样的编号标注，编号对应下面片段的序号。
- 工具结果中的具体数据（Pod 状态、Deployment 副本数、告警工单号）要逐字引用，不要改写
  数值或状态；工具结果里没有出现的字段、命令、资源名，不要补充或猜测。
- 给出的操作命令（如 kubectl 命令）必须直接来自知识片段或工具结果，不要凭经验现编；
  如果证据里没有现成的命令，就用文字描述该做什么，不要杜撰命令。
- 给出可执行的步骤，而不是泛泛的建议。
- 若证据只能部分回答，明确说明哪部分无法确认，不要为了让回答显得完整而编造缺失部分。
- 用中文回答，简洁直接，不要重复用户的问题。"""

_INSUFFICIENT_TEMPLATE = (
    "抱歉，根据现有的知识库内容和系统数据，我无法准确回答这个问题。\n\n"
    "{detail}\n\n"
    "建议：{suggestion}"
)


class Answerer:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def answer(
        self,
        question: str,
        chunks: list[RerankedChunk],
        tool_results: str,
        *,
        context_messages: list[dict[str, str]] | None = None,
        caveat: str | None = None,
    ) -> str:
        knowledge = format_knowledge(chunks)
        messages = [{"role": "system", "content": _SYSTEM}]
        if context_messages:
            messages.extend(context_messages)
        messages.append(
            {
                "role": "user",
                "content": (
                    f"[知识片段]\n{knowledge}\n\n"
                    f"[工具执行结果]\n{tool_results or '(无)'}\n\n"
                    f"[用户问题]\n{question}"
                ),
            }
        )
        try:
            text = self._llm.chat(messages, temperature=0.2)
        except AppError as exc:
            return (
                "抱歉，生成回答时后端模型服务不可用"
                f"（{exc.code.value}），请稍后重试或联系管理员。"
            )
        body = text.strip() or "抱歉，我没能生成有效回答，请换个说法再试一次。"
        return f"{body}\n\n提示：{caveat}" if caveat else body

    def insufficient_answer(
        self, missing: list[str], suggestion: str | None
    ) -> str:
        detail = (
            "缺少以下关键信息：\n" + "\n".join(f"- {m}" for m in missing)
            if missing
            else "现有资料与您的问题相关性不足。"
        )
        return _INSUFFICIENT_TEMPLATE.format(
            detail=detail,
            suggestion=suggestion or "补充更多细节后重新提问，或提交工单转人工处理。",
        )


def format_knowledge(chunks: list[RerankedChunk]) -> str:
    if not chunks:
        return "(无相关片段)"
    return "\n\n".join(
        f"[{i}] 来源: {c.chunk.citation_label()}\n{c.chunk.text}"
        for i, c in enumerate(chunks, start=1)
    )
