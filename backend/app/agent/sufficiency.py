from typing import Optional

from pydantic import BaseModel, Field

from app.errors import AppError
from app.llm.client import LLMClient

_SYSTEM = """你是回答前的信息充分性审核员，职责是防止模型在证据不足时编造答案。

判定 sufficient=true 的情形（满足其一即可）:
- 知识片段已经说明了问题的成因与处理步骤
- 工具结果已经返回了问题所需的具体事实（Pod 状态、Deployment 副本数、告警工单号等）
- 二者结合足以给出一个具体、可执行的回答

判定 sufficient=false 的情形:
- 证据与用户问题明显不相关
- 问题需要某个实时数据，但相关工具尚未调用或调用失败
- 只能给出"请联系集群管理员"这类没有信息量的空泛回答

注意: 工具已成功返回数据时不要再判不足；不要因为"还可以补充更多细节"而否决。
判不足时必须在 missing_information 写明缺什么，
并在 suggested_next_step 给出建议：调用哪个工具，或向用户追问什么。"""


class SufficiencyVerdict(BaseModel):
    sufficient: bool = Field(description="现有信息是否足以准确回答")
    reasoning: str = Field(description="判断理由")
    missing_information: list[str] = Field(
        default_factory=list, description="仍缺失的关键信息"
    )
    suggested_next_step: Optional[str] = Field(
        default=None, description="建议的补救动作：调用某工具或向用户追问"
    )


class SufficiencyChecker:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def check(
        self,
        question: str,
        knowledge: str,
        tool_results: str,
    ) -> SufficiencyVerdict:
        payload = (
            f"[用户问题]\n{question}\n\n"
            f"[知识片段]\n{knowledge or '(无)'}\n\n"
            f"[工具执行结果]\n{tool_results or '(无)'}"
        )
        try:
            return self._llm.structured(
                [
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": payload},
                ],
                SufficiencyVerdict,
            )
        except AppError as exc:
            # 校验器失败时保守放行，但显式标注未经校验，避免静默降级
            return SufficiencyVerdict(
                sufficient=True,
                reasoning=f"充分性校验不可用({exc.code.value})，未经校验直接回答。",
                missing_information=[],
                suggested_next_step=None,
            )
