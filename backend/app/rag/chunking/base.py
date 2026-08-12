from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(slots=True)
class Chunk:
    text: str
    index: int
    heading_path: list[str] = field(default_factory=list)
    chunk_type: str | None = None
    is_procedural: bool = False

    @property
    def contextual_text(self) -> str:
        """检索时把标题链拼进正文，缓解切片脱离上下文导致的语义漂移。"""
        if not self.heading_path:
            return self.text
        return " > ".join(self.heading_path) + "\n" + self.text


class ChunkStrategy(ABC):
    name: str

    @abstractmethod
    def split(self, text: str) -> list[Chunk]:
        ...
