import hashlib
import json
import threading
import time
from typing import Any


class ToolResultCache:
    """只读工具结果缓存。写操作绝对不进这里，避免重复执行与数据不一致。"""

    def __init__(self, ttl_seconds: int) -> None:
        self.ttl = ttl_seconds
        self._data: dict[str, tuple[float, dict[str, Any]]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def build_key(tool_name: str, args: dict[str, Any]) -> str:
        payload = json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
        digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
        return f"{tool_name}:{digest}"

    def get(self, key: str) -> dict[str, Any] | None:
        if self.ttl <= 0:
            return None
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            stored_at, value = entry
            if time.monotonic() - stored_at > self.ttl:
                del self._data[key]
                return None
            return value

    def set(self, key: str, value: dict[str, Any]) -> None:
        if self.ttl <= 0:
            return
        with self._lock:
            self._data[key] = (time.monotonic(), value)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._data)
