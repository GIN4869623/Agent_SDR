from __future__ import annotations

from threading import Lock


class ConversationMemory:
    def __init__(self, max_messages: int = 12) -> None:
        self.max_messages = max_messages
        self._store: dict[str, list[dict[str, str]]] = {}
        self._lock = Lock()

    def get_history(self, session_id: str) -> list[dict[str, str]]:
        with self._lock:
            return list(self._store.get(session_id, []))

    def replace(self, session_id: str, messages: list[dict[str, str]]) -> None:
        trimmed = messages[-self.max_messages :]
        with self._lock:
            self._store[session_id] = trimmed

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._store.pop(session_id, None)
