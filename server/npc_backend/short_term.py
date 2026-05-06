from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any

from server.npc_backend.config import load_config


class ShortTermMemory:
    def __init__(self) -> None:
        turns = int(load_config().get("memory", {}).get("short_term_turns", 10))
        self._max_items = max(2, turns * 2)
        self._store: dict[str, deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=self._max_items)
        )

    @staticmethod
    def _key(player_id: str, npc_id: str) -> str:
        return f"{player_id}:{npc_id}"

    def add_turn(self, player_id: str, npc_id: str, role: str, content: str) -> None:
        if not content.strip():
            return
        self._store[self._key(player_id, npc_id)].append(
            {
                "role": role,
                "content": content.strip(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def get_recent(self, player_id: str, npc_id: str) -> list[dict[str, Any]]:
        return list(self._store[self._key(player_id, npc_id)])

