from __future__ import annotations

import json
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from server.npc_backend.config import load_config
from server.npc_backend.llm import chat_completion_stream, classify_dialogue_memory
from server.npc_backend.memory import MemoryStore
from server.npc_backend.prompts import build_messages
from server.npc_backend.schemas import ChatAction
from server.npc_backend.short_term import ShortTermMemory


class NpcConversationEngine:
    def __init__(self) -> None:
        self._cfg = load_config()
        self._memory = MemoryStore()
        self._short_term = ShortTermMemory()

    def stream_chat(self, payload: dict[str, Any]) -> Iterator[str]:
        """
        主流式对话入口，逐行 yield NDJSON 字符串。

        事件类型：
          meta  - 开始，携带 npc_id
          delta - 模型 token 片段
          done  - 完整结束，携带最终 ChatAction
          error - 出错，携带 fallback
        """
        player_id: str = payload.get("player_id", "")
        npc_id: str = payload.get("npc_id", "")
        npc_name: str = payload.get("npc_name") or npc_id
        message: str = payload.get("message", "")
        scene_info: dict[str, Any] = payload.get("scene_info") or {}

        yield _event("meta", {"npc_id": npc_id})

        try:
            query = f"scene={scene_info}\nmessage={message}"

            # 并行加载短期记忆和长期记忆上下文
            short_term_history: list[dict[str, Any]] = []
            long_term_ctx: dict[str, list[str]] = {}

            with ThreadPoolExecutor(max_workers=2) as executor:
                fut_short = executor.submit(
                    self._short_term.get_recent, player_id, npc_id
                )
                fut_long = executor.submit(
                    self._memory.search_context, query, player_id, npc_id
                )
                for fut in as_completed([fut_short, fut_long]):
                    if fut is fut_short:
                        short_term_history = fut.result()
                    else:
                        long_term_ctx = fut.result()

            messages = build_messages(
                npc_name=npc_name,
                player_message=message,
                scene_info=scene_info,
                world_chunks=long_term_ctx.get("world_chunks", []),
                persona_chunks=long_term_ctx.get("persona_chunks", []),
                dialogue_daily_chunks=long_term_ctx.get("dialogue_daily_chunks", []),
                dialogue_important_chunks=long_term_ctx.get(
                    "dialogue_important_chunks", []
                ),
                short_term_history=short_term_history,
            )

            full_reply_parts: list[str] = []
            for delta in chat_completion_stream(messages):
                full_reply_parts.append(delta)
                yield _event("delta", {"text": delta})

            full_reply = "".join(full_reply_parts).strip() or "收到，我会继续和你协同。"

            # 写短期记忆
            self._short_term.add_turn(
                player_id=player_id, npc_id=npc_id, role="user", content=message
            )
            self._short_term.add_turn(
                player_id=player_id,
                npc_id=npc_id,
                role="assistant",
                content=full_reply,
            )

            # 分级并写长期记忆
            min_chars = int(self._cfg.get("memory", {}).get("min_store_chars", 6))
            if len(full_reply) >= min_chars:
                tier, text = classify_dialogue_memory(
                    player_message=message,
                    npc_reply=full_reply,
                    scene_info=scene_info,
                )
                self._memory.add_dialogue_memory(
                    player_id=player_id,
                    npc_id=npc_id,
                    dialogue_tier=tier,
                    text=text,
                    scene_info=scene_info,
                )

            action = ChatAction(
                action_type="dialogue", dialogue=full_reply, emotion="focused"
            )
            yield _event("done", {"action": action.model_dump(exclude_none=True)})

        except Exception as exc:  # noqa: BLE001
            fallback = ChatAction(
                action_type="dialogue",
                dialogue="本地 NPC 服务暂时繁忙，我会继续跟随你行动。",
                emotion="neutral",
            )
            yield _event(
                "error",
                {
                    "message": str(exc),
                    "fallback": fallback.model_dump(exclude_none=True),
                },
            )


def _event(event_type: str, data: dict[str, Any]) -> str:
    return json.dumps({"type": event_type, **data}, ensure_ascii=False) + "\n"
