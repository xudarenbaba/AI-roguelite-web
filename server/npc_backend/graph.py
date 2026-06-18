from __future__ import annotations

import json
import re
import threading
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from server.npc_backend.config import load_config
from server.npc_backend.llm import (
    chat_completion_stream,
    classify_dialogue_memory,
    classify_intent,
    generate_no_hp_reply,
)
from server.npc_backend.memory import MemoryStore
from server.npc_backend.prompts import build_messages
from server.npc_backend.schemas import ChatAction
from server.npc_backend.short_term import ShortTermMemory

_EMOTION_RE = re.compile(r"<emotion>([\w]+)</emotion>\s*$", re.IGNORECASE)
_VALID_EMOTIONS = {
    "neutral", "focused", "annoyed", "worried", "happy", "tense", "sarcastic"
}

# delta 批量缓冲阈值（字符数）：积累到此长度或收到 LLM 流结束后统一 flush
_DELTA_BATCH_CHARS = 8


class NpcConversationEngine:
    def __init__(self) -> None:
        self._cfg = load_config()
        self._memory = MemoryStore()
        self._short_term = ShortTermMemory()

    def stream_chat(self, payload: dict[str, Any]) -> Iterator[str]:
        """
        统一流式入口，逐行 yield NDJSON 字符串。

        事件类型：
          meta    - 立即返回，携带 npc_id
          command - 战术指令，携带 stance + reply，前端切换姿态后结束
          delta   - 对话 token 批次（积累 _DELTA_BATCH_CHARS 字符后 flush）
          done    - 对话完整结束，携带最终 ChatAction
          error   - 出错，携带 fallback

        并行优化：收到请求后立即同时启动：
          - 意图分类 LLM（非流式）
          - 短期记忆 + 长期记忆检索
        若分类结果为 command，直接 yield command 事件，记忆检索结果丢弃。
        若分类结果为 dialogue，记忆检索结果（大概率已完成）直接用于构建 prompt。
        """
        player_id: str = payload.get("player_id", "")
        npc_id: str = payload.get("npc_id", "")
        npc_name: str = payload.get("npc_name") or npc_id
        message: str = payload.get("message", "")
        scene_info: dict[str, Any] = payload.get("scene_info") or {}

        yield _event("meta", {"npc_id": npc_id})

        try:
            query = f"scene={scene_info}\nmessage={message}"

            # 同时并行启动：意图分类 + 短期记忆 + 长期记忆检索（共3个任务）
            with ThreadPoolExecutor(max_workers=3) as executor:
                fut_intent: Future[dict[str, Any]] = executor.submit(
                    classify_intent,
                    message=message,
                    scene_info=scene_info,
                    npc_name=npc_name,
                )
                fut_short: Future[list[dict[str, Any]]] = executor.submit(
                    self._short_term.get_recent, player_id, npc_id
                )
                fut_long: Future[dict[str, list[str]]] = executor.submit(
                    self._memory.search_context, query, player_id, npc_id
                )

                # 意图分类优先等待，它决定后续走哪条路
                intent = fut_intent.result()

            if intent["type"] == "command":
                stance = intent["stance"]
                ally_hp = int(scene_info.get("ally_hp", 100))
                # NPC 无血时拒绝突击指令，改为对话回复
                if stance == "assault" and ally_hp <= 0:
                    refusal = generate_no_hp_reply(npc_name=npc_name)
                    action = ChatAction(
                        action_type="dialogue", dialogue=refusal, emotion="annoyed"
                    )
                    yield _event("done", {"action": action.model_dump(exclude_none=True)})
                    return
                yield _event("command", {
                    "stance": stance,
                    "reply":  intent["reply"],
                })
                return

            # dialogue 分支：记忆结果此时已完成或即将完成
            short_term_history: list[dict[str, Any]] = fut_short.result()
            long_term_ctx: dict[str, list[str]] = fut_long.result()

            messages = build_messages(
                npc_name=npc_name,
                player_message=message,
                scene_info=scene_info,
                world_chunks=long_term_ctx.get("world_chunks", []),
                persona_chunks=long_term_ctx.get("persona_chunks", []),
                dialogue_daily_chunks=long_term_ctx.get("dialogue_daily_chunks", []),
                dialogue_important_chunks=long_term_ctx.get("dialogue_important_chunks", []),
                short_term_history=short_term_history,
            )

            full_reply_parts: list[str] = []
            batch: list[str] = []

            for delta in chat_completion_stream(messages):
                full_reply_parts.append(delta)
                batch.append(delta)
                if sum(len(s) for s in batch) >= _DELTA_BATCH_CHARS:
                    yield _event("delta", {"text": "".join(batch)})
                    batch = []

            if batch:
                yield _event("delta", {"text": "".join(batch)})

            raw_reply = "".join(full_reply_parts).strip()

            # 从回复末尾提取 <emotion>xxx</emotion>
            emotion = "neutral"
            m = _EMOTION_RE.search(raw_reply)
            if m:
                candidate = m.group(1).lower()
                emotion = candidate if candidate in _VALID_EMOTIONS else "neutral"
                raw_reply = _EMOTION_RE.sub("", raw_reply).strip()

            full_reply = raw_reply or "收到，我会继续和你协同。"

            action = ChatAction(action_type="dialogue", dialogue=full_reply, emotion=emotion)
            yield _event("done", {"action": action.model_dump(exclude_none=True)})

            # 后台异步写记忆，不阻塞流式响应
            min_chars = int(self._cfg.get("memory", {}).get("min_store_chars", 6))

            def _write_memory(
                reply: str = full_reply,
                msg: str = message,
                pid: str = player_id,
                nid: str = npc_id,
                si: dict = scene_info,
                mc: int = min_chars,
            ) -> None:
                self._short_term.add_turn(pid, nid, "user", msg)
                self._short_term.add_turn(pid, nid, "assistant", reply)
                if len(reply) >= mc:
                    try:
                        tier, text = classify_dialogue_memory(
                            player_message=msg,
                            npc_reply=reply,
                            scene_info=si,
                        )
                        self._memory.add_dialogue_memory(
                            player_id=pid,
                            npc_id=nid,
                            dialogue_tier=tier,
                            text=text,
                            scene_info=si,
                        )
                    except Exception:  # noqa: BLE001
                        pass

            threading.Thread(target=_write_memory, daemon=True).start()

        except Exception as exc:  # noqa: BLE001
            fallback = ChatAction(
                action_type="dialogue",
                dialogue="本地 NPC 服务暂时繁忙，我会继续跟随你行动。",
                emotion="neutral",
            )
            yield _event("error", {
                "message": str(exc),
                "fallback": fallback.model_dump(exclude_none=True),
            })


def _event(event_type: str, data: dict[str, Any]) -> str:
    return json.dumps({"type": event_type, **data}, ensure_ascii=False) + "\n"
