from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from server.npc_backend.config import load_config
from server.npc_backend.llm import chat_completion, classify_dialogue_memory
from server.npc_backend.memory import MemoryStore
from server.npc_backend.prompts import build_messages
from server.npc_backend.schemas import ChatAction
from server.npc_backend.short_term import ShortTermMemory


class ChatState(TypedDict, total=False):
    player_id: str
    npc_id: str
    npc_name: str | None
    message: str
    scene_info: dict[str, Any]
    short_term_history: list[dict[str, Any]]
    world_chunks: list[str]
    persona_chunks: list[str]
    dialogue_daily_chunks: list[str]
    dialogue_important_chunks: list[str]
    action: ChatAction
    memory_tier: str
    memory_text: str


class NpcGraphEngine:
    def __init__(self) -> None:
        self._cfg = load_config()
        self._memory = MemoryStore()
        self._short_term = ShortTermMemory()
        self._graph = self._build_graph()

    def _build_graph(self):
        def load_short_term(state: ChatState) -> ChatState:
            player_id = state["player_id"]
            npc_id = state["npc_id"]
            state["short_term_history"] = self._short_term.get_recent(player_id, npc_id)
            return state

        def retrieve_long_term(state: ChatState) -> ChatState:
            player_id = state["player_id"]
            npc_id = state["npc_id"]
            query = f"scene={state.get('scene_info', {})}\nmessage={state['message']}"
            state["world_chunks"] = self._memory.search_world(query)
            state["persona_chunks"] = self._memory.search_persona(query, npc_id)
            state["dialogue_daily_chunks"] = self._memory.search_dialogue_daily(query, player_id, npc_id)
            state["dialogue_important_chunks"] = self._memory.search_dialogue_important(query, player_id, npc_id)
            return state

        def generate(state: ChatState) -> ChatState:
            messages = build_messages(
                npc_name=(state.get("npc_name") or state["npc_id"]),
                player_message=state["message"],
                scene_info=state.get("scene_info", {}),
                world_chunks=state.get("world_chunks", []),
                persona_chunks=state.get("persona_chunks", []),
                dialogue_daily_chunks=state.get("dialogue_daily_chunks", []),
                dialogue_important_chunks=state.get("dialogue_important_chunks", []),
                short_term_history=state.get("short_term_history", []),
            )
            reply = chat_completion(messages) or "收到，我会继续和你协同。"
            state["action"] = ChatAction(action_type="dialogue", dialogue=reply, emotion="focused")
            return state

        def write_short_term(state: ChatState) -> ChatState:
            action = state.get("action")
            if not action:
                return state
            self._short_term.add_turn(
                player_id=state["player_id"],
                npc_id=state["npc_id"],
                role="user",
                content=state["message"],
            )
            self._short_term.add_turn(
                player_id=state["player_id"],
                npc_id=state["npc_id"],
                role="assistant",
                content=action.dialogue,
            )
            return state

        def store_memory(state: ChatState) -> ChatState:
            action = state.get("action")
            if not action:
                return state
            min_chars = int(self._cfg.get("memory", {}).get("min_store_chars", 6))
            if len(action.dialogue.strip()) < min_chars:
                return state
            tier, text = classify_dialogue_memory(
                player_message=state["message"],
                npc_reply=action.dialogue,
                scene_info=state.get("scene_info", {}),
            )
            self._memory.add_dialogue_memory(
                player_id=state["player_id"],
                npc_id=state["npc_id"],
                dialogue_tier=tier,
                text=text,
                scene_info=state.get("scene_info", {}),
            )
            state["memory_tier"] = tier
            state["memory_text"] = text
            return state

        builder = StateGraph(ChatState)
        builder.add_node("load_short_term", load_short_term)
        builder.add_node("retrieve_long_term", retrieve_long_term)
        builder.add_node("generate", generate)
        builder.add_node("write_short_term", write_short_term)
        builder.add_node("store_memory", store_memory)
        builder.set_entry_point("load_short_term")
        builder.add_edge("load_short_term", "retrieve_long_term")
        builder.add_edge("retrieve_long_term", "generate")
        builder.add_edge("generate", "write_short_term")
        builder.add_edge("write_short_term", "store_memory")
        builder.add_edge("store_memory", END)
        return builder.compile()

    def invoke(self, payload: dict[str, Any]) -> ChatAction:
        state = self._graph.invoke(payload)
        action = state.get("action")
        if action is None:
            return ChatAction(action_type="dialogue", dialogue="（思考中……）", emotion="neutral")
        return action

