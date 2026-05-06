"""Web demo server with built-in single NPC backend."""
from __future__ import annotations

from typing import Any

from flask import Flask, jsonify, render_template, request

from server.npc_backend.graph import NpcGraphEngine
from server.npc_backend.llm import classify_intent
from server.npc_backend.schemas import ChatRequest, CommandResponse

def _default_action(message: str) -> dict[str, Any]:
    return {
        "action_type": "dialogue",
        "dialogue": message,
        "emotion": "neutral",
    }


def create_app() -> Flask:
    app = Flask(__name__)
    engine = NpcGraphEngine()

    @app.get("/")
    def index() -> str:
        return render_template("index.html")

    @app.get("/health")
    def health() -> tuple[Any, int]:
        return jsonify({"status": "ok"}), 200

    @app.post("/api/chat")
    def api_chat() -> tuple[Any, int]:
        body = request.get_json(force=True, silent=True) or {}
        try:
            req = ChatRequest(
                player_id=str(body.get("player_id", "")).strip(),
                npc_id=str(body.get("npc_id", "")).strip(),
                message=str(body.get("message", "")).strip(),
                scene_info=body.get("scene_info") or {},
                npc_name=str(body.get("npc_name", "")).strip() or None,
            )
        except Exception as e:  # noqa: BLE001
            return jsonify({"error": f"invalid_request: {e}"}), 400

        if not req.player_id or not req.npc_id or not req.message:
            return jsonify({"error": "player_id, npc_id and message are required"}), 400

        try:
            action = engine.invoke(
                {
                    "player_id": req.player_id,
                    "npc_id": req.npc_id,
                    "npc_name": req.npc_name,
                    "message": req.message,
                    "scene_info": req.scene_info,
                }
            )
            return jsonify(action.model_dump(exclude_none=True)), 200
        except Exception as e:  # noqa: BLE001
            fallback = _default_action("本地 NPC 服务暂时繁忙，我会继续跟随你行动。")
            return jsonify({"error": "npc_backend_unavailable", "detail": str(e), "fallback": fallback}), 200

    @app.post("/api/command")
    def api_command() -> tuple[Any, int]:
        """
        轻量指令分类接口：判断玩家输入是"对话"还是"战术指令"。
        指令时返回 stance + NPC 确认短句；对话时返回 type=dialogue 由前端转发 /api/chat。
        """
        body = request.get_json(force=True, silent=True) or {}
        message = str(body.get("message", "")).strip()
        npc_name = str(body.get("npc_name", "")).strip() or None
        scene_info = body.get("scene_info") or {}

        if not message:
            return jsonify({"error": "message is required"}), 400

        try:
            result = classify_intent(
                message=message,
                scene_info=scene_info,
                npc_name=npc_name or "烬",
            )
            resp = CommandResponse(
                type=result["type"],
                stance=result.get("stance"),
                reply=result.get("reply"),
            )
            return jsonify(resp.model_dump(exclude_none=True)), 200
        except Exception as e:  # noqa: BLE001
            return jsonify({"type": "dialogue", "error": str(e)}), 200

    return app


app = create_app()
