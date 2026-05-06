from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    player_id: str = Field(min_length=1)
    npc_id: str = Field(min_length=1)
    message: str = Field(min_length=1)
    scene_info: dict[str, Any] = Field(default_factory=dict)
    npc_name: str | None = None


class ChatAction(BaseModel):
    action_type: str = "dialogue"
    dialogue: str
    emotion: str | None = None


class CommandResponse(BaseModel):
    """指令分类接口的响应体。"""
    type: Literal["command", "dialogue"]
    # type=command 时有值
    stance: str | None = None
    reply: str | None = None

