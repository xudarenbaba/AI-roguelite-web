from __future__ import annotations

from typing import Any

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

