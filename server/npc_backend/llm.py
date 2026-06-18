from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from openai import OpenAI

from server.npc_backend.config import load_config
from server.npc_backend.prompts import build_memory_classify_messages

# 支持的姿态集合（前端 game.js 中的 state.ally.stance 枚举值）
VALID_STANCES = {"guard", "assault"}

# 姿态 → NPC 确认短句（符合话痨嘴臭人设）
_STANCE_REPLIES: dict[str, str] = {
    "guard":  "行，收拢了，你别乱跑，我贴着你。",
    "assault": "好嘞，我去前面撕，你别拖后腿。",
}

_INTENT_SYSTEM_PROMPT = (
    "你是战术指令解析器，只输出 JSON，格式二选一：\n"
    '{"type":"dialogue"}\n'
    '{"type":"command","stance":"guard|assault","reply":"NPC一句确认话"}\n\n'
    "判断规则：\n"
    "1. command：玩家明确要求改变 NPC 行动模式，"
    "如[贴着我/守护/别乱跑/回来]->guard，[上去打/突击/压制/冲]->assault。\n"
    "2. dialogue：情绪交流、世界观追问、模糊意图、战斗评论 → 一律 dialogue。\n"
    "3. reply 要符合嘴臭话痨风格，简短，1句话。\n"
    "4. 不确定时默认 dialogue，宁可多对话不乱改姿态。"
)


def _client() -> OpenAI:
    cfg = load_config().get("llm", {})
    return OpenAI(
        api_key=cfg.get("api_key") or "dummy",
        base_url=cfg.get("base_url"),
    )


def chat_completion(messages: list[dict[str, str]]) -> str:
    cfg = load_config().get("llm", {})
    resp = _client().chat.completions.create(
        model=cfg.get("model", "deepseek-chat"),
        messages=messages,
        temperature=float(cfg.get("temperature", 0.3)),
        timeout=int(cfg.get("timeout_s", 60)),
    )
    choice = resp.choices[0] if resp.choices else None
    if not choice or not choice.message:
        return ""
    return (choice.message.content or "").strip()


def chat_completion_stream(messages: list[dict[str, str]]) -> Iterator[str]:
    """逐 token yield 文本 delta，不处理业务状态和记忆。"""
    cfg = load_config().get("llm", {})
    stream = _client().chat.completions.create(
        model=cfg.get("model", "deepseek-chat"),
        messages=messages,
        temperature=float(cfg.get("temperature", 0.3)),
        timeout=int(cfg.get("timeout_s", 60)),
        stream=True,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def classify_intent(
    *,
    message: str,
    scene_info: dict[str, Any],
    npc_name: str,
) -> dict[str, Any]:
    """
    判断玩家输入是"普通对话"还是"战术指令"。

    返回：
      - {"type": "dialogue"}
      - {"type": "command", "stance": "guard|assault|skirmish", "reply": "..."}
    """
    user_prompt = (
        f"npc_name={npc_name}\n"
        f"scene_info={json.dumps(scene_info, ensure_ascii=False)}\n"
        f"player_message={message}"
    )
    try:
        raw = chat_completion([
            {"role": "system", "content": _INTENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ])
        stripped = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data: dict[str, Any] = json.loads(stripped)
        if str(data.get("type", "")).strip() == "command":
            stance = str(data.get("stance", "")).strip()
            if stance not in VALID_STANCES:
                raise ValueError(f"invalid stance: {stance}")
            reply = str(data.get("reply", _STANCE_REPLIES.get(stance, "收到。"))).strip()
            return {"type": "command", "stance": stance, "reply": reply}
    except Exception:
        pass
    return {"type": "dialogue"}


def generate_no_hp_reply(*, npc_name: str) -> str:
    """
    NPC 灵核失稳（hp=0）时被要求突击，生成一句符合人设的拒绝回复。
    使用轻量单次调用，不走记忆检索。
    失败时返回硬编码兜底文本。
    """
    system_prompt = (
        f"你是 NPC「{npc_name}」，嘴臭话痨，但此刻灵核失稳、无法战斗。"
        "玩家要求你去突击，你需要用一句话拒绝，语气可以无奈、嘴硬或自嘲，"
        "符合话痨人设，不超过20字，不要解释原因只说无法执行。只输出这一句话，不加引号。"
    )
    try:
        reply = chat_completion([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "玩家让你去突击。"},
        ]).strip()
        return reply or "灵核失稳了，冲不动，别催。"
    except Exception:  # noqa: BLE001
        return "灵核失稳了，冲不动，别催。"


def classify_dialogue_memory(
    *,
    player_message: str,
    npc_reply: str,
    scene_info: dict[str, Any],
) -> tuple[str, str]:
    raw_text = f"玩家说：{player_message}；NPC 回复：{npc_reply}"
    messages = build_memory_classify_messages(
        player_message=player_message,
        npc_reply=npc_reply,
        scene_info=scene_info,
    )
    try:
        data: dict[str, Any] = json.loads(chat_completion(messages))
        tier = str(data.get("dialogue_tier", "")).strip()
        processed = str(data.get("processed_text", "")).strip()
        if tier not in {"daily", "important"} or not processed:
            raise ValueError("invalid memory classify result")
        if tier == "important":
            return "important", raw_text
        return "daily", processed
    except Exception:
        return "important", raw_text

