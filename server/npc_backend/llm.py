from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from server.npc_backend.config import load_config
from server.npc_backend.prompts import build_memory_classify_messages

# 支持的姿态集合（前端 game.js 中的 state.ally.stance 枚举值）
VALID_STANCES = {"guard", "assault", "skirmish"}

# 姿态 → NPC 确认短句（符合话痨嘴臭人设）
_STANCE_REPLIES: dict[str, str] = {
    "guard": "行，收拢了，你别乱跑，我贴着你。",
    "assault": "好嘞，我去前面撕，你别拖后腿。",
    "skirmish": "懂了，我去扫小怪，别让垃圾堆满地图。",
}


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


def classify_intent(
    *,
    message: str,
    scene_info: dict[str, Any],
    npc_name: str = "烬",
) -> dict[str, Any]:
    """
    判断玩家输入是"普通对话"还是"战术指令"。

    返回：
      - type=dialogue：走完整 LangGraph 对话流程
      - type=command：直接切换姿态，包含 stance + 确认短句 reply

    stance 枚举：guard（守护）/ assault（进攻）/ skirmish（游击）
    """
    system_prompt = (
        "你是战术指令解析器，只输出 JSON，格式二选一：\n"
        '{"type":"dialogue"}\n'
        '{"type":"command","stance":"guard|assault|skirmish","reply":"NPC一句确认话"}\n\n'
        "判断规则：\n"
        "1. command：玩家明确要求改变 NPC 行动模式，"
        "如[贴着我/守护/别乱跑]->guard，[上去打/突击/压制]->assault，[先清小怪/游击/减轻压力]->skirmish。\n"
        "2. dialogue：情绪交流、世界观追问、模糊意图、战斗评论 → 一律 dialogue。\n"
        "3. reply 要符合嘴臭话痨风格，简短，1句话。\n"
        "4. 不确定时默认 dialogue，宁可多对话不乱改姿态。"
    )
    user_prompt = (
        f"npc_name={npc_name}\n"
        f"scene_info={json.dumps(scene_info, ensure_ascii=False)}\n"
        f"player_message={message}"
    )
    cfg = load_config().get("llm", {})
    try:
        raw = chat_completion([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])
        # 剥掉 markdown 代码块（部分模型会包裹）
        stripped = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data: dict[str, Any] = json.loads(stripped)
        intent_type = str(data.get("type", "")).strip()
        if intent_type == "command":
            stance = str(data.get("stance", "")).strip()
            if stance not in VALID_STANCES:
                raise ValueError(f"invalid stance: {stance}")
            reply = str(data.get("reply", _STANCE_REPLIES.get(stance, "收到。"))).strip()
            return {"type": "command", "stance": stance, "reply": reply}
        return {"type": "dialogue"}
    except Exception:
        return {"type": "dialogue"}


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

