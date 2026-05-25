from __future__ import annotations

from typing import Any


def _join_lines(lines: list[str]) -> str:
    if not lines:
        return "无"
    return "\n".join(f"- {line}" for line in lines)


def build_messages(
    *,
    npc_name: str,
    player_message: str,
    scene_info: dict[str, Any],
    world_chunks: list[str],
    persona_chunks: list[str],
    dialogue_daily_chunks: list[str],
    dialogue_important_chunks: list[str],
    short_term_history: list[dict[str, str]],
) -> list[dict[str, str]]:
    short_term_lines = [
        f"{item.get('role', 'unknown')}: {item.get('content', '')}"
        for item in short_term_history
        if item.get("content")
    ]
    system_prompt = (
        f"你是 NPC“{npc_name}”，现在进行单 NPC 对话演示。\n"
        "要求：\n"
        "1) 回复使用中文，简洁自然。\n"
        "2) 战斗场景优先给出可执行建议，通常控制在 1-3 句。\n"
        "3) 非战斗场景可适度话痨，允许 3-6 句碎嘴吐槽。\n"
        "4) 严格基于记忆，不要编造长期事实。\n"
        "5) 注意保持与最近对话的一致性。\n"
        "6) 可以毒舌和贱嗖嗖，但不能恶意辱骂或人身攻击。\n"
        "7) 不输出工具调用、不输出 JSON。\n"
        "8) 在回复正文末尾另起一行，输出一个情绪标签，格式严格为 <emotion>单词</emotion>，"
        "从以下词中选一个最符合当前语气的：neutral focused annoyed worried happy tense sarcastic。"
        "不要解释标签，不要省略。"
    )
    user_prompt = (
        f"[场景]\n{scene_info}\n\n"
        f"[世界观设定]\n{_join_lines(world_chunks)}\n\n"
        f"[角色设定]\n{_join_lines(persona_chunks)}\n\n"
        f"[对话记忆-重要]\n{_join_lines(dialogue_important_chunks)}\n\n"
        f"[对话记忆-日常]\n{_join_lines(dialogue_daily_chunks)}\n\n"
        f"[短期记忆-最近轮次]\n{_join_lines(short_term_lines)}\n\n"
        f"[玩家输入]\n{player_message}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_memory_classify_messages(
    *,
    player_message: str,
    npc_reply: str,
    scene_info: dict[str, Any],
) -> list[dict[str, str]]:
    system_prompt = (
        "你是对话记忆分级器，只输出 JSON："
        '{"dialogue_tier":"daily|important","processed_text":"..."}。\n'
        "规则：important 保留原文；daily 压缩成 1-2 句摘要。"
    )
    user_prompt = (
        f"scene_info={scene_info}\n"
        f"player_message={player_message}\n"
        f"npc_reply={npc_reply}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

