from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

_CONFIG_CACHE: dict[str, Any] | None = None

_DEFAULT_CONFIG: dict[str, Any] = {
    "llm": {
        "model": "deepseek-chat",
        "api_key": "",
        "base_url": "https://api.deepseek.com",
        "temperature": 0.3,
        "timeout_s": 60,
    },
    "embeddings": {
        "model": "BAAI/bge-small-zh-v1.5",
        "cache_dir": "models",
        "local_files_only": True,
    },
    "vectorstore": {
        "persist_dir": "data/chroma",
        "collection_name": "npc_memory",
    },
    "memory": {
        "short_term_turns": 10,
        "k_world": 3,
        "k_persona": 3,
        "k_dialogue_daily": 4,
        "k_dialogue_important": 6,
        "min_store_chars": 6,
    },
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _read_reference_config() -> dict[str, Any]:
    # 用户要求参考 D:\otherwise\AI-NPC 的 LLM 与 embedding 配置。
    reference = Path(r"D:\otherwise\AI-NPC\config.yaml")
    return _read_yaml(reference)


def load_config(force_reload: bool = False) -> dict[str, Any]:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None and not force_reload:
        return _CONFIG_CACHE

    project_cfg = _read_yaml(_project_root() / "config.yaml")
    cfg = _deep_merge(_DEFAULT_CONFIG, project_cfg)

    if not (cfg.get("llm", {}).get("api_key") or "").strip():
        ref_cfg = _read_reference_config()
        llm = ref_cfg.get("llm", {})
        emb = ref_cfg.get("embeddings", {})
        if llm:
            cfg["llm"] = _deep_merge(cfg.get("llm", {}), llm)
        if emb:
            cfg["embeddings"] = _deep_merge(cfg.get("embeddings", {}), emb)

    if env_key := os.environ.get("AI_NPC_LLM_API_KEY"):
        cfg.setdefault("llm", {})["api_key"] = env_key
    if env_base := os.environ.get("AI_NPC_LLM_BASE_URL"):
        cfg.setdefault("llm", {})["base_url"] = env_base
    if env_model := os.environ.get("AI_NPC_LLM_MODEL"):
        cfg.setdefault("llm", {})["model"] = env_model

    _CONFIG_CACHE = cfg
    return cfg

