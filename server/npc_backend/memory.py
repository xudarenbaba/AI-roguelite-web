from __future__ import annotations

import hashlib
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from server.npc_backend.config import load_config

_EMBED_MODEL: SentenceTransformer | None = None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _get_embed_model() -> SentenceTransformer:
    global _EMBED_MODEL
    if _EMBED_MODEL is not None:
        return _EMBED_MODEL

    cfg = load_config()
    emb_cfg = cfg.get("embeddings", {})
    model_name = emb_cfg.get("model", "BAAI/bge-small-zh-v1.5")
    cache_dir = Path(emb_cfg.get("cache_dir", "models"))
    if not cache_dir.is_absolute():
        cache_dir = _project_root() / cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_files_only = bool(emb_cfg.get("local_files_only", False))
    _EMBED_MODEL = SentenceTransformer(
        model_name,
        cache_folder=str(cache_dir),
        local_files_only=local_files_only,
    )
    return _EMBED_MODEL


def _embed_texts(texts: list[str]) -> list[list[float]]:
    model = _get_embed_model()
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()


class MemoryStore:
    """单集合记忆：world + persona + dialogue(daily/important)。"""

    def __init__(self) -> None:
        cfg = load_config()
        vs_cfg = cfg.get("vectorstore", {})
        persist_dir = Path(vs_cfg.get("persist_dir", "data/chroma"))
        if not persist_dir.is_absolute():
            persist_dir = _project_root() / persist_dir
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._collection_name = vs_cfg.get("collection_name", "npc_memory")
        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        memory_cfg = cfg.get("memory", {})
        self._k_world = int(memory_cfg.get("k_world", 3))
        self._k_persona = int(memory_cfg.get("k_persona", 3))
        self._k_daily = int(memory_cfg.get("k_dialogue_daily", 4))
        self._k_important = int(memory_cfg.get("k_dialogue_important", 6))

    def _collection(self):
        return self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"description": "single npc memory"},
        )

    def _query_with_embedding(
        self, embedding: list[float], where: dict[str, Any], limit: int
    ) -> list[str]:
        coll = self._collection()
        try:
            result = coll.query(
                query_embeddings=[embedding],
                n_results=limit,
                where=where,
                include=["documents"],
            )
            documents = result.get("documents", [[]])[0] if result else []
            return [doc for doc in documents if doc]
        except Exception:
            return []

    def upsert_seed_memory(
        self,
        *,
        memory_type: str,
        npc_id: str,
        texts: list[str],
        extra_metadata: dict[str, Any] | None = None,
        replace_existing: bool = False,
    ) -> None:
        if not texts:
            return
        coll = self._collection()
        normalized = [text.strip() for text in texts if text.strip()]
        if not normalized:
            return
        if replace_existing:
            self.delete_seed_memory(memory_type=memory_type, npc_id=npc_id)
        ids = [
            f"{memory_type}:{npc_id}:{hashlib.sha1(text.encode('utf-8')).hexdigest()}"
            for text in normalized
        ]
        metadatas = [
            {
                "memory_type": memory_type,
                "npc_id": npc_id,
                "source": "seed",
                "created_at": datetime.now(timezone.utc).isoformat(),
                **(extra_metadata or {}),
            }
            for _ in normalized
        ]
        coll.upsert(
            ids=ids,
            embeddings=_embed_texts(normalized),
            documents=normalized,
            metadatas=metadatas,
        )

    def delete_seed_memory(self, *, memory_type: str, npc_id: str) -> int:
        coll = self._collection()
        try:
            existing = coll.get(
                where={"memory_type": memory_type, "npc_id": npc_id, "source": "seed"}
            )
            ids = existing.get("ids", []) if existing else []
            if ids:
                coll.delete(ids=ids)
            return len(ids)
        except Exception:
            return 0

    def add_world_seed(self, texts: list[str]) -> None:
        self.upsert_seed_memory(
            memory_type="world",
            npc_id="global",
            texts=texts,
            extra_metadata={"scope": "global"},
            replace_existing=True,
        )

    def ensure_persona_seeded(self, npc_id: str, persona_lines: list[str]) -> None:
        self.upsert_seed_memory(
            memory_type="persona",
            npc_id=npc_id,
            texts=persona_lines,
            extra_metadata={"scope": "npc"},
            replace_existing=False,
        )

    def add_dialogue_memory(
        self,
        player_id: str,
        npc_id: str,
        dialogue_tier: str,
        text: str,
        scene_info: dict[str, Any] | None = None,
    ) -> None:
        coll = self._collection()
        coll.add(
            ids=[str(uuid.uuid4())],
            embeddings=_embed_texts([text]),
            documents=[text],
            metadatas=[
                {
                    "memory_type": "dialogue",
                    "dialogue_tier": dialogue_tier,
                    "player_id": player_id,
                    "npc_id": npc_id,
                    "source": "runtime",
                    "scene": str(scene_info or {}),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            ],
        )

    def search_context(
        self, query: str, player_id: str, npc_id: str
    ) -> dict[str, list[str]]:
        """query 只 embed 一次，并行查询四路记忆，返回结构化上下文。"""
        embedding = _embed_texts([query])[0]

        tasks = {
            "world_chunks": (
                {"memory_type": "world", "npc_id": "global"},
                self._k_world,
            ),
            "persona_chunks": (
                {"memory_type": "persona", "npc_id": npc_id},
                self._k_persona,
            ),
            "dialogue_daily_chunks": (
                {
                    "memory_type": "dialogue",
                    "dialogue_tier": "daily",
                    "player_id": player_id,
                    "npc_id": npc_id,
                },
                self._k_daily,
            ),
            "dialogue_important_chunks": (
                {
                    "memory_type": "dialogue",
                    "dialogue_tier": "important",
                    "player_id": player_id,
                    "npc_id": npc_id,
                },
                self._k_important,
            ),
        }

        results: dict[str, list[str]] = {}
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self._query_with_embedding, embedding, where, limit): key
                for key, (where, limit) in tasks.items()
            }
            for future in as_completed(futures):
                key = futures[future]
                try:
                    results[key] = future.result()
                except Exception:
                    results[key] = []

        return results

