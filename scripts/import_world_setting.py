from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from server.npc_backend.memory import MemoryStore


def split_markdown_sections(text: str) -> list[str]:
    chunks = []
    for block in re.split(r"\n(?=##\s+)", text.strip()):
        cleaned = block.strip()
        if cleaned:
            chunks.append(cleaned)
    return chunks


def import_world_setting(md_path: Path) -> int:
    content = md_path.read_text(encoding="utf-8")
    chunks = split_markdown_sections(content)
    store = MemoryStore()
    store.add_world_seed(chunks)
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Import world setting markdown into ChromaDB.")
    parser.add_argument(
        "--file",
        default=str(PROJECT_ROOT / "lore" / "world_setting.md"),
        help="Path to world setting markdown file.",
    )
    args = parser.parse_args()
    md_path = Path(args.file)
    if not md_path.is_absolute():
        md_path = PROJECT_ROOT / md_path
    if not md_path.exists():
        raise FileNotFoundError(f"world setting file not found: {md_path}")
    count = import_world_setting(md_path)
    print(f"Imported world setting chunks: {count}")


if __name__ == "__main__":
    main()

