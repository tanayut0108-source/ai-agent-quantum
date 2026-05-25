"""Persistent knowledge store — long-term memory for the chat agent.

Memories are stored as a JSON file so they survive across sessions.
Each memory has a text, optional tags, and a timestamp.

Usage::

    store = KnowledgeStore("memory.json")
    store.add("Python is a programming language")
    store.add("Earth orbits the Sun", tags=["science"])

    results = store.search("programming")
    context = store.as_context(query="what is Python?", max_items=5)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Memory:
    """A single piece of stored knowledge."""

    text: str
    tags: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def matches(self, query: str) -> bool:
        """Check if this memory is relevant to a query (simple keyword match)."""
        query_lower = query.lower()
        words = query_lower.split()
        text_lower = self.text.lower()
        return any(w in text_lower for w in words if len(w) > 2)


class KnowledgeStore:
    """Persistent key-value knowledge store backed by a JSON file.

    Parameters
    ----------
    path : file path for storing memories (default: ``~/.quantum_agent_memory.json``)
    max_memories : max number of memories to keep (0 = unlimited)
    """

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        max_memories: int = 1000,
    ) -> None:
        if path is None:
            path = Path.home() / ".quantum_agent_memory.json"
        self._path = Path(path)
        self.max_memories = max_memories
        self._memories: list[Memory] = []
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        return len(self._memories)

    def add(self, text: str, tags: list[str] | None = None) -> Memory:
        """Add a new memory and persist to disk."""
        mem = Memory(text=text.strip(), tags=tags or [])
        self._memories.append(mem)
        if self.max_memories > 0 and len(self._memories) > self.max_memories:
            self._memories = self._memories[-self.max_memories :]
        self._save()
        logger.info("Memory added: %s", text[:60])
        return mem

    def search(self, query: str, max_results: int = 10) -> list[Memory]:
        """Search memories by keyword relevance."""
        if not query.strip():
            return list(self._memories[-max_results:])
        matches = [m for m in self._memories if m.matches(query)]
        return matches[-max_results:]

    def search_by_tag(self, tag: str) -> list[Memory]:
        """Return all memories with a specific tag."""
        tag_lower = tag.lower()
        return [m for m in self._memories if tag_lower in [t.lower() for t in m.tags]]

    def get_all(self) -> list[Memory]:
        """Return all memories."""
        return list(self._memories)

    def remove(self, index: int) -> Memory | None:
        """Remove a memory by index (0-based). Returns removed memory or None."""
        if 0 <= index < len(self._memories):
            mem = self._memories.pop(index)
            self._save()
            return mem
        return None

    def clear(self) -> int:
        """Remove all memories. Returns count of removed items."""
        count = len(self._memories)
        self._memories.clear()
        self._save()
        return count

    def as_context(self, query: str = "", max_items: int = 5) -> str:
        """Build a context string from relevant memories for LLM injection.

        Returns an empty string if no memories are found.
        """
        if not self._memories:
            return ""
        if query:
            relevant = self.search(query, max_results=max_items)
        else:
            relevant = self._memories[-max_items:]
        if not relevant:
            return ""
        lines = [f"- {m.text}" for m in relevant]
        return "Knowledge from memory:\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load memories from disk."""
        if not self._path.exists():
            self._memories = []
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._memories = [
                Memory(
                    text=item.get("text", ""),
                    tags=item.get("tags", []),
                    timestamp=item.get("timestamp", ""),
                )
                for item in data
                if item.get("text")
            ]
            logger.info("Loaded %d memories from %s", len(self._memories), self._path)
        except (json.JSONDecodeError, KeyError):
            logger.warning("Could not parse memory file: %s", self._path)
            self._memories = []

    def _save(self) -> None:
        """Persist memories to disk."""
        data = [asdict(m) for m in self._memories]
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def __repr__(self) -> str:
        return f"KnowledgeStore(count={self.count}, path={self._path})"
