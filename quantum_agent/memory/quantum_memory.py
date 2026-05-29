"""Quantum-inspired associative memory with amplitude-based retrieval.

Each memory entry has a *relevance amplitude* that decays over time
(decoherence).  Retrieval boosts matching entries (measurement) while
non-matching entries decay — mimicking quantum measurement back-action.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class MemoryEntry:
    key: str
    value: Any
    amplitude: float = 1.0
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    tags: list[str] = field(default_factory=list)

    @property
    def probability(self) -> float:
        return self.amplitude ** 2

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at


class QuantumMemory:
    """Associative memory with quantum-inspired dynamics.

    Features:
    - **Amplitude decay** (decoherence): unused memories fade.
    - **Retrieval amplification**: accessed memories get stronger.
    - **Interference**: similar memories can reinforce or cancel.
    """

    def __init__(
        self,
        capacity: int = 1000,
        decay_rate: float = 0.001,
        retrieval_boost: float = 1.2,
    ) -> None:
        self._entries: dict[str, MemoryEntry] = {}
        self.capacity = capacity
        self.decay_rate = decay_rate
        self.retrieval_boost = retrieval_boost

    def store(
        self,
        key: str,
        value: Any,
        relevance: float = 1.0,
        tags: list[str] | None = None,
    ) -> None:
        if len(self._entries) >= self.capacity:
            self._evict()

        self._entries[key] = MemoryEntry(
            key=key,
            value=value,
            amplitude=math.sqrt(relevance),
            tags=tags or [],
        )

    def recall(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Retrieve memories most relevant to the query.

        Matching entries are amplified, others are dampened.
        """
        self._apply_decoherence()

        scored: list[tuple[float, MemoryEntry]] = []
        for entry in self._entries.values():
            similarity = self._similarity(query, entry)
            effective_amp = entry.amplitude * (1.0 + similarity)
            scored.append((effective_amp, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        results: list[dict[str, Any]] = []

        for amp, entry in scored[:top_k]:
            entry.amplitude = min(entry.amplitude * self.retrieval_boost, 1.0)
            entry.access_count += 1
            results.append(
                {
                    "key": entry.key,
                    "value": entry.value,
                    "relevance": amp**2,
                    "access_count": entry.access_count,
                    "tags": entry.tags,
                }
            )

        return results

    def forget(self, key: str) -> bool:
        """Explicitly remove a memory (forced decoherence)."""
        return self._entries.pop(key, None) is not None

    def size(self) -> int:
        return len(self._entries)

    def snapshot(self) -> list[dict[str, Any]]:
        return [
            {
                "key": e.key,
                "amplitude": e.amplitude,
                "probability": e.probability,
                "access_count": e.access_count,
                "age_seconds": e.age_seconds,
            }
            for e in self._entries.values()
        ]

    def _similarity(self, query: str, entry: MemoryEntry) -> float:
        """Simple token-overlap similarity."""
        query_tokens = set(query.lower().split())
        key_tokens = set(entry.key.lower().split())
        tag_tokens = {t.lower() for t in entry.tags}
        combined = key_tokens | tag_tokens

        if not query_tokens or not combined:
            return 0.0

        overlap = len(query_tokens & combined)
        return overlap / max(len(query_tokens), len(combined))

    def _apply_decoherence(self) -> None:
        """Decay amplitudes based on age (quantum decoherence)."""
        for entry in self._entries.values():
            decay = math.exp(-self.decay_rate * entry.age_seconds)
            entry.amplitude *= decay

    def _evict(self) -> None:
        """Remove the weakest memory to make room."""
        if not self._entries:
            return
        weakest = min(self._entries.values(), key=lambda e: e.amplitude)
        del self._entries[weakest.key]

    def __repr__(self) -> str:
        return f"QuantumMemory(size={self.size()}, capacity={self.capacity})"
