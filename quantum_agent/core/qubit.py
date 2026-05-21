"""Quantum state representation using amplitude vectors over hypotheses.

A *Hypothesis* is a single candidate answer / plan / action that an agent
considers.  A *QuantumState* holds many hypotheses in superposition — each
weighted by a complex-valued amplitude whose squared magnitude gives the
probability of that hypothesis being selected when the state is *measured*
(collapsed).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class Hypothesis:
    """One branch of a quantum state superposition."""

    label: str
    data: dict[str, Any] = field(default_factory=dict)
    amplitude: complex = complex(1.0, 0.0)

    @property
    def probability(self) -> float:
        return abs(self.amplitude) ** 2

    def __repr__(self) -> str:
        return f"Hypothesis({self.label!r}, p={self.probability:.4f})"


class QuantumState:
    """Superposition of weighted hypotheses.

    Internally keeps an amplitude vector synchronised with the hypothesis list.
    Supports normalisation, measurement (collapse), phase rotation, and
    amplitude amplification (Grover-style).
    """

    def __init__(self, hypotheses: list[Hypothesis] | None = None) -> None:
        self._hypotheses: list[Hypothesis] = hypotheses or []
        if self._hypotheses:
            self._normalize()

    @property
    def hypotheses(self) -> list[Hypothesis]:
        return list(self._hypotheses)

    @property
    def size(self) -> int:
        return len(self._hypotheses)

    def add(self, hypothesis: Hypothesis) -> None:
        self._hypotheses.append(hypothesis)
        self._normalize()

    def remove(self, label: str) -> None:
        self._hypotheses = [h for h in self._hypotheses if h.label != label]
        if self._hypotheses:
            self._normalize()

    def _normalize(self) -> None:
        total = math.sqrt(sum(h.probability for h in self._hypotheses))
        if total > 0:
            for h in self._hypotheses:
                h.amplitude = h.amplitude / total

    def probabilities(self) -> dict[str, float]:
        return {h.label: h.probability for h in self._hypotheses}

    def measure(self) -> Hypothesis:
        """Collapse the superposition — sample one hypothesis."""
        if not self._hypotheses:
            raise ValueError("Cannot measure an empty quantum state")
        probs = np.array([h.probability for h in self._hypotheses])
        probs = probs / probs.sum()
        idx = int(np.random.choice(len(self._hypotheses), p=probs))
        chosen = self._hypotheses[idx]
        self._hypotheses = [chosen]
        chosen.amplitude = complex(1.0, 0.0)
        return chosen

    def amplify(self, label: str, factor: float = 2.0) -> None:
        """Grover-style amplitude amplification for a target hypothesis."""
        for h in self._hypotheses:
            if h.label == label:
                h.amplitude *= factor
        self._normalize()

    def dampen(self, label: str, factor: float = 0.5) -> None:
        """Reduce the amplitude of a hypothesis (destructive interference)."""
        for h in self._hypotheses:
            if h.label == label:
                h.amplitude *= factor
        self._normalize()

    def phase_rotate(self, label: str, angle: float) -> None:
        """Apply a phase rotation e^{i*angle} to a hypothesis."""
        rotation = complex(math.cos(angle), math.sin(angle))
        for h in self._hypotheses:
            if h.label == label:
                h.amplitude *= rotation
        self._normalize()

    def top_k(self, k: int = 3) -> list[Hypothesis]:
        """Return the k most probable hypotheses."""
        return sorted(self._hypotheses, key=lambda h: h.probability, reverse=True)[:k]

    def entropy(self) -> float:
        """Shannon entropy of the probability distribution."""
        return float(
            -sum(
                h.probability * math.log2(h.probability)
                for h in self._hypotheses
                if h.probability > 0
            )
        )

    def clone(self) -> QuantumState:
        return QuantumState(
            [Hypothesis(h.label, dict(h.data), h.amplitude) for h in self._hypotheses]
        )

    def __repr__(self) -> str:
        top = self.top_k(3)
        entries = ", ".join(repr(h) for h in top)
        return f"QuantumState(n={self.size}, top=[{entries}])"
