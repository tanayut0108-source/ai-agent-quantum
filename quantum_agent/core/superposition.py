"""Superposition manager — create, evolve, and collapse hypothesis spaces.

The *Superposition* class is the high-level API for generating hypothesis
branches, evaluating them in parallel, and selecting the best via
measurement.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from quantum_agent.core.qubit import Hypothesis, QuantumState

ScoringFn = Callable[[Hypothesis], float]


class Superposition:
    """Manages a QuantumState through its lifecycle.

    1. **Branch** — generate hypotheses from a prompt / context.
    2. **Evolve** — score each hypothesis and translate scores to amplitudes.
    3. **Collapse** — measure the state to select the best hypothesis.
    """

    def __init__(self) -> None:
        self._state = QuantumState()
        self._history: list[QuantumState] = []

    @property
    def state(self) -> QuantumState:
        return self._state

    @property
    def history(self) -> list[QuantumState]:
        return list(self._history)

    def branch(self, hypotheses: list[dict[str, Any]]) -> QuantumState:
        """Create a uniform superposition from raw hypothesis dicts.

        Each dict must contain at least a ``label`` key.
        """
        hyps = [
            Hypothesis(
                label=h["label"],
                data={k: v for k, v in h.items() if k != "label"},
            )
            for h in hypotheses
        ]
        self._state = QuantumState(hyps)
        self._snapshot()
        return self._state

    def evolve(self, scoring_fn: ScoringFn) -> QuantumState:
        """Score every hypothesis and update amplitudes proportionally."""
        for h in self._state.hypotheses:
            score = scoring_fn(h)
            clamped = max(score, 1e-6)
            h.amplitude = complex(clamped, 0.0)
        self._state = QuantumState(self._state.hypotheses)
        self._snapshot()
        return self._state

    def amplify_top(self, k: int = 1, factor: float = 2.0) -> None:
        """Boost the top-k hypotheses (Grover-style)."""
        for h in self._state.top_k(k):
            self._state.amplify(h.label, factor)
        self._snapshot()

    def prune(self, threshold: float = 0.01) -> int:
        """Remove hypotheses below the probability threshold."""
        before = self._state.size
        weak = [h.label for h in self._state.hypotheses if h.probability < threshold]
        for label in weak:
            self._state.remove(label)
        self._snapshot()
        return before - self._state.size

    def collapse(self) -> Hypothesis:
        """Measure (collapse) to a single hypothesis."""
        result = self._state.measure()
        self._snapshot()
        return result

    def collapse_top(self) -> Hypothesis:
        """Deterministically pick the top hypothesis (no randomness)."""
        top = self._state.top_k(1)
        if not top:
            raise ValueError("No hypotheses to collapse")
        chosen = top[0]
        self._state = QuantumState([chosen])
        self._snapshot()
        return chosen

    def _snapshot(self) -> None:
        self._history.append(self._state.clone())

    def __repr__(self) -> str:
        return f"Superposition({self._state})"
