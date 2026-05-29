"""Quantum reasoning engine — chains superposition, interference, and collapse.

The QuantumReasoner orchestrates a full reasoning cycle:
1. Generate hypotheses (superposition)
2. Evaluate with multiple criteria (interference)
3. Amplify promising paths (Grover amplification)
4. Collapse to a final answer
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from quantum_agent.core.interference import interference_filter
from quantum_agent.core.qubit import Hypothesis
from quantum_agent.core.superposition import Superposition


@dataclass
class ReasoningResult:
    answer: str
    confidence: float
    alternatives: list[dict[str, Any]]
    reasoning_steps: list[dict[str, Any]]
    entropy_trace: list[float]


class QuantumReasoner:
    """High-level reasoning pipeline using quantum-inspired primitives."""

    def __init__(
        self,
        max_rounds: int = 5,
        prune_threshold: float = 0.02,
        convergence_entropy: float = 0.5,
    ) -> None:
        self.max_rounds = max_rounds
        self.prune_threshold = prune_threshold
        self.convergence_entropy = convergence_entropy

    def reason(
        self,
        hypotheses: list[dict[str, Any]],
        evaluators: list[Callable[[Hypothesis], float]],
    ) -> ReasoningResult:
        """Run the full quantum reasoning cycle."""
        sup = Superposition()
        sup.branch(hypotheses)

        steps: list[dict[str, Any]] = []
        entropy_trace: list[float] = [sup.state.entropy()]

        for round_idx in range(self.max_rounds):
            new_state = interference_filter(
                sup.state,
                evaluators=evaluators,
            )
            sup._state = new_state

            sup.amplify_top(k=2, factor=1.5)

            pruned = sup.prune(self.prune_threshold)

            current_entropy = sup.state.entropy()
            entropy_trace.append(current_entropy)

            steps.append(
                {
                    "round": round_idx + 1,
                    "pruned": pruned,
                    "remaining": sup.state.size,
                    "entropy": current_entropy,
                    "top_3": [
                        {"label": h.label, "prob": h.probability}
                        for h in sup.state.top_k(3)
                    ],
                }
            )

            if current_entropy < self.convergence_entropy:
                break

        chosen = sup.collapse_top()
        alternatives = [
            {"label": h.label, "probability": h.probability}
            for h in sup.state.hypotheses
            if h.label != chosen.label
        ]

        return ReasoningResult(
            answer=chosen.label,
            confidence=chosen.probability,
            alternatives=alternatives,
            reasoning_steps=steps,
            entropy_trace=entropy_trace,
        )
