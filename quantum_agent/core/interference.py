"""Quantum interference — constructive and destructive hypothesis filtering.

Good solutions (high scores from multiple evaluators) experience
*constructive interference* and gain amplitude.  Poor or contradictory
solutions experience *destructive interference* and lose amplitude.
"""

from __future__ import annotations

from collections.abc import Callable

from quantum_agent.core.qubit import Hypothesis, QuantumState

EvaluatorFn = Callable[[Hypothesis], float]


def interference_filter(
    state: QuantumState,
    evaluators: list[EvaluatorFn],
    constructive_threshold: float = 0.7,
    destructive_threshold: float = 0.3,
) -> QuantumState:
    """Apply multi-evaluator interference to a quantum state.

    For each hypothesis every evaluator produces a score in [0, 1].
    - If the *mean* score >= ``constructive_threshold`` the hypothesis
      receives constructive interference (amplitude boost).
    - If the mean score <= ``destructive_threshold`` it receives
      destructive interference (amplitude reduction).
    - Otherwise the amplitude is unchanged.

    Returns a **new** QuantumState (the original is not mutated).
    """
    new_state = state.clone()

    for h in new_state.hypotheses:
        scores = [ev(h) for ev in evaluators]
        mean_score = sum(scores) / len(scores) if scores else 0.5

        if mean_score >= constructive_threshold:
            boost = 1.0 + (mean_score - constructive_threshold) * 3.0
            new_state.amplify(h.label, boost)
        elif mean_score <= destructive_threshold:
            damping = max(0.1, mean_score / destructive_threshold)
            new_state.dampen(h.label, damping)

    return new_state


def pairwise_interference(
    state: QuantumState,
    similarity_fn: Callable[[Hypothesis, Hypothesis], float],
    threshold: float = 0.8,
) -> QuantumState:
    """Merge similar hypotheses via constructive interference.

    If two hypotheses are more similar than *threshold*, the weaker one
    is dampened and the stronger one is amplified — modelling
    constructive interference between near-identical paths.
    """
    new_state = state.clone()
    hypotheses = new_state.hypotheses
    dampened: set[str] = set()

    for i, h1 in enumerate(hypotheses):
        if h1.label in dampened:
            continue
        for h2 in hypotheses[i + 1 :]:
            if h2.label in dampened:
                continue
            sim = similarity_fn(h1, h2)
            if sim >= threshold:
                if h1.probability >= h2.probability:
                    new_state.amplify(h1.label, 1.0 + sim)
                    new_state.dampen(h2.label, 0.3)
                    dampened.add(h2.label)
                else:
                    new_state.amplify(h2.label, 1.0 + sim)
                    new_state.dampen(h1.label, 0.3)
                    dampened.add(h1.label)

    return new_state
