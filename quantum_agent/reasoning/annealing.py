"""Simulated quantum annealing for optimisation over discrete hypothesis spaces.

Mimics quantum annealing: starts at high temperature (broad exploration),
gradually cools (exploitation), and allows quantum tunnelling through
energy barriers via probabilistic state transitions.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from typing import Any

EnergyFn = Callable[[str, dict[str, Any]], float]


def quantum_anneal(
    labels: list[str],
    data_map: dict[str, dict[str, Any]],
    energy_fn: EnergyFn,
    iterations: int = 100,
    initial_temp: float = 2.0,
    cooling_rate: float = 0.95,
    tunnel_probability: float = 0.1,
) -> str:
    """Find the label with the lowest energy via simulated quantum annealing.

    Parameters
    ----------
    labels : candidate labels
    data_map : label -> associated data dict
    energy_fn : returns energy (lower is better) for a (label, data) pair
    iterations : annealing steps
    initial_temp : starting temperature
    cooling_rate : multiplicative cooling per step
    tunnel_probability : chance of a "quantum tunnel" to a random state
    """
    if not labels:
        raise ValueError("Need at least one label to anneal")

    current = random.choice(labels)
    current_energy = energy_fn(current, data_map.get(current, {}))
    best = current
    best_energy = current_energy
    temp = initial_temp

    for _step in range(iterations):
        if random.random() < tunnel_probability:
            candidate = random.choice(labels)
        else:
            idx = labels.index(current)
            offset = random.choice([-1, 1])
            candidate = labels[(idx + offset) % len(labels)]

        candidate_energy = energy_fn(candidate, data_map.get(candidate, {}))
        delta = candidate_energy - current_energy

        if delta < 0 or (temp > 0 and random.random() < math.exp(-delta / temp)):
            current = candidate
            current_energy = candidate_energy

        if current_energy < best_energy:
            best = current
            best_energy = current_energy

        temp *= cooling_rate

    return best


def multi_objective_anneal(
    labels: list[str],
    data_map: dict[str, dict[str, Any]],
    energy_fns: list[EnergyFn],
    weights: list[float] | None = None,
    iterations: int = 100,
    initial_temp: float = 2.0,
) -> str:
    """Quantum annealing with multiple weighted objectives."""
    if not energy_fns:
        raise ValueError("At least one energy function required")
    weights = weights or [1.0 / len(energy_fns)] * len(energy_fns)

    def combined_energy(label: str, data: dict[str, Any]) -> float:
        return sum(
            w * fn(label, data) for w, fn in zip(weights, energy_fns, strict=True)
        )

    return quantum_anneal(
        labels=labels,
        data_map=data_map,
        energy_fn=combined_energy,
        iterations=iterations,
        initial_temp=initial_temp,
    )
