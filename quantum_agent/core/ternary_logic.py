"""Ternary quantum logic — three-valued superposition states.

A *TernaryState* extends the classical binary qubit to three basis states:

    |−1⟩  →  NO / negative / reject
    | 0⟩  →  UNCERTAIN / empty / superposed
    |+1⟩  →  YES / positive / accept

Each ternary qubit (qutrit) is represented by a 3-element complex amplitude
vector ``[α₋₁, α₀, α₊₁]`` whose squared magnitudes sum to 1.  The value
*starts from 0* (uncertain) and evolves through operations until measured.

Logical operations (AND, OR, NOT) work on amplitude vectors so that
superposition is preserved throughout computation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray


class TernaryBasis(IntEnum):
    """Basis states for ternary logic."""

    NEGATIVE = -1  # NO
    ZERO = 0       # UNCERTAIN / empty
    POSITIVE = 1   # YES

    @classmethod
    def index(cls, value: TernaryBasis) -> int:
        """Map basis state to array index (0, 1, 2)."""
        return {cls.NEGATIVE: 0, cls.ZERO: 1, cls.POSITIVE: 2}[value]


# Convenience aliases
NO = TernaryBasis.NEGATIVE
UNCERTAIN = TernaryBasis.ZERO
YES = TernaryBasis.POSITIVE


@dataclass
class TernaryState:
    """A single qutrit — three-valued quantum state.

    The amplitude vector ``[α₋₁, α₀, α₊₁]`` encodes the superposition
    over the three basis states.  The state begins at |0⟩ (uncertain)
    by default.
    """

    amplitudes: NDArray[np.complex128] = field(
        default_factory=lambda: np.array([0.0, 1.0, 0.0], dtype=np.complex128)
    )

    def __post_init__(self) -> None:
        self.amplitudes = np.asarray(self.amplitudes, dtype=np.complex128)
        self._normalize()

    def _normalize(self) -> None:
        norm = float(np.linalg.norm(self.amplitudes))
        if norm > 0:
            self.amplitudes /= norm

    @property
    def probabilities(self) -> dict[TernaryBasis, float]:
        """Probability of each basis state."""
        return {
            NO: float(np.abs(self.amplitudes[0]) ** 2),
            UNCERTAIN: float(np.abs(self.amplitudes[1]) ** 2),
            YES: float(np.abs(self.amplitudes[2]) ** 2),
        }

    @property
    def dominant(self) -> TernaryBasis:
        """The most probable basis state (no collapse)."""
        probs = self.probabilities
        return max(probs, key=lambda k: probs[k])

    def measure(self) -> TernaryBasis:
        """Collapse the qutrit — probabilistically select one basis state."""
        p = np.abs(self.amplitudes) ** 2
        p = p / p.sum()
        idx = int(np.random.choice(3, p=p))
        self.amplitudes = np.zeros(3, dtype=np.complex128)
        self.amplitudes[idx] = 1.0
        return [NO, UNCERTAIN, YES][idx]

    def set_pure(self, basis: TernaryBasis) -> None:
        """Set to a pure basis state (no superposition)."""
        self.amplitudes = np.zeros(3, dtype=np.complex128)
        self.amplitudes[TernaryBasis.index(basis)] = 1.0

    def superpose_uniform(self) -> None:
        """Equal superposition of all three basis states."""
        self.amplitudes = np.ones(3, dtype=np.complex128) / math.sqrt(3)

    def rotate(self, angle: float) -> None:
        """Apply a phase rotation across the amplitude vector."""
        phase = np.exp(1j * angle * np.array([-1.0, 0.0, 1.0]))
        self.amplitudes *= phase
        self._normalize()

    def entropy(self) -> float:
        """Shannon entropy of the probability distribution."""
        probs = np.abs(self.amplitudes) ** 2
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log2(probs)))

    def clone(self) -> TernaryState:
        return TernaryState(self.amplitudes.copy())

    def __repr__(self) -> str:
        p = self.probabilities
        return (
            f"TernaryState(NO={p[NO]:.3f}, "
            f"UNCERTAIN={p[UNCERTAIN]:.3f}, "
            f"YES={p[YES]:.3f})"
        )


# ---------------------------------------------------------------------------
# Ternary logic gates (operate on amplitude vectors)
# ---------------------------------------------------------------------------

def ternary_not(state: TernaryState) -> TernaryState:
    """Negate: swap |−1⟩ and |+1⟩, leave |0⟩ unchanged."""
    a = state.amplitudes.copy()
    return TernaryState(np.array([a[2], a[1], a[0]], dtype=np.complex128))


def ternary_and(a: TernaryState, b: TernaryState) -> TernaryState:
    """Ternary AND via outer-product combination.

    Truth table (dominant values):
        AND | -1  |  0  | +1
        -1  | -1  | -1  | -1
         0  | -1  |  0  |  0
        +1  | -1  |  0  | +1

    In superposition the result amplitudes are computed from the
    joint probability distribution, mapping each pair to its output
    basis state.
    """
    result = np.zeros(3, dtype=np.complex128)
    mapping = [
        # (idx_a, idx_b) → result_idx
        (0, 0, 0), (0, 1, 0), (0, 2, 0),  # -1 AND * = -1
        (1, 0, 0), (1, 1, 1), (1, 2, 1),  # 0 AND -1=-1, 0 AND 0=0, 0 AND +1=0
        (2, 0, 0), (2, 1, 1), (2, 2, 2),  # +1 AND -1=-1, +1 AND 0=0, +1 AND +1=+1
    ]
    for ia, ib, ir in mapping:
        result[ir] += a.amplitudes[ia] * b.amplitudes[ib]
    return TernaryState(result)


def ternary_or(a: TernaryState, b: TernaryState) -> TernaryState:
    """Ternary OR via outer-product combination.

    Truth table (dominant values):
        OR  | -1  |  0  | +1
        -1  | -1  |  0  | +1
         0  |  0  |  0  | +1
        +1  | +1  | +1  | +1
    """
    result = np.zeros(3, dtype=np.complex128)
    mapping = [
        (0, 0, 0), (0, 1, 1), (0, 2, 2),
        (1, 0, 1), (1, 1, 1), (1, 2, 2),
        (2, 0, 2), (2, 1, 2), (2, 2, 2),
    ]
    for ia, ib, ir in mapping:
        result[ir] += a.amplitudes[ia] * b.amplitudes[ib]
    return TernaryState(result)


def ternary_consensus(states: list[TernaryState]) -> TernaryState:
    """Combine multiple ternary states via amplitude averaging."""
    if not states:
        return TernaryState()
    stacked = np.stack([s.amplitudes for s in states])
    mean_amp = stacked.mean(axis=0)
    return TernaryState(mean_amp)
