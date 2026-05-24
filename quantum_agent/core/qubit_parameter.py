"""Quantum qubit parameters — convert regular parameters to quantum qubits.

A *QubitParameter* wraps a scalar value into a ternary quantum state so that
it can exist in superposition of ``-1`` (negative / no), ``0`` (uncertain),
and ``+1`` (positive / yes) simultaneously.

Converting a classical float to a qubit parameter:
    1. Map the float onto the ternary basis amplitude vector.
    2. The parameter now *is* a qutrit — it can be evolved, measured,
       entangled with neighbours, and processed through quantum layers.

Collections of qubit parameters form a *ParameterRegister* which supports
bulk operations (measure all, evolve all, entangle pairs).
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quantum_agent.core.ternary_logic import (
    NO,
    UNCERTAIN,
    YES,
    TernaryBasis,
    TernaryState,
)


@dataclass
class QubitParameter:
    """A single parameter represented as a ternary qubit (qutrit).

    Attributes
    ----------
    name : human-readable identifier
    state : the underlying ternary quantum state
    metadata : arbitrary key-value metadata (e.g. layer index, position)
    """

    name: str = field(default_factory=lambda: f"qp-{uuid.uuid4().hex[:6]}")
    state: TernaryState = field(default_factory=TernaryState)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Factory methods — convert classical values to qubit parameters
    # ------------------------------------------------------------------

    @classmethod
    def from_float(cls, value: float, name: str = "") -> QubitParameter:
        """Convert a scalar in [-1, +1] to a qubit parameter.

        The mapping distributes amplitude across the three basis states
        proportional to proximity:

        * ``value = -1``  →  pure |−1⟩
        * ``value =  0``  →  pure |0⟩
        * ``value = +1``  →  pure |+1⟩
        * values in between → superposition
        """
        clamped = max(-1.0, min(1.0, value))

        if clamped < 0:
            a_neg = math.sqrt(abs(clamped))
            a_zero = math.sqrt(1.0 - abs(clamped))
            a_pos = 0.0
        elif clamped > 0:
            a_neg = 0.0
            a_zero = math.sqrt(1.0 - abs(clamped))
            a_pos = math.sqrt(abs(clamped))
        else:
            a_neg = 0.0
            a_zero = 1.0
            a_pos = 0.0

        amps = np.array([a_neg, a_zero, a_pos], dtype=np.complex128)
        return cls(
            name=name or f"qp-{uuid.uuid4().hex[:6]}",
            state=TernaryState(amps),
        )

    @classmethod
    def from_bool(cls, value: bool, name: str = "") -> QubitParameter:
        """True → pure |+1⟩, False → pure |−1⟩."""
        amps = (
            np.array([0.0, 0.0, 1.0], dtype=np.complex128) if value
            else np.array([1.0, 0.0, 0.0], dtype=np.complex128)
        )
        return cls(
            name=name or f"qp-{uuid.uuid4().hex[:6]}",
            state=TernaryState(amps),
        )

    @classmethod
    def uncertain(cls, name: str = "") -> QubitParameter:
        """Create a parameter in the fully uncertain state |0⟩."""
        return cls(
            name=name or f"qp-{uuid.uuid4().hex[:6]}",
            state=TernaryState(),
        )

    @classmethod
    def superposed(cls, name: str = "") -> QubitParameter:
        """Create a parameter in equal superposition of all three states."""
        s = TernaryState()
        s.superpose_uniform()
        return cls(name=name or f"qp-{uuid.uuid4().hex[:6]}", state=s)

    # ------------------------------------------------------------------
    # Quantum operations
    # ------------------------------------------------------------------

    def measure(self) -> TernaryBasis:
        """Collapse to a classical ternary value."""
        return self.state.measure()

    def to_float(self) -> float:
        """Expected value: ⟨ψ|V̂|ψ⟩ where V̂ = diag(-1, 0, +1)."""
        p = self.state.probabilities
        return -1.0 * p[NO] + 0.0 * p[UNCERTAIN] + 1.0 * p[YES]

    def rotate(self, angle: float) -> None:
        """Phase rotation on the qutrit."""
        self.state.rotate(angle)

    def amplify(self, basis: TernaryBasis, factor: float = 2.0) -> None:
        """Boost the amplitude of a specific basis state."""
        idx = TernaryBasis.index(basis)
        self.state.amplitudes[idx] *= factor
        self.state._normalize()

    def dampen(self, basis: TernaryBasis, factor: float = 0.5) -> None:
        """Reduce the amplitude of a specific basis state."""
        idx = TernaryBasis.index(basis)
        self.state.amplitudes[idx] *= factor
        self.state._normalize()

    @property
    def entropy(self) -> float:
        return self.state.entropy()

    def clone(self) -> QubitParameter:
        return QubitParameter(
            name=self.name,
            state=self.state.clone(),
            metadata=dict(self.metadata),
        )

    def __repr__(self) -> str:
        return f"QubitParameter({self.name!r}, E[v]={self.to_float():.4f}, H={self.entropy:.3f})"


# ---------------------------------------------------------------------------
# Parameter register — bulk operations on many qubit parameters
# ---------------------------------------------------------------------------


class ParameterRegister:
    """A collection of qubit parameters supporting bulk quantum operations.

    This register can hold up to billions of parameters using a sparse
    representation — only parameters that have been explicitly created or
    modified are stored in memory.  Uncreated positions are implicitly
    in the |0⟩ (uncertain) state.
    """

    def __init__(self, capacity: int = 1_000_000_000_000) -> None:
        self._params: dict[str, QubitParameter] = {}
        self.capacity = capacity

    @property
    def size(self) -> int:
        return len(self._params)

    def add(self, param: QubitParameter) -> None:
        self._params[param.name] = param

    def get(self, name: str) -> QubitParameter | None:
        return self._params.get(name)

    def remove(self, name: str) -> bool:
        return self._params.pop(name, None) is not None

    def measure_all(self) -> dict[str, TernaryBasis]:
        """Collapse every parameter and return classical ternary values."""
        return {name: p.measure() for name, p in self._params.items()}

    def expected_values(self) -> dict[str, float]:
        """Expected float value for every parameter (no collapse)."""
        return {name: p.to_float() for name, p in self._params.items()}

    def bulk_rotate(self, angle: float) -> None:
        """Apply the same phase rotation to all parameters."""
        for p in self._params.values():
            p.rotate(angle)

    def entangle_pair(self, name_a: str, name_b: str, strength: float = 0.5) -> None:
        """Entangle two parameters by blending their amplitude vectors.

        After entanglement, changing one will be correlated with the other
        (simulated via shared amplitude bias).
        """
        pa = self._params.get(name_a)
        pb = self._params.get(name_b)
        if pa is None or pb is None:
            return

        blended = strength * pa.state.amplitudes + (1.0 - strength) * pb.state.amplitudes
        pa.state.amplitudes = (
            (1.0 - strength) * pa.state.amplitudes + strength * blended
        )
        pb.state.amplitudes = (
            (1.0 - strength) * pb.state.amplitudes + strength * blended
        )
        pa.state._normalize()
        pb.state._normalize()

    def from_float_array(
        self,
        values: NDArray[np.floating[Any]],
        prefix: str = "p",
    ) -> list[str]:
        """Bulk-convert a numpy array of floats to qubit parameters.

        Returns the list of parameter names created.
        """
        names: list[str] = []
        for i, v in enumerate(values.flat):
            name = f"{prefix}_{i}"
            self.add(QubitParameter.from_float(float(v), name=name))
            names.append(name)
        return names

    def to_float_array(self, names: list[str]) -> NDArray[np.float64]:
        """Get expected values for a list of parameter names."""
        return np.array(
            [self._params[n].to_float() if n in self._params else 0.0 for n in names],
            dtype=np.float64,
        )

    def summary(self) -> dict[str, Any]:
        """Aggregate statistics over all stored parameters."""
        if not self._params:
            return {"count": 0}
        vals = [p.to_float() for p in self._params.values()]
        ents = [p.entropy for p in self._params.values()]
        return {
            "count": len(vals),
            "capacity": self.capacity,
            "mean_value": float(np.mean(vals)),
            "std_value": float(np.std(vals)),
            "mean_entropy": float(np.mean(ents)),
        }

    def items(self) -> list[tuple[str, QubitParameter]]:
        return list(self._params.items())

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return f"ParameterRegister(active={self.size}, capacity={self.capacity:,})"
