"""Quantum layer stack — 1000-layer deep processing with geodesic mesh wiring.

A *QuantumLayerStack* is a pipeline of *QuantumLayer* instances.  Each
layer:

1. Receives input qubit parameters arranged on a geodesic shell.
2. Applies a *ternary quantum transform* (rotation + entanglement with
   neighbours on the mesh).
3. Optionally measures or passes the evolved state to the next layer.

The full stack processes information from the innermost shell to the
outermost — like signal propagation through layers of a brain wrapped
in a geodesic web.

Scaling note: with subdivision level 2 (162 vertices per shell) and
1 000 shells the stack holds 162 000 active qubit parameters.  At
higher subdivision levels or with virtual parameter expansion the
register can address up to 1 000 000 000 000 (1 T) parameter slots.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quantum_agent.core.geodesic_mesh import GeodesicMesh3D
from quantum_agent.core.qubit_parameter import QubitParameter

# ---------------------------------------------------------------------------
# Transform functions (pluggable per layer)
# ---------------------------------------------------------------------------

TransformFn = Callable[[QubitParameter, list[QubitParameter]], None]


def _default_transform(param: QubitParameter, neighbours: list[QubitParameter]) -> None:
    """Default layer transform: rotate + blend with neighbours."""
    if not neighbours:
        return
    nbr_amps = np.stack([n.state.amplitudes for n in neighbours])
    mean_nbr = nbr_amps.mean(axis=0)
    param.state.amplitudes = 0.8 * param.state.amplitudes + 0.2 * mean_nbr
    param.state._normalize()


def _amplifying_transform(param: QubitParameter, neighbours: list[QubitParameter]) -> None:
    """Boost the dominant state based on neighbour consensus."""
    if not neighbours:
        return
    nbr_amps = np.stack([n.state.amplitudes for n in neighbours])
    consensus = nbr_amps.mean(axis=0)
    dominant_idx = int(np.argmax(np.abs(consensus) ** 2))
    param.state.amplitudes[dominant_idx] *= 1.5
    param.state._normalize()


def _interference_transform(param: QubitParameter, neighbours: list[QubitParameter]) -> None:
    """Constructive/destructive interference with neighbours."""
    if not neighbours:
        return
    for nbr in neighbours:
        dot = float(np.real(np.vdot(param.state.amplitudes, nbr.state.amplitudes)))
        if dot > 0.5:
            param.state.amplitudes += 0.1 * nbr.state.amplitudes
        elif dot < -0.5:
            param.state.amplitudes -= 0.1 * nbr.state.amplitudes
    param.state._normalize()


# ---------------------------------------------------------------------------
# QuantumLayer
# ---------------------------------------------------------------------------


@dataclass
class QuantumLayer:
    """A single processing layer operating on one geodesic shell.

    Attributes
    ----------
    layer_index : position in the stack (0 = innermost)
    transform : callable that evolves each parameter given its neighbours
    activation_angle : optional phase rotation applied after the transform
    """

    layer_index: int = 0
    transform: TransformFn = field(default=_default_transform)
    activation_angle: float = 0.0
    _stats: dict[str, float] = field(default_factory=dict)

    def forward(self, mesh: GeodesicMesh3D) -> dict[str, float]:
        """Process one shell of the mesh (index = layer_index).

        Returns per-layer statistics (mean expected value, mean entropy).
        """
        shell_idx = self.layer_index % mesh.num_shells
        shell = mesh.shells[shell_idx]

        for v_idx, param in enumerate(shell.parameters):
            nbr_coords = mesh.neighbours(shell_idx, v_idx)
            nbr_params = [mesh.get_parameter(s, v) for s, v in nbr_coords]
            self.transform(param, nbr_params)

            if self.activation_angle != 0.0:
                param.rotate(self.activation_angle)

        vals = [p.to_float() for p in shell.parameters]
        ents = [p.entropy for p in shell.parameters]
        self._stats = {
            "layer": self.layer_index,
            "mean_value": float(np.mean(vals)),
            "std_value": float(np.std(vals)),
            "mean_entropy": float(np.mean(ents)),
        }
        return dict(self._stats)


# ---------------------------------------------------------------------------
# QuantumLayerStack — the full 1000-layer pipeline
# ---------------------------------------------------------------------------


class QuantumLayerStack:
    """Stack of QuantumLayers wired through a GeodesicMesh3D.

    Parameters
    ----------
    mesh : the underlying 3D geodesic mesh (provides shells + connectivity)
    num_layers : number of processing layers (defaults to mesh.num_shells)
    transform : per-layer transform function (or per-layer list)
    activation_angle : phase rotation applied after each layer
    """

    def __init__(
        self,
        mesh: GeodesicMesh3D | None = None,
        num_layers: int = 1000,
        transform: TransformFn | list[TransformFn] | None = None,
        activation_angle: float = 0.0,
    ) -> None:
        if mesh is None:
            mesh = GeodesicMesh3D(num_shells=num_layers, subdivision_level=2)
        self.mesh = mesh
        self.num_layers = num_layers

        if isinstance(transform, list):
            transforms = transform
        else:
            transforms = [transform or _default_transform] * num_layers

        self.layers = [
            QuantumLayer(
                layer_index=i,
                transform=transforms[i % len(transforms)],
                activation_angle=activation_angle,
            )
            for i in range(num_layers)
        ]
        self._history: list[list[dict[str, float]]] = []

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, propagate_radial: bool = True) -> list[dict[str, float]]:
        """Run a full forward pass through all layers (inner → outer).

        After each layer's local processing, an optional radial
        propagation step pushes information outward through the
        spider-web connections.
        """
        stats: list[dict[str, float]] = []
        for layer in self.layers:
            layer_stats = layer.forward(self.mesh)
            stats.append(layer_stats)

        if propagate_radial:
            self.mesh.propagate_radial(strength=0.05)

        self._history.append(stats)
        return stats

    def forward_range(
        self,
        start: int = 0,
        end: int | None = None,
        propagate_radial: bool = True,
    ) -> list[dict[str, float]]:
        """Process a sub-range of layers."""
        end = end or self.num_layers
        stats: list[dict[str, float]] = []
        for layer in self.layers[start:end]:
            stats.append(layer.forward(self.mesh))
        if propagate_radial:
            self.mesh.propagate_radial(strength=0.05)
        self._history.append(stats)
        return stats

    # ------------------------------------------------------------------
    # Input / output
    # ------------------------------------------------------------------

    def inject_input(
        self,
        values: NDArray[np.floating[Any]],
        shell_index: int = 0,
    ) -> None:
        """Load classical float values into the innermost (or specified) shell.

        ``values`` length must match vertices_per_shell.
        """
        shell = self.mesh.shells[shell_index]
        for i, v in enumerate(values[: len(shell.parameters)]):
            name = f"in_s{shell_index}_v{i}"
            shell.parameters[i] = QubitParameter.from_float(float(v), name=name)

    def read_output(self, shell_index: int = -1) -> NDArray[np.float64]:
        """Read expected float values from the outermost (or specified) shell."""
        idx = shell_index if shell_index >= 0 else self.mesh.num_shells - 1
        return self.mesh.expected_values_shell(idx)

    def measure_output(self, shell_index: int = -1) -> list[int]:
        """Collapse outermost shell to classical ternary values."""
        idx = shell_index if shell_index >= 0 else self.mesh.num_shells - 1
        return self.mesh.measure_shell(idx)

    # ------------------------------------------------------------------
    # Statistics / introspection
    # ------------------------------------------------------------------

    @property
    def total_parameters(self) -> int:
        return self.mesh.total_parameters

    @property
    def history(self) -> list[list[dict[str, float]]]:
        return list(self._history)

    def convergence_trace(self) -> list[float]:
        """Mean entropy per forward pass — shows convergence over time."""
        trace: list[float] = []
        for run in self._history:
            mean_ent = float(np.mean([s.get("mean_entropy", 0.0) for s in run]))
            trace.append(mean_ent)
        return trace

    def summary(self) -> dict[str, Any]:
        mesh_summary = self.mesh.summary()
        return {
            **mesh_summary,
            "num_layers": self.num_layers,
            "forward_passes": len(self._history),
            "convergence": self.convergence_trace(),
        }

    def __repr__(self) -> str:
        return (
            f"QuantumLayerStack(layers={self.num_layers}, "
            f"params={self.total_parameters:,}, "
            f"passes={len(self._history)})"
        )
