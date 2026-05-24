"""3D Geodesic mesh — spider-web-wrapping-football topology for parameters.

The mesh is built by *subdividing an icosahedron* (20-faced polyhedron
whose dual is a soccer-ball / Buckminster-Fuller dome).  Each subdivision
level roughly quadruples the vertex count:

    level 0  →    12 vertices  (raw icosahedron)
    level 1  →    42
    level 2  →   162
    level 3  →   642
    level 4  →  2562
    level 5  → 10242

Multiple concentric shells (layers of the geodesic sphere) are stacked
like layers of a brain, with *radial connections* between corresponding
vertices across shells — forming the spider-web pattern the user
described.

Each vertex holds a :class:`QubitParameter`.  Edges encode entanglement
pathways between neighbouring parameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quantum_agent.core.qubit_parameter import QubitParameter

# ---------------------------------------------------------------------------
# Icosahedron generation and geodesic subdivision
# ---------------------------------------------------------------------------

def _icosahedron_vertices() -> NDArray[np.float64]:
    """Return the 12 unit-sphere vertices of a regular icosahedron."""
    phi = (1.0 + math.sqrt(5.0)) / 2.0  # golden ratio
    verts = np.array([
        [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
        [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
        [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1],
    ], dtype=np.float64)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    return verts / norms


_ICO_FACES: list[tuple[int, int, int]] = [
    (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
    (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
    (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
    (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
]


def subdivide_icosahedron(
    level: int = 1,
) -> tuple[NDArray[np.float64], list[tuple[int, int]]]:
    """Subdivide an icosahedron *level* times and return (vertices, edges).

    Vertices are projected back onto the unit sphere after each split.
    """
    verts_list: list[list[float]] = _icosahedron_vertices().tolist()
    faces: list[tuple[int, int, int]] = list(_ICO_FACES)

    midpoint_cache: dict[tuple[int, int], int] = {}

    def _midpoint(a: int, b: int) -> int:
        key = (min(a, b), max(a, b))
        if key in midpoint_cache:
            return midpoint_cache[key]
        va = np.array(verts_list[a])
        vb = np.array(verts_list[b])
        mid = (va + vb) / 2.0
        norm = float(np.linalg.norm(mid))
        if norm > 0:
            mid /= norm
        idx = len(verts_list)
        verts_list.append(mid.tolist())
        midpoint_cache[key] = idx
        return idx

    for _ in range(level):
        new_faces: list[tuple[int, int, int]] = []
        midpoint_cache = {}
        for a, b, c in faces:
            ab = _midpoint(a, b)
            bc = _midpoint(b, c)
            ca = _midpoint(c, a)
            new_faces.extend([
                (a, ab, ca),
                (b, bc, ab),
                (c, ca, bc),
                (ab, bc, ca),
            ])
        faces = new_faces

    edge_set: set[tuple[int, int]] = set()
    for a, b, c in faces:
        for e in [(a, b), (b, c), (c, a)]:
            edge_set.add((min(e), max(e)))

    vertices = np.array(verts_list, dtype=np.float64)
    edges = sorted(edge_set)
    return vertices, edges


# ---------------------------------------------------------------------------
# GeodesicShell — single spherical layer
# ---------------------------------------------------------------------------

@dataclass
class GeodesicShell:
    """One concentric geodesic sphere holding qubit parameters at vertices."""

    shell_index: int
    radius: float
    vertices: NDArray[np.float64]
    edges: list[tuple[int, int]]
    parameters: list[QubitParameter] = field(default_factory=list)

    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    @property
    def edge_count(self) -> int:
        return len(self.edges)


# ---------------------------------------------------------------------------
# GeodesicMesh3D — multi-shell 3D mesh
# ---------------------------------------------------------------------------


class GeodesicMesh3D:
    """Multi-shell geodesic mesh with radial (spider-web) connections.

    Parameters
    ----------
    num_shells : number of concentric geodesic spheres (default 1000)
    subdivision_level : icosahedron subdivision (controls vertex density)
    inner_radius : radius of the innermost shell
    outer_radius : radius of the outermost shell
    """

    def __init__(
        self,
        num_shells: int = 1000,
        subdivision_level: int = 2,
        inner_radius: float = 1.0,
        outer_radius: float = 100.0,
    ) -> None:
        self.num_shells = num_shells
        self.subdivision_level = subdivision_level
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius

        base_verts, base_edges = subdivide_icosahedron(subdivision_level)
        self._base_vertices = base_verts
        self._base_edges = base_edges
        self._verts_per_shell = len(base_verts)

        self.shells: list[GeodesicShell] = []
        self._radial_edges: list[tuple[tuple[int, int], tuple[int, int]]] = []

        self._build_shells()
        self._build_radial_connections()

    def _build_shells(self) -> None:
        """Create concentric shells with scaled vertices."""
        for i in range(self.num_shells):
            t = i / max(self.num_shells - 1, 1)
            radius = self.inner_radius + t * (self.outer_radius - self.inner_radius)
            scaled = self._base_vertices * radius
            params = [
                QubitParameter.uncertain(name=f"s{i}_v{j}")
                for j in range(self._verts_per_shell)
            ]
            shell = GeodesicShell(
                shell_index=i,
                radius=radius,
                vertices=scaled,
                edges=list(self._base_edges),
                parameters=params,
            )
            self.shells.append(shell)

    def _build_radial_connections(self) -> None:
        """Connect corresponding vertices across adjacent shells (spider-web)."""
        for i in range(self.num_shells - 1):
            for v in range(self._verts_per_shell):
                self._radial_edges.append(((i, v), (i + 1, v)))

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    @property
    def total_parameters(self) -> int:
        return self.num_shells * self._verts_per_shell

    @property
    def vertices_per_shell(self) -> int:
        return self._verts_per_shell

    @property
    def total_edges(self) -> int:
        lateral = len(self._base_edges) * self.num_shells
        radial = len(self._radial_edges)
        return lateral + radial

    def get_parameter(self, shell: int, vertex: int) -> QubitParameter:
        return self.shells[shell].parameters[vertex]

    def set_parameter(self, shell: int, vertex: int, param: QubitParameter) -> None:
        self.shells[shell].parameters[vertex] = param

    def neighbours(self, shell: int, vertex: int) -> list[tuple[int, int]]:
        """Return (shell_idx, vertex_idx) of all connected neighbours."""
        nbrs: list[tuple[int, int]] = []

        for a, b in self.shells[shell].edges:
            if a == vertex:
                nbrs.append((shell, b))
            elif b == vertex:
                nbrs.append((shell, a))

        if shell > 0:
            nbrs.append((shell - 1, vertex))
        if shell < self.num_shells - 1:
            nbrs.append((shell + 1, vertex))

        return nbrs

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def propagate_shell(self, shell_index: int, strength: float = 0.1) -> None:
        """Propagate quantum state along edges within a single shell.

        Each vertex blends a fraction of its neighbours' amplitudes into
        its own state, simulating lateral entanglement.
        """
        shell = self.shells[shell_index]
        new_amps = [p.state.amplitudes.copy() for p in shell.parameters]

        for a, b in shell.edges:
            blend_a = shell.parameters[b].state.amplitudes * strength
            blend_b = shell.parameters[a].state.amplitudes * strength
            new_amps[a] = new_amps[a] + blend_a
            new_amps[b] = new_amps[b] + blend_b

        for idx, p in enumerate(shell.parameters):
            p.state.amplitudes = new_amps[idx]
            p.state._normalize()

    def propagate_radial(self, strength: float = 0.1) -> None:
        """Propagate quantum state between adjacent shells (radial spider-web)."""
        for i in range(self.num_shells - 1):
            for v in range(self._verts_per_shell):
                pa = self.shells[i].parameters[v]
                pb = self.shells[i + 1].parameters[v]
                blend = strength * pa.state.amplitudes
                pb.state.amplitudes = pb.state.amplitudes + blend
                pb.state._normalize()

    def propagate_full(self, strength: float = 0.1) -> None:
        """One full propagation step: lateral + radial."""
        for i in range(self.num_shells):
            self.propagate_shell(i, strength)
        self.propagate_radial(strength)

    def measure_shell(self, shell_index: int) -> list[int]:
        """Collapse all parameters in a shell and return ternary values."""
        return [int(p.measure()) for p in self.shells[shell_index].parameters]

    def expected_values_shell(self, shell_index: int) -> NDArray[np.float64]:
        """Expected float values for all parameters in a shell."""
        return np.array(
            [p.to_float() for p in self.shells[shell_index].parameters],
            dtype=np.float64,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "num_shells": self.num_shells,
            "vertices_per_shell": self._verts_per_shell,
            "total_parameters": self.total_parameters,
            "total_edges": self.total_edges,
            "subdivision_level": self.subdivision_level,
            "inner_radius": self.inner_radius,
            "outer_radius": self.outer_radius,
        }

    def __repr__(self) -> str:
        return (
            f"GeodesicMesh3D(shells={self.num_shells}, "
            f"verts/shell={self._verts_per_shell}, "
            f"total_params={self.total_parameters:,})"
        )
