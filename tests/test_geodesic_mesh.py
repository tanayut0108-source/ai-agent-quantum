"""Tests for the geodesic mesh module."""

from __future__ import annotations

import numpy as np
import pytest

from quantum_agent.core.geodesic_mesh import (
    GeodesicMesh3D,
    GeodesicShell,
    subdivide_icosahedron,
)
from quantum_agent.core.qubit_parameter import QubitParameter


class TestSubdivideIcosahedron:
    def test_level_0(self) -> None:
        verts, edges = subdivide_icosahedron(level=0)
        assert len(verts) == 12
        assert len(edges) == 30

    def test_level_1(self) -> None:
        verts, edges = subdivide_icosahedron(level=1)
        assert len(verts) == 42

    def test_level_2(self) -> None:
        verts, edges = subdivide_icosahedron(level=2)
        assert len(verts) == 162

    def test_vertices_on_unit_sphere(self) -> None:
        verts, _ = subdivide_icosahedron(level=2)
        norms = np.linalg.norm(verts, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_edges_are_sorted_pairs(self) -> None:
        _, edges = subdivide_icosahedron(level=1)
        for a, b in edges:
            assert a < b


class TestGeodesicShell:
    def test_basic_properties(self) -> None:
        verts, edges = subdivide_icosahedron(level=1)
        params = [QubitParameter.uncertain(name=f"v{i}") for i in range(len(verts))]
        shell = GeodesicShell(
            shell_index=0,
            radius=1.0,
            vertices=verts,
            edges=edges,
            parameters=params,
        )
        assert shell.vertex_count == 42
        assert shell.edge_count == len(edges)


class TestGeodesicMesh3D:
    def test_small_mesh_creation(self) -> None:
        mesh = GeodesicMesh3D(num_shells=3, subdivision_level=0)
        assert len(mesh.shells) == 3
        assert mesh.vertices_per_shell == 12
        assert mesh.total_parameters == 36

    def test_default_mesh_dimensions(self) -> None:
        mesh = GeodesicMesh3D(num_shells=5, subdivision_level=1)
        assert mesh.vertices_per_shell == 42
        assert mesh.total_parameters == 5 * 42

    def test_get_parameter(self) -> None:
        mesh = GeodesicMesh3D(num_shells=3, subdivision_level=0)
        p = mesh.get_parameter(0, 0)
        assert isinstance(p, QubitParameter)

    def test_set_parameter(self) -> None:
        mesh = GeodesicMesh3D(num_shells=3, subdivision_level=0)
        new_p = QubitParameter.from_float(0.9, name="custom")
        mesh.set_parameter(1, 5, new_p)
        assert mesh.get_parameter(1, 5).name == "custom"

    def test_neighbours_inner(self) -> None:
        mesh = GeodesicMesh3D(num_shells=3, subdivision_level=0)
        nbrs = mesh.neighbours(1, 0)
        shell_indices = {s for s, _ in nbrs}
        assert 0 in shell_indices
        assert 2 in shell_indices
        assert 1 in shell_indices

    def test_neighbours_boundary_first_shell(self) -> None:
        mesh = GeodesicMesh3D(num_shells=3, subdivision_level=0)
        nbrs = mesh.neighbours(0, 0)
        shell_indices = {s for s, _ in nbrs}
        assert 0 not in shell_indices or any(s == 0 for s, _ in nbrs)
        assert 1 in shell_indices

    def test_neighbours_boundary_last_shell(self) -> None:
        mesh = GeodesicMesh3D(num_shells=3, subdivision_level=0)
        nbrs = mesh.neighbours(2, 0)
        shell_indices = {s for s, _ in nbrs}
        assert 1 in shell_indices

    def test_propagate_shell(self) -> None:
        mesh = GeodesicMesh3D(num_shells=3, subdivision_level=0)
        mesh.shells[0].parameters[0] = QubitParameter.from_float(1.0, name="hot")
        mesh.propagate_shell(0, strength=0.3)
        for p in mesh.shells[0].parameters:
            total = sum(p.state.probabilities.values())
            assert total == pytest.approx(1.0, abs=1e-6)

    def test_propagate_radial(self) -> None:
        mesh = GeodesicMesh3D(num_shells=3, subdivision_level=0)
        mesh.shells[0].parameters[0] = QubitParameter.from_float(1.0, name="hot")
        mesh.propagate_radial(strength=0.5)
        p_next = mesh.get_parameter(1, 0)
        assert p_next.to_float() != pytest.approx(0.0, abs=0.01)

    def test_propagate_full(self) -> None:
        mesh = GeodesicMesh3D(num_shells=3, subdivision_level=0)
        mesh.propagate_full(strength=0.1)

    def test_measure_shell(self) -> None:
        mesh = GeodesicMesh3D(num_shells=2, subdivision_level=0)
        results = mesh.measure_shell(0)
        assert len(results) == 12
        assert all(r in (-1, 0, 1) for r in results)

    def test_expected_values_shell(self) -> None:
        mesh = GeodesicMesh3D(num_shells=2, subdivision_level=0)
        vals = mesh.expected_values_shell(0)
        assert len(vals) == 12
        assert all(-1.0 <= v <= 1.0 for v in vals)

    def test_total_edges(self) -> None:
        mesh = GeodesicMesh3D(num_shells=3, subdivision_level=0)
        assert mesh.total_edges > 0

    def test_summary(self) -> None:
        mesh = GeodesicMesh3D(num_shells=3, subdivision_level=1)
        s = mesh.summary()
        assert s["num_shells"] == 3
        assert s["vertices_per_shell"] == 42
        assert "total_parameters" in s

    def test_repr(self) -> None:
        mesh = GeodesicMesh3D(num_shells=5, subdivision_level=0)
        assert "GeodesicMesh3D" in repr(mesh)
