"""Tests for the quantum layer module."""

from __future__ import annotations

import numpy as np
import pytest

from quantum_agent.core.geodesic_mesh import GeodesicMesh3D
from quantum_agent.core.quantum_layer import (
    QuantumLayer,
    QuantumLayerStack,
    _amplifying_transform,
    _default_transform,
    _interference_transform,
)
from quantum_agent.core.qubit_parameter import QubitParameter


class TestTransforms:
    def test_default_transform_no_neighbours(self) -> None:
        p = QubitParameter.superposed()
        _default_transform(p, [])
        total = sum(p.state.probabilities.values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_default_transform_with_neighbours(self) -> None:
        p = QubitParameter.from_float(0.5)
        nbrs = [QubitParameter.from_float(-0.5), QubitParameter.from_float(0.8)]
        _default_transform(p, nbrs)
        total = sum(p.state.probabilities.values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_amplifying_transform(self) -> None:
        p = QubitParameter.superposed()
        nbrs = [QubitParameter.from_bool(True), QubitParameter.from_bool(True)]
        _amplifying_transform(p, nbrs)
        total = sum(p.state.probabilities.values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_interference_transform(self) -> None:
        p = QubitParameter.superposed()
        nbrs = [QubitParameter.superposed()]
        _interference_transform(p, nbrs)
        total = sum(p.state.probabilities.values())
        assert total == pytest.approx(1.0, abs=1e-6)


class TestQuantumLayer:
    def test_forward_single_layer(self) -> None:
        mesh = GeodesicMesh3D(num_shells=3, subdivision_level=0)
        layer = QuantumLayer(layer_index=0)
        stats = layer.forward(mesh)
        assert "mean_value" in stats
        assert "mean_entropy" in stats
        assert stats["layer"] == 0

    def test_forward_with_activation(self) -> None:
        mesh = GeodesicMesh3D(num_shells=3, subdivision_level=0)
        layer = QuantumLayer(layer_index=0, activation_angle=0.3)
        stats = layer.forward(mesh)
        assert "mean_value" in stats


class TestQuantumLayerStack:
    def test_creation_with_mesh(self) -> None:
        mesh = GeodesicMesh3D(num_shells=5, subdivision_level=0)
        stack = QuantumLayerStack(mesh=mesh, num_layers=5)
        assert stack.num_layers == 5
        assert stack.total_parameters == 5 * 12

    def test_creation_without_mesh(self) -> None:
        stack = QuantumLayerStack(num_layers=3)
        assert stack.num_layers == 3
        assert stack.mesh.num_shells == 3

    def test_forward_pass(self) -> None:
        mesh = GeodesicMesh3D(num_shells=3, subdivision_level=0)
        stack = QuantumLayerStack(mesh=mesh, num_layers=3)
        stats = stack.forward()
        assert len(stats) == 3
        assert all("mean_value" in s for s in stats)

    def test_forward_range(self) -> None:
        mesh = GeodesicMesh3D(num_shells=5, subdivision_level=0)
        stack = QuantumLayerStack(mesh=mesh, num_layers=5)
        stats = stack.forward_range(start=1, end=3)
        assert len(stats) == 2

    def test_inject_input(self) -> None:
        mesh = GeodesicMesh3D(num_shells=3, subdivision_level=0)
        stack = QuantumLayerStack(mesh=mesh, num_layers=3)
        values = np.array([0.5] * 12, dtype=np.float64)
        stack.inject_input(values, shell_index=0)
        for p in mesh.shells[0].parameters:
            assert p.to_float() > 0.0

    def test_read_output(self) -> None:
        mesh = GeodesicMesh3D(num_shells=3, subdivision_level=0)
        stack = QuantumLayerStack(mesh=mesh, num_layers=3)
        output = stack.read_output()
        assert len(output) == 12
        assert all(-1.0 <= v <= 1.0 for v in output)

    def test_measure_output(self) -> None:
        mesh = GeodesicMesh3D(num_shells=3, subdivision_level=0)
        stack = QuantumLayerStack(mesh=mesh, num_layers=3)
        results = stack.measure_output()
        assert len(results) == 12
        assert all(r in (-1, 0, 1) for r in results)

    def test_convergence_trace(self) -> None:
        mesh = GeodesicMesh3D(num_shells=3, subdivision_level=0)
        stack = QuantumLayerStack(mesh=mesh, num_layers=3)
        stack.forward()
        stack.forward()
        trace = stack.convergence_trace()
        assert len(trace) == 2

    def test_history(self) -> None:
        mesh = GeodesicMesh3D(num_shells=3, subdivision_level=0)
        stack = QuantumLayerStack(mesh=mesh, num_layers=3)
        stack.forward()
        assert len(stack.history) == 1

    def test_summary(self) -> None:
        mesh = GeodesicMesh3D(num_shells=3, subdivision_level=0)
        stack = QuantumLayerStack(mesh=mesh, num_layers=3)
        s = stack.summary()
        assert s["num_layers"] == 3
        assert s["num_shells"] == 3
        assert "total_parameters" in s

    def test_custom_transform_list(self) -> None:
        mesh = GeodesicMesh3D(num_shells=3, subdivision_level=0)
        transforms = [_default_transform, _amplifying_transform, _interference_transform]
        stack = QuantumLayerStack(mesh=mesh, num_layers=3, transform=transforms)
        stats = stack.forward()
        assert len(stats) == 3

    def test_repr(self) -> None:
        mesh = GeodesicMesh3D(num_shells=3, subdivision_level=0)
        stack = QuantumLayerStack(mesh=mesh, num_layers=3)
        assert "QuantumLayerStack" in repr(stack)

    def test_end_to_end_small(self) -> None:
        """Full pipeline: inject → forward → read output."""
        mesh = GeodesicMesh3D(num_shells=5, subdivision_level=0)
        stack = QuantumLayerStack(mesh=mesh, num_layers=5)

        inputs = np.random.uniform(-1, 1, size=12).astype(np.float64)
        stack.inject_input(inputs, shell_index=0)

        stack.forward()

        output = stack.read_output(shell_index=4)
        assert len(output) == 12
        assert all(-1.0 <= v <= 1.0 for v in output)
