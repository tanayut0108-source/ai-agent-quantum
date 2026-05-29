"""Tests for quantum annealing."""

import pytest

from quantum_agent.reasoning.annealing import multi_objective_anneal, quantum_anneal


class TestQuantumAnneal:
    def test_finds_minimum_energy(self):
        labels = ["a", "b", "c"]
        data = {"a": {"cost": 10}, "b": {"cost": 1}, "c": {"cost": 5}}
        best = quantum_anneal(
            labels, data, lambda lbl, d: d.get("cost", 0), iterations=500
        )
        assert best == "b"

    def test_empty_labels_raises(self):
        with pytest.raises(ValueError):
            quantum_anneal([], {}, lambda lbl, d: 0)

    def test_single_label(self):
        best = quantum_anneal(["only"], {"only": {}}, lambda lbl, d: 0)
        assert best == "only"

    def test_tunnel_probability(self):
        labels = ["a", "b", "c", "d", "e"]
        data = {lbl: {"val": i} for i, lbl in enumerate(labels)}
        best = quantum_anneal(
            labels, data, lambda lbl, d: d["val"], iterations=200, tunnel_probability=0.5
        )
        assert best == "a"


class TestMultiObjectiveAnneal:
    def test_weighted_objectives(self):
        labels = ["x", "y"]
        data = {"x": {"a": 0.1, "b": 0.9}, "y": {"a": 0.5, "b": 0.5}}
        best = multi_objective_anneal(
            labels,
            data,
            [lambda lbl, d: d["a"], lambda lbl, d: d["b"]],
            weights=[0.8, 0.2],
            iterations=300,
        )
        assert best == "x"

    def test_no_energy_fns_raises(self):
        with pytest.raises(ValueError):
            multi_objective_anneal(["a"], {}, [])
