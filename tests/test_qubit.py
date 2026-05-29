"""Tests for quantum state and hypothesis primitives."""

import math

import pytest

from quantum_agent.core.qubit import Hypothesis, QuantumState


class TestHypothesis:
    def test_default_amplitude(self):
        h = Hypothesis("test")
        assert h.amplitude == complex(1.0, 0.0)
        assert h.probability == 1.0

    def test_probability_from_amplitude(self):
        h = Hypothesis("x", amplitude=complex(0.6, 0.8))
        assert abs(h.probability - 1.0) < 1e-9

    def test_repr(self):
        h = Hypothesis("foo")
        assert "foo" in repr(h)


class TestQuantumState:
    def test_normalisation(self):
        state = QuantumState([Hypothesis("a"), Hypothesis("b"), Hypothesis("c")])
        total = sum(h.probability for h in state.hypotheses)
        assert abs(total - 1.0) < 1e-9

    def test_add_maintains_normalisation(self):
        state = QuantumState([Hypothesis("a"), Hypothesis("b")])
        state.add(Hypothesis("c"))
        total = sum(h.probability for h in state.hypotheses)
        assert abs(total - 1.0) < 1e-9

    def test_remove(self):
        state = QuantumState([Hypothesis("a"), Hypothesis("b"), Hypothesis("c")])
        state.remove("b")
        assert state.size == 2
        labels = {h.label for h in state.hypotheses}
        assert "b" not in labels

    def test_measure_returns_single(self):
        state = QuantumState([Hypothesis("a"), Hypothesis("b")])
        result = state.measure()
        assert result.label in ("a", "b")
        assert state.size == 1

    def test_measure_empty_raises(self):
        state = QuantumState()
        with pytest.raises(ValueError):
            state.measure()

    def test_amplify(self):
        state = QuantumState([Hypothesis("a"), Hypothesis("b")])
        prob_before = state.probabilities()["a"]
        state.amplify("a", 3.0)
        assert state.probabilities()["a"] > prob_before

    def test_dampen(self):
        state = QuantumState([Hypothesis("a"), Hypothesis("b")])
        prob_before = state.probabilities()["a"]
        state.dampen("a", 0.1)
        assert state.probabilities()["a"] < prob_before

    def test_top_k(self):
        state = QuantumState([Hypothesis("a"), Hypothesis("b"), Hypothesis("c")])
        state.amplify("c", 5.0)
        top = state.top_k(1)
        assert top[0].label == "c"

    def test_entropy_uniform(self):
        state = QuantumState([Hypothesis("a"), Hypothesis("b")])
        assert state.entropy() > 0.9

    def test_entropy_collapsed(self):
        state = QuantumState([Hypothesis("a")])
        assert state.entropy() == 0.0

    def test_clone(self):
        state = QuantumState([Hypothesis("a"), Hypothesis("b")])
        cloned = state.clone()
        cloned.remove("a")
        assert state.size == 2
        assert cloned.size == 1

    def test_phase_rotate(self):
        state = QuantumState([Hypothesis("a"), Hypothesis("b")])
        state.phase_rotate("a", math.pi / 4)
        total = sum(h.probability for h in state.hypotheses)
        assert abs(total - 1.0) < 1e-9
