"""Tests for the Superposition manager."""

from quantum_agent.core.qubit import Hypothesis
from quantum_agent.core.superposition import Superposition


class TestSuperposition:
    def test_branch_creates_state(self):
        sup = Superposition()
        sup.branch([{"label": "a"}, {"label": "b"}])
        assert sup.state.size == 2

    def test_evolve(self):
        sup = Superposition()
        sup.branch([{"label": "good"}, {"label": "bad"}])
        sup.evolve(lambda h: 0.9 if h.label == "good" else 0.1)
        probs = sup.state.probabilities()
        assert probs["good"] > probs["bad"]

    def test_prune(self):
        sup = Superposition()
        sup.branch([{"label": "a"}, {"label": "b"}, {"label": "c"}])
        sup.evolve(lambda h: 10.0 if h.label == "a" else 0.001)
        removed = sup.prune(0.05)
        assert removed >= 1

    def test_collapse_returns_hypothesis(self):
        sup = Superposition()
        sup.branch([{"label": "x"}, {"label": "y"}])
        result = sup.collapse()
        assert isinstance(result, Hypothesis)

    def test_collapse_top(self):
        sup = Superposition()
        sup.branch([{"label": "x"}, {"label": "y"}])
        sup.evolve(lambda h: 10.0 if h.label == "y" else 0.1)
        result = sup.collapse_top()
        assert result.label == "y"

    def test_history_recorded(self):
        sup = Superposition()
        sup.branch([{"label": "a"}])
        sup.collapse_top()
        assert len(sup.history) >= 2
