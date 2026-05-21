"""Tests for quantum interference."""

from quantum_agent.core.interference import interference_filter, pairwise_interference
from quantum_agent.core.qubit import Hypothesis, QuantumState


class TestInterferenceFilter:
    def test_constructive_boost(self):
        state = QuantumState([Hypothesis("good"), Hypothesis("bad")])
        evaluators = [lambda h: 0.9 if h.label == "good" else 0.1]
        result = interference_filter(state, evaluators)
        probs = result.probabilities()
        assert probs["good"] > probs["bad"]

    def test_destructive_dampen(self):
        state = QuantumState([Hypothesis("a"), Hypothesis("b")])
        evaluators = [lambda h: 0.1 if h.label == "a" else 0.9]
        result = interference_filter(state, evaluators)
        probs = result.probabilities()
        assert probs["b"] > probs["a"]

    def test_original_not_mutated(self):
        state = QuantumState([Hypothesis("x"), Hypothesis("y")])
        original_probs = state.probabilities()
        interference_filter(state, [lambda h: 0.5])
        assert state.probabilities() == original_probs


class TestPairwiseInterference:
    def test_similar_hypotheses_merge(self):
        state = QuantumState([
            Hypothesis("plan_v1"),
            Hypothesis("plan_v2"),
            Hypothesis("other"),
        ])
        state.amplify("plan_v1", 2.0)

        def sim(h1, h2):
            if "plan" in h1.label and "plan" in h2.label:
                return 0.9
            return 0.1

        result = pairwise_interference(state, sim, threshold=0.8)
        probs = result.probabilities()
        assert probs["plan_v1"] > probs["plan_v2"]
