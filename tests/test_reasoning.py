"""Tests for quantum reasoning engine."""

from quantum_agent.reasoning.quantum_reasoning import QuantumReasoner


class TestQuantumReasoner:
    def test_basic_reasoning(self):
        reasoner = QuantumReasoner(max_rounds=3)
        hypotheses = [
            {"label": "option_A"},
            {"label": "option_B"},
            {"label": "option_C"},
        ]
        evaluators = [lambda h: 0.9 if "A" in h.label else 0.3]
        result = reasoner.reason(hypotheses, evaluators)
        assert result.answer == "option_A"
        assert result.confidence > 0

    def test_entropy_decreases(self):
        reasoner = QuantumReasoner(max_rounds=5)
        hypotheses = [{"label": f"h_{i}"} for i in range(5)]
        evaluators = [lambda h: 0.95 if h.label == "h_0" else 0.1]
        result = reasoner.reason(hypotheses, evaluators)
        assert result.entropy_trace[-1] <= result.entropy_trace[0]

    def test_multiple_evaluators(self):
        reasoner = QuantumReasoner()
        hypotheses = [{"label": "fast"}, {"label": "accurate"}, {"label": "balanced"}]
        evaluators = [
            lambda h: 0.9 if h.label == "fast" else 0.5,
            lambda h: 0.9 if h.label == "accurate" else 0.5,
            lambda h: 0.8 if h.label == "balanced" else 0.4,
        ]
        result = reasoner.reason(hypotheses, evaluators)
        assert result.answer in ("fast", "accurate", "balanced")

    def test_reasoning_steps_recorded(self):
        reasoner = QuantumReasoner(max_rounds=2)
        result = reasoner.reason(
            [{"label": "a"}, {"label": "b"}],
            [lambda h: 0.5],
        )
        assert len(result.reasoning_steps) > 0
