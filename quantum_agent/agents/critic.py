"""Critic agent — evaluates outputs and provides feedback via interference."""

from __future__ import annotations

from typing import Any

from quantum_agent.agents.base import AgentConfig, AgentRole, BaseQuantumAgent
from quantum_agent.core.interference import interference_filter


class CriticAgent(BaseQuantumAgent):
    """Reviews execution results and provides quality scores.

    Uses quantum interference to boost good outcomes and suppress poor
    ones, feeding the result back to the orchestrator for potential
    re-planning.
    """

    def __init__(self, **kwargs: Any) -> None:
        config = kwargs.pop("config", None) or AgentConfig(
            name="critic",
            role=AgentRole.CRITIC,
            max_iterations=3,
        )
        super().__init__(config=config, **kwargs)
        self._execution_results: list[dict[str, Any]] = []
        self._evaluations: list[dict[str, Any]] = []

    async def perceive(
        self, task: str, context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        execution = context.get("execution", {})
        self._execution_results = execution.get("results", [])

        hypotheses = [
            {"label": "accept", "verdict": "The execution met quality standards"},
            {"label": "refine", "verdict": "Minor improvements needed"},
            {"label": "reject", "verdict": "Major issues — re-planning required"},
        ]

        self._record_thought(
            action="perceive",
            observation=f"Reviewing {len(self._execution_results)} execution results",
        )
        return hypotheses

    async def reason(self, task: str, context: dict[str, Any]) -> bool:
        evaluators = [
            self._completeness_evaluator,
            self._consistency_evaluator,
            self._quality_evaluator,
        ]

        new_state = interference_filter(
            self.superposition.state,
            evaluators=evaluators,
            constructive_threshold=0.6,
            destructive_threshold=0.3,
        )
        self.superposition._state = new_state

        evaluations = []
        for result in self._execution_results:
            score = self._score_result(result)
            evaluations.append(
                {
                    "step": result.get("step_text", "unknown"),
                    "score": score,
                    "feedback": self._generate_feedback(result, score),
                }
            )
        self._evaluations = evaluations

        self._record_thought(
            action="evaluate",
            observation=f"Evaluated {len(evaluations)} steps, "
            f"avg score: {sum(e['score'] for e in evaluations) / max(len(evaluations), 1):.2f}",
        )
        return True

    async def act(self, task: str, context: dict[str, Any]) -> dict[str, Any]:
        chosen = self.superposition.collapse_top()

        avg_score = (
            sum(e["score"] for e in self._evaluations) / max(len(self._evaluations), 1)
        )

        review = {
            "verdict": chosen.label,
            "confidence": chosen.probability,
            "average_score": avg_score,
            "evaluations": self._evaluations,
            "recommendation": chosen.data.get("verdict", "No recommendation"),
        }

        self.emit("review", review)

        self._record_thought(
            action="act",
            observation=f"Verdict: {chosen.label} (avg_score={avg_score:.2f})",
        )
        return review

    def _completeness_evaluator(self, h: Any) -> float:
        if not self._execution_results:
            return 0.3
        completed = sum(
            1 for r in self._execution_results if r.get("result", {}).get("status") == "ok"
        )
        ratio = completed / len(self._execution_results)
        if h.label == "accept":
            return ratio
        elif h.label == "refine":
            return 1.0 - abs(ratio - 0.7)
        else:
            return 1.0 - ratio

    def _consistency_evaluator(self, h: Any) -> float:
        if not self._execution_results:
            return 0.5
        statuses = [r.get("result", {}).get("status", "unknown") for r in self._execution_results]
        consistency = statuses.count(statuses[0]) / len(statuses) if statuses else 0.5
        if h.label == "accept":
            return consistency
        elif h.label == "refine":
            return 0.6
        else:
            return 1.0 - consistency

    def _quality_evaluator(self, h: Any) -> float:
        if h.label == "accept":
            return 0.7
        elif h.label == "refine":
            return 0.6
        return 0.3

    def _score_result(self, result: dict[str, Any]) -> float:
        score = 0.5
        if result.get("result", {}).get("status") == "ok":
            score += 0.3
        if result.get("tool_used"):
            score += 0.1
        if result.get("step_text") and len(result["step_text"]) > 5:
            score += 0.1
        return min(score, 1.0)

    def _generate_feedback(self, result: dict[str, Any], score: float) -> str:
        step = result.get("step_text", "unknown step")
        if score >= 0.8:
            return f"'{step}' executed well."
        elif score >= 0.5:
            return f"'{step}' acceptable but could be more thorough."
        return f"'{step}' needs re-execution with a better approach."
