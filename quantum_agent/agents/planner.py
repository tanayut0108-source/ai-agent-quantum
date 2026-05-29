"""Planner agent — decomposes tasks into plans using quantum annealing."""

from __future__ import annotations

from typing import Any

from quantum_agent.agents.base import AgentConfig, AgentRole, BaseQuantumAgent
from quantum_agent.reasoning.annealing import quantum_anneal


class PlannerAgent(BaseQuantumAgent):
    """Generates and refines multi-step plans.

    Uses quantum annealing to search the plan space: each hypothesis is a
    candidate plan (sequence of steps), and the energy function penalises
    plans that are incomplete, redundant, or infeasible.
    """

    def __init__(self, **kwargs: Any) -> None:
        config = kwargs.pop("config", None) or AgentConfig(
            name="planner",
            role=AgentRole.PLANNER,
            max_iterations=5,
        )
        super().__init__(config=config, **kwargs)

    async def perceive(
        self, task: str, context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        strategies = self._generate_strategies(task, context)

        self._record_thought(
            action="perceive",
            observation=f"Generated {len(strategies)} candidate plans",
        )
        return strategies

    async def reason(self, task: str, context: dict[str, Any]) -> bool:
        state = self.superposition.state

        def energy_fn(label: str, data: dict[str, Any]) -> float:
            steps = data.get("steps", [])
            completeness = min(len(steps) / max(len(task.split()) // 3, 1), 1.0)
            specificity = sum(1 for s in steps if len(s) > 10) / max(len(steps), 1)
            return 1.0 - (0.6 * completeness + 0.4 * specificity)

        best_label = quantum_anneal(
            labels=[h.label for h in state.hypotheses],
            data_map={h.label: h.data for h in state.hypotheses},
            energy_fn=energy_fn,
            iterations=50,
            initial_temp=1.0,
        )

        self.superposition.state.amplify(best_label, 2.0)
        self.superposition.prune(self.config.prune_threshold)

        self._record_thought(
            action="anneal",
            observation=f"Best plan: {best_label}",
            metadata={"best": best_label, "iteration": self._iteration},
        )

        top = self.superposition.state.top_k(1)
        return bool(top and top[0].probability > self.config.collapse_threshold)

    async def act(self, task: str, context: dict[str, Any]) -> dict[str, Any]:
        chosen = self.superposition.collapse_top()
        plan = {
            "plan_label": chosen.label,
            "steps": chosen.data.get("steps", []),
            "confidence": chosen.probability,
            "reasoning_trace": [
                {"iteration": t.iteration, "action": t.action, "observation": t.observation}
                for t in self.trace
            ],
        }
        self.emit("plan", plan)
        self._record_thought(
            action="act",
            observation=f"Selected plan: {chosen.label} (p={chosen.probability:.4f})",
        )
        return plan

    def _generate_strategies(
        self, task: str, context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Heuristic plan generation — creates multiple candidate strategies."""
        base_steps = [s.strip() for s in task.split(",") if s.strip()]
        if len(base_steps) < 2:
            base_steps = [f"Analyse: {task}", f"Execute: {task}", f"Verify: {task}"]

        strategies: list[dict[str, Any]] = [
            {
                "label": "sequential",
                "steps": base_steps,
                "approach": "Execute steps one by one in order",
            },
            {
                "label": "divide-and-conquer",
                "steps": [f"Decompose: {task}"]
                + [f"Solve sub-problem: {s}" for s in base_steps]
                + ["Merge results"],
                "approach": "Break into independent sub-problems",
            },
            {
                "label": "iterative-refinement",
                "steps": [
                    f"Draft solution for: {task}",
                    "Evaluate draft quality",
                    "Refine weak areas",
                    "Final validation",
                ],
                "approach": "Produce a draft and iteratively improve",
            },
            {
                "label": "parallel-exploration",
                "steps": [
                    f"Explore approach A for: {task}",
                    f"Explore approach B for: {task}",
                    "Compare results",
                    "Select best approach",
                ],
                "approach": "Try multiple approaches simultaneously",
            },
        ]

        memory_hits = self.memory.recall(task, top_k=2)
        if memory_hits:
            strategies.append(
                {
                    "label": "memory-informed",
                    "steps": [
                        "Retrieve relevant past solutions",
                        f"Adapt to: {task}",
                        "Validate adaptation",
                    ],
                    "approach": "Leverage past experience",
                }
            )

        return strategies
