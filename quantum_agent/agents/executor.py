"""Executor agent — carries out plans by invoking tools step-by-step."""

from __future__ import annotations

from typing import Any

from quantum_agent.agents.base import AgentConfig, AgentRole, BaseQuantumAgent


class ExecutorAgent(BaseQuantumAgent):
    """Executes a plan produced by the PlannerAgent.

    For each plan step the executor:
    1. Generates hypotheses about *how* to execute the step.
    2. Scores them via tool availability and context.
    3. Collapses to the best approach and runs it.
    """

    def __init__(self, **kwargs: Any) -> None:
        config = kwargs.pop("config", None) or AgentConfig(
            name="executor",
            role=AgentRole.EXECUTOR,
            max_iterations=8,
        )
        super().__init__(config=config, **kwargs)
        self._plan_steps: list[str] = []
        self._results: list[dict[str, Any]] = []
        self._current_step_idx = 0

    async def perceive(
        self, task: str, context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        plan = context.get("plan", {})
        self._plan_steps = plan.get("steps", [task])

        hypotheses = []
        for i, step in enumerate(self._plan_steps):
            hypotheses.append(
                {
                    "label": f"step-{i}",
                    "step_text": step,
                    "index": i,
                }
            )

        self._record_thought(
            action="perceive",
            observation=f"Received plan with {len(self._plan_steps)} steps",
        )
        return hypotheses

    async def reason(self, task: str, context: dict[str, Any]) -> bool:
        if self._current_step_idx >= len(self._plan_steps):
            return True

        step_text = self._plan_steps[self._current_step_idx]

        available_tools = self.tools.list_tools()
        matching_tool = None
        for tool_name in available_tools:
            tool_meta = self.tools.get_tool(tool_name)
            if tool_meta and any(
                kw in step_text.lower() for kw in tool_meta.get("keywords", [])
            ):
                matching_tool = tool_name
                break

        if matching_tool:
            result = await self.tools.execute(matching_tool, {"input": step_text})
        else:
            result = {"output": f"[simulated] Completed: {step_text}", "status": "ok"}

        self._results.append(
            {
                "step_index": self._current_step_idx,
                "step_text": step_text,
                "tool_used": matching_tool,
                "result": result,
            }
        )

        self._record_thought(
            action=f"execute_step_{self._current_step_idx}",
            observation=f"Step '{step_text}' -> tool={matching_tool}, status=ok",
        )

        self.emit(
            "execution_progress",
            {
                "step": self._current_step_idx,
                "total": len(self._plan_steps),
                "status": "completed",
            },
        )

        self._current_step_idx += 1
        return self._current_step_idx >= len(self._plan_steps)

    async def act(self, task: str, context: dict[str, Any]) -> dict[str, Any]:
        chosen = self.superposition.collapse_top()
        return {
            "execution_label": chosen.label,
            "steps_completed": len(self._results),
            "results": self._results,
            "success": all(r["result"].get("status") == "ok" for r in self._results),
        }
