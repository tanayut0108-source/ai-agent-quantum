"""Quantum Orchestrator — coordinates Planner, Executor, and Critic agents.

The orchestrator implements a Plan-Execute-Critique loop:
1. The **Planner** generates a plan in superposition and collapses to the best.
2. The **Executor** carries out the plan step by step.
3. The **Critic** evaluates the results via quantum interference.
4. If the critic rejects, the loop re-plans with updated context.

All agents share an EntanglementBus for implicit coordination.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from quantum_agent.agents.critic import CriticAgent
from quantum_agent.agents.executor import ExecutorAgent
from quantum_agent.agents.planner import PlannerAgent
from quantum_agent.core.entanglement import EntanglementBus
from quantum_agent.memory.quantum_memory import QuantumMemory
from quantum_agent.tools.registry import ToolRegistry


@dataclass
class OrchestratorConfig:
    max_cycles: int = 3
    auto_retry: bool = True
    verbose: bool = True


@dataclass
class CycleResult:
    cycle: int
    plan: dict[str, Any]
    execution: dict[str, Any]
    review: dict[str, Any]
    duration_seconds: float


class QuantumOrchestrator:
    """Top-level coordinator for quantum multi-agent workflows."""

    def __init__(
        self,
        config: OrchestratorConfig | None = None,
        tools: ToolRegistry | None = None,
    ) -> None:
        self.config = config or OrchestratorConfig()
        self.bus = EntanglementBus()
        self.memory = QuantumMemory()
        self.tools = tools or ToolRegistry()

        self.planner = PlannerAgent(bus=self.bus, memory=self.memory, tools=self.tools)
        self.executor = ExecutorAgent(bus=self.bus, memory=self.memory, tools=self.tools)
        self.critic = CriticAgent(bus=self.bus, memory=self.memory, tools=self.tools)

        self._cycles: list[CycleResult] = []

    async def run(self, task: str) -> dict[str, Any]:
        """Execute the full Plan-Execute-Critique loop."""
        context: dict[str, Any] = {"original_task": task}
        final_result: dict[str, Any] = {}

        for cycle_idx in range(self.config.max_cycles):
            start = time.monotonic()

            plan = await self.planner.run(task, context)
            context["plan"] = plan

            self.executor = ExecutorAgent(
                bus=self.bus, memory=self.memory, tools=self.tools
            )
            execution = await self.executor.run(task, context)
            context["execution"] = execution

            self.critic = CriticAgent(
                bus=self.bus, memory=self.memory, tools=self.tools
            )
            review = await self.critic.run(task, context)
            context["review"] = review

            duration = time.monotonic() - start

            cycle_result = CycleResult(
                cycle=cycle_idx + 1,
                plan=plan,
                execution=execution,
                review=review,
                duration_seconds=round(duration, 3),
            )
            self._cycles.append(cycle_result)

            verdict = review.get("verdict", "reject")
            if verdict == "accept":
                final_result = self._build_final_result(task, "accepted")
                break
            elif verdict == "refine" and self.config.auto_retry:
                context["feedback"] = review.get("evaluations", [])
                self.planner = PlannerAgent(
                    bus=self.bus, memory=self.memory, tools=self.tools
                )
                continue
            else:
                if not self.config.auto_retry:
                    final_result = self._build_final_result(task, verdict)
                    break
        else:
            final_result = self._build_final_result(task, "max_cycles_reached")

        self.memory.store(
            key=f"task:{task[:50]}",
            value=final_result,
            relevance=0.9,
            tags=["orchestrator", "result"],
        )

        return final_result

    def _build_final_result(self, task: str, status: str) -> dict[str, Any]:
        last_cycle = self._cycles[-1] if self._cycles else None
        return {
            "task": task,
            "status": status,
            "total_cycles": len(self._cycles),
            "cycles": [
                {
                    "cycle": c.cycle,
                    "plan_label": c.plan.get("plan_label"),
                    "steps_completed": c.execution.get("steps_completed"),
                    "verdict": c.review.get("verdict"),
                    "avg_score": c.review.get("average_score"),
                    "duration": c.duration_seconds,
                }
                for c in self._cycles
            ],
            "final_plan": last_cycle.plan if last_cycle else {},
            "final_execution": last_cycle.execution if last_cycle else {},
            "final_review": last_cycle.review if last_cycle else {},
            "entanglement_state": self.bus.snapshot(),
            "memory_size": self.memory.size(),
        }

    @property
    def cycles(self) -> list[CycleResult]:
        return list(self._cycles)

    def __repr__(self) -> str:
        return (
            f"QuantumOrchestrator(cycles={len(self._cycles)}, "
            f"memory={self.memory.size()})"
        )
