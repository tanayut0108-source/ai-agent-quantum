"""Base quantum agent — the building block for all specialised agents.

Every agent maintains its own QuantumState (superposition of hypotheses),
can use tools, and participates in the entanglement bus.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from quantum_agent.core.entanglement import EntanglementBus
from quantum_agent.core.superposition import Superposition
from quantum_agent.memory.quantum_memory import QuantumMemory
from quantum_agent.tools.registry import ToolRegistry


class AgentRole(StrEnum):
    PLANNER = "planner"
    EXECUTOR = "executor"
    CRITIC = "critic"
    SPECIALIST = "specialist"


@dataclass
class AgentConfig:
    name: str = "quantum-agent"
    role: AgentRole = AgentRole.SPECIALIST
    max_iterations: int = 10
    collapse_strategy: str = "top"  # "top" | "measure" | "threshold"
    collapse_threshold: float = 0.7
    prune_threshold: float = 0.01
    temperature: float = 0.7


@dataclass
class ThoughtStep:
    """One step in the agent's reasoning trace."""

    iteration: int
    action: str
    observation: str
    state_snapshot: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseQuantumAgent(ABC):
    """Abstract base for all quantum agents.

    Lifecycle:
        1. ``perceive`` — receive input, generate hypotheses.
        2. ``reason`` — iteratively evolve the quantum state.
        3. ``act`` — collapse to a decision and execute.
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        bus: EntanglementBus | None = None,
        memory: QuantumMemory | None = None,
        tools: ToolRegistry | None = None,
    ) -> None:
        self.id = str(uuid.uuid4())[:8]
        self.config = config or AgentConfig()
        self.superposition = Superposition()
        self.bus = bus or EntanglementBus()
        self.memory = memory or QuantumMemory()
        self.tools = tools or ToolRegistry()
        self.trace: list[ThoughtStep] = []
        self._iteration = 0

    @property
    def name(self) -> str:
        return f"{self.config.name}-{self.id}"

    async def run(self, task: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Full agent lifecycle: perceive -> reason -> act."""
        context = context or {}

        hypotheses = await self.perceive(task, context)
        self.superposition.branch(hypotheses)

        for i in range(self.config.max_iterations):
            self._iteration = i + 1
            should_stop = await self.reason(task, context)
            if should_stop:
                break

        result = await self.act(task, context)

        self.memory.store(
            key=task[:50],
            value=result,
            relevance=1.0,
        )

        return result

    @abstractmethod
    async def perceive(
        self, task: str, context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Generate initial hypotheses from the task."""

    @abstractmethod
    async def reason(self, task: str, context: dict[str, Any]) -> bool:
        """One reasoning iteration. Return True to stop early."""

    @abstractmethod
    async def act(self, task: str, context: dict[str, Any]) -> dict[str, Any]:
        """Collapse to a decision and produce output."""

    def _record_thought(
        self, action: str, observation: str, metadata: dict[str, Any] | None = None
    ) -> None:
        step = ThoughtStep(
            iteration=self._iteration,
            action=action,
            observation=observation,
            state_snapshot=self.superposition.state.probabilities(),
            metadata=metadata or {},
        )
        self.trace.append(step)

    def emit(self, key: str, value: Any) -> None:
        """Publish to the entanglement bus."""
        self.bus.publish(self.name, key, value)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name}, role={self.config.role.value})"
