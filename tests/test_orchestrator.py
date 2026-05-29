"""Tests for the quantum orchestrator."""

import asyncio

from quantum_agent.orchestrator.quantum_orchestrator import (
    OrchestratorConfig,
    QuantumOrchestrator,
)


class TestQuantumOrchestrator:
    def test_basic_run(self):
        config = OrchestratorConfig(max_cycles=1, auto_retry=False)
        orch = QuantumOrchestrator(config=config)
        result = asyncio.run(orch.run("Build a simple calculator"))
        assert "status" in result
        assert "total_cycles" in result
        assert result["total_cycles"] >= 1

    def test_multi_cycle(self):
        config = OrchestratorConfig(max_cycles=2, auto_retry=True)
        orch = QuantumOrchestrator(config=config)
        result = asyncio.run(orch.run("Design a web scraper"))
        assert result["total_cycles"] >= 1

    def test_memory_populated(self):
        config = OrchestratorConfig(max_cycles=1, auto_retry=False)
        orch = QuantumOrchestrator(config=config)
        asyncio.run(orch.run("Test task"))
        assert orch.memory.size() > 0

    def test_entanglement_bus_has_state(self):
        config = OrchestratorConfig(max_cycles=1, auto_retry=False)
        orch = QuantumOrchestrator(config=config)
        asyncio.run(orch.run("Orchestration test"))
        snap = orch.bus.snapshot()
        assert len(snap) > 0

    def test_cycles_recorded(self):
        config = OrchestratorConfig(max_cycles=1, auto_retry=False)
        orch = QuantumOrchestrator(config=config)
        asyncio.run(orch.run("Record test"))
        assert len(orch.cycles) == 1
