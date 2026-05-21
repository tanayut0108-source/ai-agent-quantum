"""Tests for quantum memory."""

from quantum_agent.memory.quantum_memory import QuantumMemory


class TestQuantumMemory:
    def test_store_and_recall(self):
        mem = QuantumMemory()
        mem.store("python api", "FastAPI", tags=["python"])
        results = mem.recall("python api")
        assert len(results) >= 1
        assert results[0]["key"] == "python api"

    def test_capacity_eviction(self):
        mem = QuantumMemory(capacity=3)
        for i in range(5):
            mem.store(f"key_{i}", f"val_{i}")
        assert mem.size() <= 3

    def test_forget(self):
        mem = QuantumMemory()
        mem.store("temp", "data")
        assert mem.forget("temp") is True
        assert mem.forget("nonexistent") is False

    def test_recall_relevance_ordering(self):
        mem = QuantumMemory()
        mem.store("python web framework", "FastAPI", tags=["python", "web"])
        mem.store("database sql", "PostgreSQL", tags=["database"])
        results = mem.recall("python web", top_k=2)
        assert results[0]["key"] == "python web framework"

    def test_snapshot(self):
        mem = QuantumMemory()
        mem.store("a", 1)
        mem.store("b", 2)
        snap = mem.snapshot()
        assert len(snap) == 2

    def test_access_count_increases(self):
        mem = QuantumMemory()
        mem.store("key", "val")
        mem.recall("key")
        mem.recall("key")
        snap = mem.snapshot()
        assert snap[0]["access_count"] == 2
