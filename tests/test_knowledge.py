"""Tests for the persistent knowledge store."""

from __future__ import annotations

from pathlib import Path

from quantum_agent.chat.knowledge import KnowledgeStore, Memory

# ---------------------------------------------------------------------------
# Tests: Memory dataclass
# ---------------------------------------------------------------------------


class TestMemory:
    def test_matches_keyword(self) -> None:
        mem = Memory(text="Python is a programming language")
        assert mem.matches("Python")
        assert mem.matches("programming")
        assert not mem.matches("go")  # too short (2 chars, skipped)

    def test_matches_case_insensitive(self) -> None:
        mem = Memory(text="Hello World")
        assert mem.matches("hello")
        assert mem.matches("WORLD")

    def test_matches_short_words_skipped(self) -> None:
        mem = Memory(text="AI is great")
        assert not mem.matches("is")
        assert not mem.matches("AI")  # 2 chars, skipped

    def test_has_timestamp(self) -> None:
        mem = Memory(text="test")
        assert mem.timestamp


# ---------------------------------------------------------------------------
# Tests: KnowledgeStore CRUD
# ---------------------------------------------------------------------------


class TestKnowledgeStore:
    def test_add_and_count(self, tmp_path: Path) -> None:
        store = KnowledgeStore(tmp_path / "mem.json")
        assert store.count == 0
        store.add("First memory")
        assert store.count == 1
        store.add("Second memory")
        assert store.count == 2

    def test_search(self, tmp_path: Path) -> None:
        store = KnowledgeStore(tmp_path / "mem.json")
        store.add("Python is great for scripting")
        store.add("JavaScript runs in browsers")
        store.add("Rust is fast and safe")

        results = store.search("Python scripting")
        assert len(results) >= 1
        assert any("Python" in m.text for m in results)

    def test_search_empty_query_returns_recent(self, tmp_path: Path) -> None:
        store = KnowledgeStore(tmp_path / "mem.json")
        store.add("A")
        store.add("B")
        store.add("C")
        results = store.search("", max_results=2)
        assert len(results) == 2

    def test_search_by_tag(self, tmp_path: Path) -> None:
        store = KnowledgeStore(tmp_path / "mem.json")
        store.add("Mars is red", tags=["science"])
        store.add("Python is fun", tags=["tech"])
        results = store.search_by_tag("science")
        assert len(results) == 1
        assert "Mars" in results[0].text

    def test_remove(self, tmp_path: Path) -> None:
        store = KnowledgeStore(tmp_path / "mem.json")
        store.add("To be removed")
        store.add("To keep")
        removed = store.remove(0)
        assert removed is not None
        assert "removed" in removed.text
        assert store.count == 1

    def test_remove_invalid_index(self, tmp_path: Path) -> None:
        store = KnowledgeStore(tmp_path / "mem.json")
        store.add("Only one")
        assert store.remove(5) is None
        assert store.remove(-1) is None

    def test_clear(self, tmp_path: Path) -> None:
        store = KnowledgeStore(tmp_path / "mem.json")
        store.add("A")
        store.add("B")
        count = store.clear()
        assert count == 2
        assert store.count == 0

    def test_max_memories(self, tmp_path: Path) -> None:
        store = KnowledgeStore(tmp_path / "mem.json", max_memories=3)
        for i in range(5):
            store.add(f"Memory {i}")
        assert store.count == 3
        texts = [m.text for m in store.get_all()]
        assert "Memory 2" in texts
        assert "Memory 0" not in texts

    def test_as_context(self, tmp_path: Path) -> None:
        store = KnowledgeStore(tmp_path / "mem.json")
        store.add("Earth is the third planet")
        store.add("Python was created by Guido")
        context = store.as_context(query="planet")
        assert "Knowledge from memory:" in context
        assert "Earth" in context

    def test_as_context_empty(self, tmp_path: Path) -> None:
        store = KnowledgeStore(tmp_path / "mem.json")
        assert store.as_context() == ""

    def test_repr(self, tmp_path: Path) -> None:
        store = KnowledgeStore(tmp_path / "mem.json")
        assert "KnowledgeStore" in repr(store)
        assert "count=0" in repr(store)


# ---------------------------------------------------------------------------
# Tests: Persistence (load/save)
# ---------------------------------------------------------------------------


class TestKnowledgePersistence:
    def test_save_and_reload(self, tmp_path: Path) -> None:
        path = tmp_path / "mem.json"
        store1 = KnowledgeStore(path)
        store1.add("Persistent fact")
        store1.add("Another fact", tags=["important"])

        store2 = KnowledgeStore(path)
        assert store2.count == 2
        assert store2.get_all()[0].text == "Persistent fact"
        assert store2.get_all()[1].tags == ["important"]

    def test_load_missing_file(self, tmp_path: Path) -> None:
        store = KnowledgeStore(tmp_path / "nonexistent.json")
        assert store.count == 0

    def test_load_corrupt_file(self, tmp_path: Path) -> None:
        path = tmp_path / "corrupt.json"
        path.write_text("not valid json {{{", encoding="utf-8")
        store = KnowledgeStore(path)
        assert store.count == 0


# ---------------------------------------------------------------------------
# Tests: ChatSession integration
# ---------------------------------------------------------------------------


class TestKnowledgeInSession:
    def test_remember_and_list(self, tmp_path: Path) -> None:
        from quantum_agent.chat.session import ChatSession
        from quantum_agent.llm.provider import GenerationResult, LLMProvider, ModelInfo

        class StubProvider(LLMProvider):
            def load(self) -> None: pass
            def unload(self) -> None: pass
            def is_loaded(self) -> bool: return True
            def model_info(self) -> ModelInfo: return ModelInfo(name="stub")
            def generate(self, prompt: str, **kw: object) -> GenerationResult:
                return GenerationResult(text="ok")
            def generate_choices(self, prompt: str, **kw: object) -> list[GenerationResult]:
                return [self.generate(prompt)]
            def score(self, prompt: str, text: str) -> float:
                return 0.5

        store = KnowledgeStore(tmp_path / "mem.json")
        session = ChatSession(StubProvider(), knowledge=store)
        result = session.remember("The sky is blue")
        assert "Remembered" in result
        memories = session.list_memories()
        assert len(memories) == 1
        assert "sky" in memories[0]

    def test_forget(self, tmp_path: Path) -> None:
        store = KnowledgeStore(tmp_path / "mem.json")
        store.add("To forget")
        store.add("To keep")

        from quantum_agent.chat.session import ChatSession
        from quantum_agent.llm.provider import GenerationResult, LLMProvider, ModelInfo

        class StubProvider(LLMProvider):
            def load(self) -> None: pass
            def unload(self) -> None: pass
            def is_loaded(self) -> bool: return True
            def model_info(self) -> ModelInfo: return ModelInfo(name="stub")
            def generate(self, prompt: str, **kw: object) -> GenerationResult:
                return GenerationResult(text="ok")
            def generate_choices(self, prompt: str, **kw: object) -> list[GenerationResult]:
                return [self.generate(prompt)]
            def score(self, prompt: str, text: str) -> float:
                return 0.5

        session = ChatSession(StubProvider(), knowledge=store)
        result = session.forget(0)
        assert "Forgot" in result
        assert store.count == 1

    def test_context_summary_with_memories(self, tmp_path: Path) -> None:
        from quantum_agent.chat.session import ChatSession
        from quantum_agent.llm.provider import GenerationResult, LLMProvider, ModelInfo

        class StubProvider(LLMProvider):
            def load(self) -> None: pass
            def unload(self) -> None: pass
            def is_loaded(self) -> bool: return True
            def model_info(self) -> ModelInfo: return ModelInfo(name="stub")
            def generate(self, prompt: str, **kw: object) -> GenerationResult:
                return GenerationResult(text="ok")
            def generate_choices(self, prompt: str, **kw: object) -> list[GenerationResult]:
                return [self.generate(prompt)]
            def score(self, prompt: str, text: str) -> float:
                return 0.5

        store = KnowledgeStore(tmp_path / "mem.json")
        store.add("fact1")
        store.add("fact2")
        session = ChatSession(StubProvider(), knowledge=store)
        info = session.get_context_summary()
        assert info["memories"] == 2

    def test_no_knowledge_store(self) -> None:
        from quantum_agent.chat.session import ChatSession
        from quantum_agent.llm.provider import GenerationResult, LLMProvider, ModelInfo

        class StubProvider(LLMProvider):
            def load(self) -> None: pass
            def unload(self) -> None: pass
            def is_loaded(self) -> bool: return True
            def model_info(self) -> ModelInfo: return ModelInfo(name="stub")
            def generate(self, prompt: str, **kw: object) -> GenerationResult:
                return GenerationResult(text="ok")
            def generate_choices(self, prompt: str, **kw: object) -> list[GenerationResult]:
                return [self.generate(prompt)]
            def score(self, prompt: str, text: str) -> float:
                return 0.5

        session = ChatSession(StubProvider())
        assert session.remember("test") == "No knowledge store configured."
        assert session.forget(0) == "No knowledge store configured."
        assert session.list_memories() == []
