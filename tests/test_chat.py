"""Tests for the chat/conversation module."""

from __future__ import annotations

from quantum_agent.chat.session import ChatMessage, ChatSession
from quantum_agent.llm.provider import GenerationResult, LLMProvider, ModelInfo

# ---------------------------------------------------------------------------
# Mock provider
# ---------------------------------------------------------------------------


class MockChatProvider(LLMProvider):
    """Mock LLM that echoes back the last user message."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self._loaded = True
        self._responses = responses or []
        self._idx = 0

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded

    def model_info(self) -> ModelInfo:
        return ModelInfo(name="mock-chat-model")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> GenerationResult:
        text = self._next_response(f"Generated: {prompt[:50]}")
        return GenerationResult(text=text, tokens_used=10)

    def generate_choices(
        self,
        prompt: str,
        n: int = 4,
        max_tokens: int = 256,
        temperature: float = 0.8,
        stop: list[str] | None = None,
    ) -> list[GenerationResult]:
        return [self.generate(prompt) for _ in range(n)]

    def score(self, prompt: str, text: str) -> float:
        return 0.7

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> GenerationResult:
        last_user = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                last_user = msg["content"]
                break
        text = self._next_response(f"Reply to: {last_user[:50]}")
        return GenerationResult(text=text, tokens_used=15)

    def _next_response(self, default: str) -> str:
        if self._idx < len(self._responses):
            text = self._responses[self._idx]
            self._idx += 1
            return text
        return default


# ---------------------------------------------------------------------------
# Tests: ChatMessage
# ---------------------------------------------------------------------------


class TestChatMessage:
    def test_create(self) -> None:
        msg = ChatMessage(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"
        assert msg.timestamp  # non-empty

    def test_to_dict(self) -> None:
        msg = ChatMessage(role="assistant", content="hi there")
        d = msg.to_dict()
        assert d == {"role": "assistant", "content": "hi there"}

    def test_metadata(self) -> None:
        msg = ChatMessage(role="user", content="x", metadata={"turn": 1})
        assert msg.metadata["turn"] == 1


# ---------------------------------------------------------------------------
# Tests: ChatSession
# ---------------------------------------------------------------------------


class TestChatSession:
    def test_create_session(self) -> None:
        provider = MockChatProvider()
        session = ChatSession(provider)
        assert session.turn_count == 0
        assert len(session.history) == 0

    def test_send_message(self) -> None:
        provider = MockChatProvider(["Hello! How can I help?"])
        session = ChatSession(provider)
        reply = session.send("Hi")
        assert reply.role == "assistant"
        assert reply.content == "Hello! How can I help?"
        assert session.turn_count == 1
        assert len(session.history) == 2  # user + assistant

    def test_multi_turn(self) -> None:
        provider = MockChatProvider(["First reply", "Second reply"])
        session = ChatSession(provider)
        r1 = session.send("Hello")
        r2 = session.send("Follow up")
        assert r1.content == "First reply"
        assert r2.content == "Second reply"
        assert session.turn_count == 2
        assert len(session.history) == 4  # 2 user + 2 assistant

    def test_history_contains_messages(self) -> None:
        provider = MockChatProvider(["Reply"])
        session = ChatSession(provider)
        session.send("Test message")
        history = session.history
        assert history[0].role == "user"
        assert history[0].content == "Test message"
        assert history[1].role == "assistant"
        assert history[1].content == "Reply"

    def test_reset(self) -> None:
        provider = MockChatProvider(["Reply"])
        session = ChatSession(provider)
        session.send("Hi")
        assert session.turn_count == 1
        session.reset()
        assert session.turn_count == 0
        assert len(session.history) == 0

    def test_custom_system_prompt(self) -> None:
        provider = MockChatProvider()
        session = ChatSession(provider, system_prompt="You are a pirate.")
        session.send("Hello")
        messages = session._build_messages()
        assert messages[0]["content"] == "You are a pirate."

    def test_set_system_prompt(self) -> None:
        provider = MockChatProvider()
        session = ChatSession(provider)
        session.set_system_prompt("New prompt")
        session.send("test")
        messages = session._build_messages()
        assert messages[0]["content"] == "New prompt"

    def test_max_history_trim(self) -> None:
        provider = MockChatProvider()
        session = ChatSession(provider, max_history=4)
        session.send("msg 1")
        session.send("msg 2")
        session.send("msg 3")
        # 6 messages total but max_history=4, oldest trimmed
        assert len(session.history) == 4

    def test_context_summary(self) -> None:
        provider = MockChatProvider()
        session = ChatSession(provider)
        session.send("Hi")
        info = session.get_context_summary()
        assert info["turn_count"] == 1
        assert info["message_count"] == 2
        assert info["model"] == "mock-chat-model"
        assert info["quantum_mode"] is False

    def test_custom_temperature_and_tokens(self) -> None:
        provider = MockChatProvider(["OK"])
        session = ChatSession(provider, max_tokens=100, temperature=0.3)
        reply = session.send("test", max_tokens=200, temperature=0.9)
        assert reply.content == "OK"

    def test_quantum_mode(self) -> None:
        responses = [
            # generate_hypotheses calls generate
            "1. Approach A\n- step one\n- step two\n\n"
            "2. Approach B\n- step three\n- step four",
            # synthesis reply
            "Here is my quantum-reasoned answer.",
        ]
        provider = MockChatProvider(responses)
        session = ChatSession(provider, quantum_mode=True)
        reply = session.send("solve a problem")
        assert reply.role == "assistant"
        assert len(reply.content) > 0
        assert session.turn_count == 1

    def test_quantum_mode_toggle(self) -> None:
        provider = MockChatProvider()
        session = ChatSession(provider, quantum_mode=False)
        assert session.quantum_mode is False
        session.quantum_mode = True
        assert session.quantum_mode is True

    def test_repr(self) -> None:
        provider = MockChatProvider()
        session = ChatSession(provider)
        assert "ChatSession" in repr(session)
        assert "turns=0" in repr(session)

    def test_fallback_to_generate(self) -> None:
        """Test session works when provider has no chat method."""

        class NoChatProvider(LLMProvider):
            def __init__(self) -> None:
                self._loaded = True

            def load(self) -> None:
                pass

            def unload(self) -> None:
                pass

            def is_loaded(self) -> bool:
                return True

            def model_info(self) -> ModelInfo:
                return ModelInfo(name="no-chat-model")

            def generate(
                self,
                prompt: str,
                max_tokens: int = 512,
                temperature: float = 0.7,
                stop: list[str] | None = None,
            ) -> GenerationResult:
                return GenerationResult(text="Fallback reply")

            def generate_choices(
                self,
                prompt: str,
                n: int = 4,
                max_tokens: int = 256,
                temperature: float = 0.8,
                stop: list[str] | None = None,
            ) -> list[GenerationResult]:
                return [self.generate(prompt)]

            def score(self, prompt: str, text: str) -> float:
                return 0.5

        provider = NoChatProvider()
        session = ChatSession(provider)
        reply = session.send("Hi")
        assert reply.content == "Fallback reply"


# ---------------------------------------------------------------------------
# Tests: CLI argument parsing
# ---------------------------------------------------------------------------


class TestCLI:
    def test_cli_import(self) -> None:
        from quantum_agent.chat.cli import main
        assert callable(main)
