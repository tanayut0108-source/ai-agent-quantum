"""Tests for the LLM integration modules.

Uses a mock LLM provider to test the quantum-LLM bridge without
requiring an actual GGUF model file.
"""

from __future__ import annotations

import json

import pytest

from quantum_agent.core.qubit import Hypothesis, QuantumState
from quantum_agent.core.superposition import Superposition
from quantum_agent.llm.llama_backend import LlamaBackend, LlamaConfig
from quantum_agent.llm.provider import GenerationResult, LLMProvider, ModelInfo
from quantum_agent.llm.quantum_llm import QuantumLLM

# ---------------------------------------------------------------------------
# Mock LLM provider for testing
# ---------------------------------------------------------------------------


class MockLLMProvider(LLMProvider):
    """Deterministic mock provider for testing."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self._loaded = False
        self._responses = responses or ["mock response"]
        self._call_idx = 0

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded

    def model_info(self) -> ModelInfo:
        return ModelInfo(name="mock-model", path="/mock/path", context_length=2048)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> GenerationResult:
        text = self._responses[self._call_idx % len(self._responses)]
        self._call_idx += 1
        return GenerationResult(text=text, tokens_used=len(text.split()))

    def generate_choices(
        self,
        prompt: str,
        n: int = 4,
        max_tokens: int = 256,
        temperature: float = 0.8,
        stop: list[str] | None = None,
    ) -> list[GenerationResult]:
        return [self.generate(prompt, max_tokens, temperature, stop) for _ in range(n)]

    def score(self, prompt: str, text: str) -> float:
        return 0.75

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> GenerationResult:
        text = self._responses[self._call_idx % len(self._responses)]
        self._call_idx += 1
        return GenerationResult(text=text, tokens_used=len(text.split()))


# ---------------------------------------------------------------------------
# Tests: LLMProvider protocol
# ---------------------------------------------------------------------------


class TestLLMProvider:
    def test_mock_provider_lifecycle(self) -> None:
        provider = MockLLMProvider()
        assert not provider.is_loaded()
        provider.load()
        assert provider.is_loaded()
        provider.unload()
        assert not provider.is_loaded()

    def test_mock_provider_model_info(self) -> None:
        provider = MockLLMProvider()
        info = provider.model_info()
        assert info.name == "mock-model"
        assert info.context_length == 2048

    def test_mock_provider_generate(self) -> None:
        provider = MockLLMProvider(["hello world"])
        result = provider.generate("test prompt")
        assert result.text == "hello world"
        assert result.tokens_used == 2

    def test_mock_provider_generate_choices(self) -> None:
        provider = MockLLMProvider(["response"])
        results = provider.generate_choices("test", n=3)
        assert len(results) == 3

    def test_mock_provider_score(self) -> None:
        provider = MockLLMProvider()
        score = provider.score("prompt", "text")
        assert score == 0.75

    def test_provider_repr(self) -> None:
        provider = MockLLMProvider()
        assert "not loaded" in repr(provider)
        provider.load()
        assert "loaded" in repr(provider)


# ---------------------------------------------------------------------------
# Tests: GenerationResult and ModelInfo
# ---------------------------------------------------------------------------


class TestDataClasses:
    def test_generation_result_defaults(self) -> None:
        r = GenerationResult(text="test")
        assert r.tokens_used == 0
        assert r.finish_reason == "stop"
        assert r.logprobs == []
        assert r.metadata == {}

    def test_model_info_defaults(self) -> None:
        info = ModelInfo(name="test")
        assert info.path == ""
        assert info.parameters == 0
        assert info.context_length == 2048

    def test_model_info_frozen(self) -> None:
        info = ModelInfo(name="test")
        with pytest.raises(AttributeError):
            info.name = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Tests: LlamaBackend
# ---------------------------------------------------------------------------


class TestLlamaBackend:
    def test_init_default(self) -> None:
        backend = LlamaBackend()
        assert not backend.is_loaded()
        assert backend.config.model_path == ""

    def test_init_with_path(self) -> None:
        backend = LlamaBackend(model_path="/path/to/model.gguf")
        assert backend.config.model_path == "/path/to/model.gguf"

    def test_init_with_config(self) -> None:
        config = LlamaConfig(
            model_path="/test.gguf",
            n_ctx=4096,
            n_gpu_layers=-1,
        )
        backend = LlamaBackend(config=config)
        assert backend.config.n_ctx == 4096
        assert backend.config.n_gpu_layers == -1

    def test_model_info_not_loaded(self) -> None:
        backend = LlamaBackend()
        info = backend.model_info()
        assert info.name == "(not loaded)"

    def test_ensure_loaded_raises(self) -> None:
        backend = LlamaBackend()
        with pytest.raises(RuntimeError, match="not loaded"):
            backend.generate("test")

    def test_load_missing_file(self) -> None:
        backend = LlamaBackend(model_path="/nonexistent/model.gguf")
        with pytest.raises((ImportError, FileNotFoundError)):
            backend.load()

    def test_detect_quantization(self) -> None:
        assert LlamaBackend._detect_quantization("model-Q4_K_M.gguf") == "Q4_K_M"
        assert LlamaBackend._detect_quantization("model-Q5_K_S.gguf") == "Q5_K_S"
        assert LlamaBackend._detect_quantization("model-F16.gguf") == "F16"
        assert LlamaBackend._detect_quantization("model-q8_0.gguf") == "Q8_0"
        assert LlamaBackend._detect_quantization("random.gguf") == "unknown"

    def test_messages_to_prompt(self) -> None:
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]
        prompt = LlamaBackend._messages_to_prompt(messages)
        assert "[System] You are helpful" in prompt
        assert "[User] Hello" in prompt
        assert prompt.endswith("[Assistant]")

    def test_repr_not_loaded(self) -> None:
        backend = LlamaBackend(model_path="/test.gguf")
        assert "not loaded" in repr(backend)

    def test_unload(self) -> None:
        backend = LlamaBackend()
        backend.unload()
        assert not backend.is_loaded()

    def test_is_chat_model_by_name(self) -> None:
        backend = LlamaBackend(model_path="/tinyllama-chat.gguf")
        backend._model_info = ModelInfo(name="tinyllama-1.1b-chat-v1.0")
        assert backend._is_chat_model is True

    def test_is_not_chat_model(self) -> None:
        backend = LlamaBackend(model_path="/base-model.gguf")
        backend._model_info = ModelInfo(name="llama-7b-base")
        assert backend._is_chat_model is False

    def test_is_chat_model_by_format(self) -> None:
        backend = LlamaBackend(model_path="/base.gguf", chat_format="chatml")
        assert backend._is_chat_model is True


# ---------------------------------------------------------------------------
# Tests: QuantumLLM
# ---------------------------------------------------------------------------


class TestQuantumLLM:
    def _make_hypotheses_response(self) -> str:
        data = [
            {"label": "approach-1", "steps": ["analyse task", "plan solution"]},
            {"label": "approach-2", "steps": ["research", "implement", "test"]},
            {"label": "approach-3", "steps": ["prototype", "iterate"]},
        ]
        return json.dumps(data)

    def test_generate_hypotheses(self) -> None:
        provider = MockLLMProvider([self._make_hypotheses_response()])
        qllm = QuantumLLM(provider)
        hyps = qllm.generate_hypotheses("build a web app", n=3)
        assert len(hyps) == 3
        assert hyps[0]["label"] == "approach-1"
        assert "steps" in hyps[0]

    def test_generate_hypotheses_fallback(self) -> None:
        provider = MockLLMProvider(["this is not valid json at all"])
        qllm = QuantumLLM(provider)
        hyps = qllm.generate_hypotheses("build something", n=3)
        assert len(hyps) == 3
        assert hyps[0]["label"] == "direct"

    def test_generate_hypotheses_as_state(self) -> None:
        provider = MockLLMProvider([self._make_hypotheses_response()])
        qllm = QuantumLLM(provider)
        sup = qllm.generate_hypotheses_as_state("test task", n=3)
        assert isinstance(sup, Superposition)
        assert sup.state.size == 3

    def test_evaluate_hypothesis(self) -> None:
        provider = MockLLMProvider(["0.85"])
        qllm = QuantumLLM(provider)
        h = Hypothesis(label="test", data={"steps": ["step 1", "step 2"]})
        score = qllm.evaluate_hypothesis("task", h)
        assert score == pytest.approx(0.85, abs=0.01)

    def test_evaluate_hypothesis_bad_output(self) -> None:
        provider = MockLLMProvider(["no number here"])
        qllm = QuantumLLM(provider)
        h = Hypothesis(label="test", data={})
        score = qllm.evaluate_hypothesis("task", h)
        assert score == 0.5

    def test_evaluate_hypothesis_clamps(self) -> None:
        provider = MockLLMProvider(["5.0"])
        qllm = QuantumLLM(provider)
        h = Hypothesis(label="test", data={})
        score = qllm.evaluate_hypothesis("task", h)
        assert score == 1.0

    def test_evaluate_all(self) -> None:
        provider = MockLLMProvider(["0.9", "0.5", "0.3"])
        qllm = QuantumLLM(provider)
        state = QuantumState([
            Hypothesis("a", {"steps": ["s1"]}),
            Hypothesis("b", {"steps": ["s2"]}),
            Hypothesis("c", {"steps": ["s3"]}),
        ])
        scores = qllm.evaluate_all("task", state)
        assert len(scores) == 3
        assert all(0.0 <= s <= 1.0 for s in scores.values())

    def test_create_evaluator(self) -> None:
        provider = MockLLMProvider(["0.8"])
        qllm = QuantumLLM(provider)
        evaluator = qllm.create_evaluator("task")
        h = Hypothesis(label="test", data={"steps": ["a"]})
        score = evaluator(h)
        assert 0.0 <= score <= 1.0

    def test_evolve_state(self) -> None:
        provider = MockLLMProvider(["0.9", "0.3"])
        qllm = QuantumLLM(provider)
        sup = Superposition()
        sup.branch([
            {"label": "good", "steps": ["step"]},
            {"label": "bad", "steps": ["step"]},
        ])
        new_state = qllm.evolve_state("task", sup)
        assert new_state.size == 2
        probs = new_state.probabilities()
        assert probs["good"] > probs["bad"]

    def test_chat_reason(self) -> None:
        provider = MockLLMProvider(["Here is my analysis of the task..."])
        qllm = QuantumLLM(provider)
        result = qllm.chat_reason("solve a problem")
        assert "response" in result
        assert result["response"] == "Here is my analysis of the task..."
        assert "tokens_used" in result

    def test_chat_reason_with_context(self) -> None:
        provider = MockLLMProvider(["Considering the context..."])
        qllm = QuantumLLM(provider)
        result = qllm.chat_reason("task", context={"feedback": "needs improvement"})
        assert "response" in result

    def test_score_text(self) -> None:
        provider = MockLLMProvider()
        qllm = QuantumLLM(provider)
        score = qllm.score_text("prompt", "text")
        assert score == 0.75

    def test_parse_hypotheses_valid(self) -> None:
        data = json.dumps([
            {"label": "a", "steps": ["s1"]},
            {"label": "b", "steps": ["s2"]},
        ])
        result = QuantumLLM._parse_hypotheses(data, n=2)
        assert len(result) == 2

    def test_parse_hypotheses_embedded_json(self) -> None:
        text = "Here are hypotheses: [" + json.dumps(
            {"label": "x", "steps": ["s1"]}
        ) + "] done"
        result = QuantumLLM._parse_hypotheses(text, n=1)
        assert len(result) == 1

    def test_parse_hypotheses_invalid(self) -> None:
        result = QuantumLLM._parse_hypotheses("no json here", n=3)
        assert result == []

    def test_parse_score_valid(self) -> None:
        assert QuantumLLM._parse_score("0.85") == pytest.approx(0.85)
        assert QuantumLLM._parse_score("Score: 0.7") == pytest.approx(0.7)

    def test_parse_score_invalid(self) -> None:
        assert QuantumLLM._parse_score("no number") == 0.5

    def test_fallback_hypotheses(self) -> None:
        hyps = QuantumLLM._fallback_hypotheses("test task", 3)
        assert len(hyps) == 3
        assert all("label" in h and "steps" in h for h in hyps)

    def test_repr(self) -> None:
        provider = MockLLMProvider()
        qllm = QuantumLLM(provider)
        assert "QuantumLLM" in repr(qllm)
