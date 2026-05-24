"""Tests for the Ollama backend (mocked HTTP calls)."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from quantum_agent.llm.ollama_backend import OllamaBackend, OllamaConfig
from quantum_agent.llm.provider import GenerationResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_urlopen(response_data: dict[str, Any]) -> MagicMock:
    """Create a mock for urllib.request.urlopen that returns JSON."""
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps(response_data).encode()
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=False)
    return mock_response


# ---------------------------------------------------------------------------
# Tests: OllamaConfig
# ---------------------------------------------------------------------------


class TestOllamaConfig:
    def test_defaults(self) -> None:
        config = OllamaConfig()
        assert config.model == "tinyllama"
        assert config.base_url == "http://localhost:11434"
        assert config.timeout == 120

    def test_custom(self) -> None:
        config = OllamaConfig(model="qwen2.5:7b", base_url="http://192.168.1.100:11434")
        assert config.model == "qwen2.5:7b"
        assert config.base_url == "http://192.168.1.100:11434"


# ---------------------------------------------------------------------------
# Tests: OllamaBackend lifecycle
# ---------------------------------------------------------------------------


class TestOllamaBackendLifecycle:
    def test_not_loaded_initially(self) -> None:
        llm = OllamaBackend(model="tinyllama")
        assert not llm.is_loaded()

    def test_model_info_not_loaded(self) -> None:
        llm = OllamaBackend()
        info = llm.model_info()
        assert info.name == "(not loaded)"

    @patch("quantum_agent.llm.ollama_backend.urllib.request.urlopen")
    def test_load_model_exists(self, mock_urlopen_fn: MagicMock) -> None:
        tags_response = _mock_urlopen({
            "models": [{"name": "tinyllama:latest"}]
        })
        mock_urlopen_fn.return_value = tags_response
        llm = OllamaBackend(model="tinyllama")
        llm.load()
        assert llm.is_loaded()
        assert llm.model_info().name == "tinyllama"

    @patch("quantum_agent.llm.ollama_backend.urllib.request.urlopen")
    def test_load_model_pulls(self, mock_urlopen_fn: MagicMock) -> None:
        tags_response = _mock_urlopen({"models": []})
        pull_response = _mock_urlopen({"status": "success"})
        mock_urlopen_fn.side_effect = [tags_response, pull_response]
        llm = OllamaBackend(model="qwen2.5:7b")
        llm.load()
        assert llm.is_loaded()
        assert mock_urlopen_fn.call_count == 2

    @patch("quantum_agent.llm.ollama_backend.urllib.request.urlopen")
    def test_unload(self, mock_urlopen_fn: MagicMock) -> None:
        tags_response = _mock_urlopen({"models": [{"name": "tinyllama:latest"}]})
        unload_response = _mock_urlopen({})
        mock_urlopen_fn.side_effect = [tags_response, unload_response]
        llm = OllamaBackend(model="tinyllama")
        llm.load()
        llm.unload()
        assert not llm.is_loaded()

    def test_generate_not_loaded_raises(self) -> None:
        llm = OllamaBackend()
        with pytest.raises(RuntimeError, match="not loaded"):
            llm.generate("hello")

    def test_repr_not_loaded(self) -> None:
        llm = OllamaBackend(model="tinyllama")
        assert "not loaded" in repr(llm)

    @patch("quantum_agent.llm.ollama_backend.urllib.request.urlopen")
    def test_repr_loaded(self, mock_urlopen_fn: MagicMock) -> None:
        mock_urlopen_fn.return_value = _mock_urlopen({
            "models": [{"name": "tinyllama:latest"}]
        })
        llm = OllamaBackend(model="tinyllama")
        llm.load()
        assert "tinyllama" in repr(llm)


# ---------------------------------------------------------------------------
# Tests: OllamaBackend generation
# ---------------------------------------------------------------------------


class TestOllamaBackendGeneration:
    @patch("quantum_agent.llm.ollama_backend.urllib.request.urlopen")
    def _loaded_backend(self, mock_fn: MagicMock) -> tuple[OllamaBackend, MagicMock]:
        mock_fn.return_value = _mock_urlopen({
            "models": [{"name": "tinyllama:latest"}]
        })
        llm = OllamaBackend(model="tinyllama")
        llm.load()
        return llm, mock_fn

    @patch("quantum_agent.llm.ollama_backend.urllib.request.urlopen")
    def test_generate(self, mock_fn: MagicMock) -> None:
        # Load
        mock_fn.return_value = _mock_urlopen({
            "models": [{"name": "tinyllama:latest"}]
        })
        llm = OllamaBackend(model="tinyllama")
        llm.load()

        # Generate
        mock_fn.return_value = _mock_urlopen({
            "response": "Hello! I am TinyLlama.",
            "eval_count": 8,
            "prompt_eval_count": 5,
            "done_reason": "stop",
            "model": "tinyllama:latest",
        })
        result = llm.generate("Hello")
        assert isinstance(result, GenerationResult)
        assert result.text == "Hello! I am TinyLlama."
        assert result.tokens_used == 13
        assert result.finish_reason == "stop"

    @patch("quantum_agent.llm.ollama_backend.urllib.request.urlopen")
    def test_generate_choices(self, mock_fn: MagicMock) -> None:
        mock_fn.return_value = _mock_urlopen({
            "models": [{"name": "tinyllama:latest"}]
        })
        llm = OllamaBackend(model="tinyllama")
        llm.load()

        mock_fn.return_value = _mock_urlopen({
            "response": "Choice",
            "eval_count": 3,
            "prompt_eval_count": 2,
        })
        results = llm.generate_choices("test", n=3)
        assert len(results) == 3
        assert all(r.text == "Choice" for r in results)

    @patch("quantum_agent.llm.ollama_backend.urllib.request.urlopen")
    def test_chat(self, mock_fn: MagicMock) -> None:
        mock_fn.return_value = _mock_urlopen({
            "models": [{"name": "tinyllama:latest"}]
        })
        llm = OllamaBackend(model="tinyllama")
        llm.load()

        mock_fn.return_value = _mock_urlopen({
            "message": {"role": "assistant", "content": "Hi there!"},
            "eval_count": 5,
            "prompt_eval_count": 3,
            "model": "tinyllama:latest",
        })
        result = llm.chat(
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert result.text == "Hi there!"

    @patch("quantum_agent.llm.ollama_backend.urllib.request.urlopen")
    def test_score(self, mock_fn: MagicMock) -> None:
        mock_fn.return_value = _mock_urlopen({
            "models": [{"name": "tinyllama:latest"}]
        })
        llm = OllamaBackend(model="tinyllama")
        llm.load()

        mock_fn.return_value = _mock_urlopen({
            "response": "0.85",
            "eval_count": 2,
            "prompt_eval_count": 10,
        })
        score = llm.score("prompt", "text")
        assert score == pytest.approx(0.85)

    @patch("quantum_agent.llm.ollama_backend.urllib.request.urlopen")
    def test_score_invalid_returns_default(self, mock_fn: MagicMock) -> None:
        mock_fn.return_value = _mock_urlopen({
            "models": [{"name": "tinyllama:latest"}]
        })
        llm = OllamaBackend(model="tinyllama")
        llm.load()

        mock_fn.return_value = _mock_urlopen({
            "response": "I cannot rate this.",
            "eval_count": 5,
            "prompt_eval_count": 10,
        })
        score = llm.score("prompt", "text")
        assert score == 0.5


# ---------------------------------------------------------------------------
# Tests: CLI backend detection
# ---------------------------------------------------------------------------


class TestCLIBackendDetection:
    def test_auto_detects_gguf(self) -> None:
        from quantum_agent.chat.cli import _detect_backend
        assert _detect_backend("auto", "model.gguf") == "llama"

    def test_auto_detects_path(self) -> None:
        from quantum_agent.chat.cli import _detect_backend
        assert _detect_backend("auto", "/path/to/model.gguf") == "llama"
        assert _detect_backend("auto", r"C:\Users\model.gguf") == "llama"

    def test_auto_detects_ollama(self) -> None:
        from quantum_agent.chat.cli import _detect_backend
        assert _detect_backend("auto", "tinyllama") == "ollama"
        assert _detect_backend("auto", "qwen2.5:7b") == "ollama"

    def test_explicit_backend(self) -> None:
        from quantum_agent.chat.cli import _detect_backend
        assert _detect_backend("ollama", "model.gguf") == "ollama"
        assert _detect_backend("llama", "tinyllama") == "llama"
