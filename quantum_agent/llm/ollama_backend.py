"""Ollama backend — use models served by Ollama's local API.

Ollama runs a local HTTP server (default ``http://localhost:11434``)
that manages model downloads, loading, and inference.  This backend
sends requests to that server, making it trivial to switch models.

Usage::

    from quantum_agent.llm import OllamaBackend

    llm = OllamaBackend(model="tinyllama")
    llm.load()                     # pulls the model if needed
    result = llm.generate("Hello!")
    print(result.text)

On Termux::

    # Install and run Ollama, then:
    ollama pull tinyllama
    # In Python:
    llm = OllamaBackend(model="tinyllama")
"""

from __future__ import annotations

import contextlib
import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from quantum_agent.llm.provider import GenerationResult, LLMProvider, ModelInfo

logger = logging.getLogger(__name__)


@dataclass
class OllamaConfig:
    """Configuration for the Ollama backend.

    Attributes
    ----------
    model : model name (e.g. "tinyllama", "qwen2.5:7b", "llama3.1")
    base_url : Ollama API base URL
    timeout : request timeout in seconds
    keep_alive : how long Ollama keeps the model in memory (e.g. "5m")
    extra : additional options passed to Ollama API
    """

    model: str = "tinyllama"
    base_url: str = "http://localhost:11434"
    timeout: int = 120
    keep_alive: str = "5m"
    extra: dict[str, Any] = field(default_factory=dict)


class OllamaBackend(LLMProvider):
    """LLM backend using Ollama's local HTTP API.

    Parameters
    ----------
    model : model name (e.g. "tinyllama", "qwen2.5:7b")
    base_url : Ollama server URL (default: http://localhost:11434)
    config : optional :class:`OllamaConfig` for advanced settings
    """

    def __init__(
        self,
        model: str = "tinyllama",
        base_url: str = "http://localhost:11434",
        config: OllamaConfig | None = None,
    ) -> None:
        self.config = config or OllamaConfig()
        if model != "tinyllama" or config is None:
            self.config.model = model
        if base_url != "http://localhost:11434" or config is None:
            self.config.base_url = base_url.rstrip("/")
        self._loaded = False
        self._model_info: ModelInfo | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Verify the model is available (pull if needed).

        Raises
        ------
        ConnectionError
            If Ollama server is not reachable.
        RuntimeError
            If the model cannot be loaded.
        """
        try:
            tags = self._api_get("/api/tags")
        except urllib.error.URLError as exc:
            raise ConnectionError(
                f"Cannot reach Ollama at {self.config.base_url}. "
                "Is Ollama running? Start with: ollama serve"
            ) from exc

        models = [m.get("name", "") for m in tags.get("models", [])]
        model_name = self.config.model

        found = any(
            model_name == m or model_name == m.split(":")[0]
            for m in models
        )

        if not found:
            logger.info("Model %s not found locally, pulling...", model_name)
            self._api_post("/api/pull", {"name": model_name, "stream": False})
            logger.info("Model %s pulled successfully", model_name)

        self._model_info = ModelInfo(
            name=model_name,
            path=self.config.base_url,
            extra={"backend": "ollama"},
        )
        self._loaded = True
        logger.info("Ollama backend ready: %s", model_name)

    def unload(self) -> None:
        """Release the model from Ollama's memory."""
        if self._loaded:
            with contextlib.suppress(Exception):
                self._api_post("/api/generate", {
                    "model": self.config.model,
                    "keep_alive": 0,
                })
        self._loaded = False
        self._model_info = None
        logger.info("Ollama model unloaded")

    def is_loaded(self) -> bool:
        return self._loaded

    def model_info(self) -> ModelInfo:
        if self._model_info is None:
            return ModelInfo(name="(not loaded)")
        return self._model_info

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> GenerationResult:
        """Generate a completion using Ollama's generate API."""
        self._ensure_loaded()

        payload: dict[str, Any] = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }
        if stop:
            payload["options"]["stop"] = stop
        payload["options"].update(self.config.extra)

        response = self._api_post("/api/generate", payload)
        text = response.get("response", "").strip()
        tokens = response.get("eval_count", 0) + response.get("prompt_eval_count", 0)
        done_reason = response.get("done_reason", "stop")

        return GenerationResult(
            text=text,
            tokens_used=tokens,
            finish_reason=done_reason,
            metadata={
                "model": response.get("model", ""),
                "total_duration": response.get("total_duration", 0),
            },
        )

    def generate_choices(
        self,
        prompt: str,
        n: int = 4,
        max_tokens: int = 256,
        temperature: float = 0.8,
        stop: list[str] | None = None,
    ) -> list[GenerationResult]:
        """Generate multiple completions by calling generate N times."""
        return [
            self.generate(prompt, max_tokens, temperature, stop)
            for _ in range(n)
        ]

    def score(
        self,
        prompt: str,
        text: str,
    ) -> float:
        """Score text by asking the model to evaluate it.

        Since Ollama doesn't expose raw logprobs, we use a simple
        evaluation prompt and parse the result.
        """
        self._ensure_loaded()

        eval_prompt = (
            f"Rate the following text on a scale of 0.0 to 1.0 "
            f"for how well it answers the prompt.\n\n"
            f"Prompt: {prompt}\n"
            f"Text: {text}\n\n"
            f"Respond with ONLY a number between 0.0 and 1.0.\n\nScore:"
        )
        result = self.generate(eval_prompt, max_tokens=10, temperature=0.0)

        import re
        match = re.search(r"(\d+\.?\d*)", result.text)
        if match:
            score = float(match.group(1))
            return max(0.0, min(1.0, score))
        return 0.5

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> GenerationResult:
        """Chat-style generation using Ollama's chat API."""
        self._ensure_loaded()

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }
        payload["options"].update(self.config.extra)

        response = self._api_post("/api/chat", payload)
        msg = response.get("message", {})
        text = msg.get("content", "").strip()
        tokens = response.get("eval_count", 0) + response.get("prompt_eval_count", 0)

        return GenerationResult(
            text=text,
            tokens_used=tokens,
            finish_reason=response.get("done_reason", "stop"),
            metadata={"model": response.get("model", "")},
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError(
                "Ollama backend not loaded. Call .load() first."
            )

    def _api_get(self, path: str) -> dict[str, Any]:
        """Send a GET request to the Ollama API."""
        url = f"{self.config.base_url}{path}"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
            return json.loads(resp.read().decode())

    def _api_post(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        """Send a POST request to the Ollama API."""
        url = f"{self.config.base_url}{path}"
        body = json.dumps(data).encode()
        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
            return json.loads(resp.read().decode())

    def __repr__(self) -> str:
        if self._model_info:
            return (
                f"OllamaBackend({self._model_info.name!r}, "
                f"url={self.config.base_url!r})"
            )
        return (
            f"OllamaBackend(model={self.config.model!r}, "
            f"not loaded)"
        )
