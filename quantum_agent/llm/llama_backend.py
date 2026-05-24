"""GGUF / llama.cpp backend using ``llama-cpp-python``.

This module wraps the ``llama_cpp`` library to load quantised GGUF models
and expose them through the :class:`LLMProvider` protocol.  It is designed
to run on resource-constrained devices (e.g. Termux on Android) where
full-precision models are impractical.

Usage::

    from quantum_agent.llm import LlamaBackend

    llm = LlamaBackend(model_path="/path/to/model.gguf")
    llm.load()
    result = llm.generate("Explain quantum computing in one sentence.")
    print(result.text)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from quantum_agent.llm.provider import GenerationResult, LLMProvider, ModelInfo

logger = logging.getLogger(__name__)


@dataclass
class LlamaConfig:
    """Configuration for the llama.cpp backend.

    Attributes
    ----------
    model_path : path to the ``.gguf`` file
    n_ctx : context window size (tokens)
    n_threads : CPU threads (0 = auto-detect)
    n_gpu_layers : layers to offload to GPU (-1 = all, 0 = CPU only)
    n_batch : batch size for prompt evaluation
    verbose : whether to print llama.cpp logs
    seed : random seed (-1 = random)
    rope_freq_base : RoPE frequency base (0 = model default)
    rope_freq_scale : RoPE frequency scale (0 = model default)
    extra : additional kwargs passed to ``Llama()``
    """

    model_path: str = ""
    n_ctx: int = 2048
    n_threads: int = 0
    n_gpu_layers: int = 0
    n_batch: int = 512
    verbose: bool = False
    seed: int = -1
    rope_freq_base: float = 0.0
    rope_freq_scale: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


class LlamaBackend(LLMProvider):
    """GGUF model backend via ``llama-cpp-python``.

    Parameters
    ----------
    model_path : path to the ``.gguf`` file (or set via config)
    config : optional :class:`LlamaConfig` for advanced settings
    """

    def __init__(
        self,
        model_path: str = "",
        config: LlamaConfig | None = None,
    ) -> None:
        self.config = config or LlamaConfig()
        if model_path:
            self.config.model_path = model_path
        self._model: Any = None
        self._model_info: ModelInfo | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the GGUF model into memory.

        Raises
        ------
        ImportError
            If ``llama-cpp-python`` is not installed.
        FileNotFoundError
            If the model file does not exist.
        """
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ImportError(
                "llama-cpp-python is required for the GGUF backend. "
                "Install it with: pip install llama-cpp-python"
            ) from exc

        path = Path(self.config.model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        kwargs: dict[str, Any] = {
            "model_path": str(path),
            "n_ctx": self.config.n_ctx,
            "n_batch": self.config.n_batch,
            "n_gpu_layers": self.config.n_gpu_layers,
            "verbose": self.config.verbose,
            "seed": self.config.seed,
        }
        if self.config.n_threads > 0:
            kwargs["n_threads"] = self.config.n_threads
        if self.config.rope_freq_base > 0:
            kwargs["rope_freq_base"] = self.config.rope_freq_base
        if self.config.rope_freq_scale > 0:
            kwargs["rope_freq_scale"] = self.config.rope_freq_scale
        kwargs.update(self.config.extra)

        logger.info("Loading GGUF model: %s", path.name)
        self._model = Llama(**kwargs)

        self._model_info = ModelInfo(
            name=path.stem,
            path=str(path),
            context_length=self.config.n_ctx,
            quantization=self._detect_quantization(path.name),
        )
        logger.info("Model loaded: %s", self._model_info.name)

    def unload(self) -> None:
        """Release model resources."""
        self._model = None
        self._model_info = None
        logger.info("Model unloaded")

    def is_loaded(self) -> bool:
        return self._model is not None

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
        """Generate a single text completion."""
        self._ensure_loaded()

        response = self._model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop or [],
            echo=False,
        )

        choice = response["choices"][0]
        text = choice["text"].strip()
        usage = response.get("usage", {})

        return GenerationResult(
            text=text,
            tokens_used=usage.get("total_tokens", 0),
            finish_reason=choice.get("finish_reason", "stop"),
            metadata={"model": response.get("model", "")},
        )

    def generate_choices(
        self,
        prompt: str,
        n: int = 4,
        max_tokens: int = 256,
        temperature: float = 0.8,
        stop: list[str] | None = None,
    ) -> list[GenerationResult]:
        """Generate multiple independent completions."""
        results: list[GenerationResult] = []
        for _ in range(n):
            result = self.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
            )
            results.append(result)
        return results

    def score(
        self,
        prompt: str,
        text: str,
    ) -> float:
        """Score text quality by measuring per-token log-probability.

        Returns a normalised score in [0, 1] where higher is better.
        """
        self._ensure_loaded()

        full_text = prompt + text
        response = self._model(
            full_text,
            max_tokens=1,
            temperature=0.0,
            logprobs=1,
            echo=True,
        )

        logprobs_data = response["choices"][0].get("logprobs")
        if not logprobs_data or not logprobs_data.get("token_logprobs"):
            return 0.5

        token_lps = logprobs_data["token_logprobs"]
        valid_lps = [lp for lp in token_lps if lp is not None]
        if not valid_lps:
            return 0.5

        avg_lp = sum(valid_lps) / len(valid_lps)
        score = 1.0 / (1.0 + math.exp(-avg_lp - 2.0))
        return max(0.0, min(1.0, score))

    # ------------------------------------------------------------------
    # Chat-style generation (convenience)
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> GenerationResult:
        """Chat-style generation using a message list.

        Messages should follow the format:
        ``[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]``

        Falls back to prompt-style if the model does not support chat.
        """
        self._ensure_loaded()

        if hasattr(self._model, "create_chat_completion"):
            response = self._model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            choice = response["choices"][0]
            text = choice["message"]["content"].strip()
            usage = response.get("usage", {})
            return GenerationResult(
                text=text,
                tokens_used=usage.get("total_tokens", 0),
                finish_reason=choice.get("finish_reason", "stop"),
            )

        prompt = self._messages_to_prompt(messages)
        return self.generate(prompt, max_tokens=max_tokens, temperature=temperature)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is None:
            raise RuntimeError(
                "Model not loaded. Call .load() first or check model_path."
            )

    @staticmethod
    def _detect_quantization(filename: str) -> str:
        """Guess quantization from the filename (e.g. Q4_K_M, Q5_K_S)."""
        upper = filename.upper()
        for q in [
            "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L",
            "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M",
            "Q5_0", "Q5_1", "Q5_K_S", "Q5_K_M",
            "Q6_K", "Q8_0", "F16", "F32",
        ]:
            if q in upper:
                return q
        return "unknown"

    @staticmethod
    def _messages_to_prompt(messages: list[dict[str, str]]) -> str:
        """Convert chat messages to a plain prompt string."""
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"[System] {content}")
            elif role == "assistant":
                parts.append(f"[Assistant] {content}")
            else:
                parts.append(f"[User] {content}")
        parts.append("[Assistant]")
        return "\n".join(parts)

    def __repr__(self) -> str:
        if self._model_info:
            return (
                f"LlamaBackend({self._model_info.name!r}, "
                f"q={self._model_info.quantization}, "
                f"ctx={self._model_info.context_length})"
            )
        return f"LlamaBackend(path={self.config.model_path!r}, not loaded)"
