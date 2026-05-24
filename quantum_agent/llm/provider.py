"""Abstract LLM provider protocol.

Any LLM backend (llama.cpp, OpenAI, Ollama, etc.) can implement this
protocol and be used interchangeably by the quantum agent framework.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelInfo:
    """Metadata about the loaded model."""

    name: str
    path: str = ""
    parameters: int = 0
    quantization: str = ""
    context_length: int = 2048
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    """Output from a single LLM generation call."""

    text: str
    tokens_used: int = 0
    finish_reason: str = "stop"
    logprobs: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMProvider(ABC):
    """Abstract base for LLM backends.

    Implementations must provide:
    - ``load()`` — initialise the model
    - ``generate()`` — produce text from a prompt
    - ``generate_choices()`` — produce multiple completions
    - ``score()`` — score/evaluate a text given a prompt
    """

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory."""

    @abstractmethod
    def unload(self) -> None:
        """Release model resources."""

    @abstractmethod
    def is_loaded(self) -> bool:
        """Return True if a model is currently loaded."""

    @abstractmethod
    def model_info(self) -> ModelInfo:
        """Return metadata about the loaded model."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> GenerationResult:
        """Generate a single completion."""

    @abstractmethod
    def generate_choices(
        self,
        prompt: str,
        n: int = 4,
        max_tokens: int = 256,
        temperature: float = 0.8,
        stop: list[str] | None = None,
    ) -> list[GenerationResult]:
        """Generate multiple independent completions for the same prompt."""

    @abstractmethod
    def score(
        self,
        prompt: str,
        text: str,
    ) -> float:
        """Score how well *text* follows *prompt* (higher = better).

        Returns a normalised score in [0, 1].
        """

    def __repr__(self) -> str:
        loaded = "loaded" if self.is_loaded() else "not loaded"
        return f"{self.__class__.__name__}({loaded})"
