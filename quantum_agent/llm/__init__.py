"""LLM integration — GGUF / llama.cpp backend for quantum agent reasoning."""

from quantum_agent.llm.llama_backend import LlamaBackend
from quantum_agent.llm.provider import GenerationResult, LLMProvider, ModelInfo
from quantum_agent.llm.quantum_llm import QuantumLLM

__all__ = [
    "GenerationResult",
    "LLMProvider",
    "LlamaBackend",
    "ModelInfo",
    "QuantumLLM",
]
