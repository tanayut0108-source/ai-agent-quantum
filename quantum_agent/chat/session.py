"""Chat session with conversation memory and quantum reasoning.

A :class:`ChatSession` wraps an :class:`LLMProvider` to maintain a
multi-turn conversation.  Each turn can optionally trigger quantum
hypothesis generation and evaluation so the agent explores multiple
approaches before answering.

Usage::

    from quantum_agent.llm import LlamaBackend
    from quantum_agent.chat import ChatSession

    llm = LlamaBackend(model_path="model.gguf")
    llm.load()

    session = ChatSession(llm)
    reply = session.send("How do I build a web scraper?")
    print(reply.content)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from quantum_agent.chat.knowledge import KnowledgeStore
from quantum_agent.llm.provider import LLMProvider

logger = logging.getLogger(__name__)

_TERNARY_SYSTEM = (
    "You are a ternary logic oracle. You MUST answer every question "
    "with ONLY one of these three values:\n"
    "  1  = yes / true / agree\n"
    "  0  = uncertain / unknown / maybe\n"
    " -1  = no / false / disagree\n\n"
    "Respond with ONLY the number (-1, 0, or 1). "
    "Do NOT add any other text, explanation, or punctuation."
)


@dataclass(slots=True)
class ChatMessage:
    """A single message in a conversation."""

    role: str  # "user", "assistant", or "system"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, str]:
        """Convert to the format expected by LLM chat APIs."""
        return {"role": self.role, "content": self.content}


class ChatSession:
    """Multi-turn conversation session with quantum reasoning.

    Parameters
    ----------
    provider : an :class:`LLMProvider` (e.g. LlamaBackend)
    system_prompt : optional system message prepended to every request
    max_history : max messages to keep in context window (0 = unlimited)
    max_tokens : default max tokens per response
    temperature : default generation temperature
    quantum_mode : when True, use hypothesis generation for complex queries
    knowledge : optional :class:`KnowledgeStore` for persistent memory
    """

    def __init__(
        self,
        provider: LLMProvider,
        system_prompt: str = "",
        *,
        max_history: int = 50,
        max_tokens: int = 512,
        temperature: float = 0.7,
        quantum_mode: bool = False,
        ternary_mode: bool = False,
        knowledge: KnowledgeStore | None = None,
    ) -> None:
        self.provider = provider
        self.max_history = max_history
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.quantum_mode = quantum_mode
        self.ternary_mode = ternary_mode
        self.knowledge = knowledge

        self._history: list[ChatMessage] = []
        self._system_prompt = system_prompt or (
            "You are a helpful quantum reasoning assistant. "
            "Answer concisely and clearly."
        )
        self._turn_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def history(self) -> list[ChatMessage]:
        """Return a copy of the conversation history."""
        return list(self._history)

    @property
    def turn_count(self) -> int:
        return self._turn_count

    def send(
        self,
        message: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ChatMessage:
        """Send a user message and get the assistant's reply.

        Returns the assistant's :class:`ChatMessage`.
        """
        user_msg = ChatMessage(role="user", content=message)
        self._history.append(user_msg)

        messages = self._build_messages()

        tokens = max_tokens if max_tokens is not None else self.max_tokens
        temp = temperature if temperature is not None else self.temperature

        if self.ternary_mode:
            reply_text = self._ternary_reply(messages, tokens, temp)
        elif self.quantum_mode:
            reply_text = self._quantum_reply(message, messages, tokens, temp)
        else:
            reply_text = self._standard_reply(messages, tokens, temp)

        assistant_msg = ChatMessage(
            role="assistant",
            content=reply_text,
            metadata={"turn": self._turn_count},
        )
        self._history.append(assistant_msg)
        self._turn_count += 1

        self._trim_history()

        return assistant_msg

    def reset(self) -> None:
        """Clear all conversation history."""
        self._history.clear()
        self._turn_count = 0

    def set_system_prompt(self, prompt: str) -> None:
        """Update the system prompt for future messages."""
        self._system_prompt = prompt

    def remember(self, text: str, tags: list[str] | None = None) -> str:
        """Store a piece of knowledge in persistent memory."""
        if self.knowledge is None:
            return "No knowledge store configured."
        self.knowledge.add(text, tags=tags)
        return f"Remembered ({self.knowledge.count} total)."

    def forget(self, index: int) -> str:
        """Remove a memory by index."""
        if self.knowledge is None:
            return "No knowledge store configured."
        mem = self.knowledge.remove(index)
        if mem:
            return f"Forgot: {mem.text[:60]}"
        return "Invalid index."

    def list_memories(self, query: str = "") -> list[str]:
        """List stored memories, optionally filtered by query."""
        if self.knowledge is None:
            return []
        memories = self.knowledge.search(query) if query else self.knowledge.get_all()
        return [f"[{i}] {m.text}" for i, m in enumerate(memories)]

    def get_context_summary(self) -> dict[str, Any]:
        """Return a summary of the current conversation state."""
        summary: dict[str, Any] = {
            "turn_count": self._turn_count,
            "message_count": len(self._history),
            "system_prompt": self._system_prompt[:80] + "..."
            if len(self._system_prompt) > 80
            else self._system_prompt,
            "quantum_mode": self.quantum_mode,
            "ternary_mode": self.ternary_mode,
            "model": self.provider.model_info().name,
        }
        if self.knowledge:
            summary["memories"] = self.knowledge.count
        return summary

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_messages(self) -> list[dict[str, str]]:
        """Build the message list for the LLM, including system prompt."""
        system = _TERNARY_SYSTEM if self.ternary_mode else self._system_prompt

        if self.knowledge and not self.ternary_mode:
            user_query = ""
            for msg in reversed(self._history):
                if msg.role == "user":
                    user_query = msg.content
                    break
            context = self.knowledge.as_context(query=user_query, max_items=5)
            if context:
                system = f"{system}\n\n{context}"

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
        ]
        for msg in self._history:
            messages.append(msg.to_dict())
        return messages

    def _standard_reply(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Generate a reply using chat or generate fallback."""
        if hasattr(self.provider, "chat"):
            result = self.provider.chat(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return result.text

        prompt = self._messages_to_prompt(messages)
        result = self.provider.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return result.text

    def _quantum_reply(
        self,
        user_message: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Generate a reply with quantum hypothesis exploration."""
        from quantum_agent.llm.quantum_llm import QuantumLLM

        qllm = QuantumLLM(
            self.provider,
            default_temperature=temperature,
            default_max_tokens=max_tokens,
        )

        context = {}
        if len(self._history) > 1:
            recent = [
                m.content for m in self._history[-5:]
                if m.role == "user"
            ]
            if recent:
                context["recent_topics"] = "; ".join(recent[-3:])

        hypotheses = qllm.generate_hypotheses(
            task=user_message, n=3, context=context,
        )

        hyp_summary = "\n".join(
            f"- {h['label']}: {', '.join(h.get('steps', [])[:3])}"
            for h in hypotheses
        )
        synthesis_prompt = (
            f"Based on these approaches:\n{hyp_summary}\n\n"
            f"Give a clear, helpful answer to: {user_message}"
        )
        messages_with_synthesis = messages[:-1] + [
            {"role": "user", "content": synthesis_prompt},
        ]

        return self._standard_reply(
            messages_with_synthesis, max_tokens, temperature,
        )

    def _trim_history(self) -> None:
        """Trim history to max_history if set."""
        if self.max_history > 0 and len(self._history) > self.max_history:
            excess = len(self._history) - self.max_history
            self._history = self._history[excess:]

    @staticmethod
    def _messages_to_prompt(messages: list[dict[str, str]]) -> str:
        """Convert chat messages to a plain prompt for non-chat models."""
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

    def _ternary_reply(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Generate a ternary (-1, 0, 1) answer."""
        raw = self._standard_reply(messages, max_tokens=20, temperature=0.0)
        return self._parse_ternary(raw)

    @staticmethod
    def _parse_ternary(text: str) -> str:
        """Extract -1, 0, or 1 from LLM output."""
        text = text.strip()
        if text in ("-1", "0", "1"):
            return text
        match = re.search(r"(-1|[01])", text)
        if match:
            return match.group(1)
        lower = text.lower()
        uncertain = (
            "maybe", "perhaps", "not sure", "uncertain", "unknown",
            "possibly", "might", "ไม่แน่", "อาจจะ", "ไม่แน่นอน",
        )
        if any(w in lower for w in uncertain):
            return "0"
        if any(w in lower for w in ("yes", "true", "agree", "correct", "ใช่", "ถูก")):
            return "1"
        if any(w in lower for w in ("no", "false", "disagree", "wrong", "ไม่", "ผิด")):
            return "-1"
        return "0"

    def __repr__(self) -> str:
        return (
            f"ChatSession(turns={self._turn_count}, "
            f"messages={len(self._history)}, "
            f"quantum={self.quantum_mode}, "
            f"ternary={self.ternary_mode})"
        )
