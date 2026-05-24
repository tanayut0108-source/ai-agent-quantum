"""Quantum–LLM bridge — use an LLM backend for quantum agent reasoning.

:class:`QuantumLLM` connects the quantum framework's hypothesis-based
reasoning to an actual language model.  It can:

1. **Generate hypotheses** — ask the LLM to brainstorm multiple candidate
   plans/answers, each becoming a hypothesis in superposition.
2. **Evaluate hypotheses** — use the LLM to score each hypothesis, then
   translate scores into quantum amplitudes.
3. **LLM-as-evaluator** — serve as an ``EvaluatorFn`` for interference
   filters, letting the LLM judge hypothesis quality.
4. **Chat reasoning** — run a full reasoning cycle using chat messages.

All methods work with any :class:`LLMProvider` (GGUF, OpenAI, etc.).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from quantum_agent.core.qubit import Hypothesis, QuantumState
from quantum_agent.core.superposition import Superposition
from quantum_agent.llm.provider import LLMProvider

logger = logging.getLogger(__name__)


_HYPOTHESIS_PROMPT = """Generate exactly {n} different approaches to: {task}

{context_str}For each approach, write a numbered heading and bullet point steps."""

_EVALUATION_PROMPT = """Rate this approach for the task "{task}" \
on a scale of 0.0 to 1.0.

Approach: {label}
Steps: {steps}

Respond with ONLY a number between 0.0 and 1.0.

Score:"""

_CHAT_REASON_SYSTEM = (
    "You are a quantum reasoning agent. You explore multiple hypotheses "
    "in parallel and evaluate them critically. Be concise and structured."
)


class QuantumLLM:
    """Bridge between quantum agent reasoning and an LLM backend.

    Parameters
    ----------
    provider : an :class:`LLMProvider` implementation (e.g. LlamaBackend)
    default_temperature : temperature for generation calls
    default_max_tokens : max tokens per generation
    """

    def __init__(
        self,
        provider: LLMProvider,
        default_temperature: float = 0.7,
        default_max_tokens: int = 512,
    ) -> None:
        self.provider = provider
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens

    # ------------------------------------------------------------------
    # Hypothesis generation
    # ------------------------------------------------------------------

    def generate_hypotheses(
        self,
        task: str,
        n: int = 4,
        context: dict[str, Any] | None = None,
        temperature: float | None = None,
    ) -> list[dict[str, Any]]:
        """Ask the LLM to generate *n* candidate hypotheses for a task.

        Returns a list of dicts suitable for ``Superposition.branch()``.
        Falls back to a simple heuristic if the LLM output cannot be parsed.
        """
        context = context or {}
        context_str = ""
        if context:
            context_parts = [f"- {k}: {v}" for k, v in context.items() if k != "original_task"]
            if context_parts:
                context_str = "Context:\n" + "\n".join(context_parts) + "\n\n"

        prompt = _HYPOTHESIS_PROMPT.format(
            n=n, task=task, context_str=context_str
        )

        result = self.provider.generate(
            prompt=prompt,
            max_tokens=self.default_max_tokens,
            temperature=temperature if temperature is not None else self.default_temperature,
        )

        hypotheses = self._parse_hypotheses(result.text, n)

        if not hypotheses:
            logger.warning("LLM output could not be parsed; using fallback hypotheses")
            hypotheses = self._fallback_hypotheses(task, n)

        return hypotheses

    def generate_hypotheses_as_state(
        self,
        task: str,
        n: int = 4,
        context: dict[str, Any] | None = None,
    ) -> Superposition:
        """Generate hypotheses and return a ready-to-use Superposition."""
        hyps = self.generate_hypotheses(task, n=n, context=context)
        sup = Superposition()
        sup.branch(hyps)
        return sup

    # ------------------------------------------------------------------
    # Hypothesis evaluation
    # ------------------------------------------------------------------

    def evaluate_hypothesis(
        self,
        task: str,
        hypothesis: Hypothesis,
    ) -> float:
        """Ask the LLM to score a single hypothesis. Returns a float in [0, 1]."""
        steps = hypothesis.data.get("steps", [])
        steps_str = ", ".join(steps) if steps else "(no steps)"

        prompt = _EVALUATION_PROMPT.format(
            task=task, label=hypothesis.label, steps=steps_str
        )

        result = self.provider.generate(
            prompt=prompt,
            max_tokens=16,
            temperature=0.1,
        )

        return self._parse_score(result.text)

    def evaluate_all(
        self,
        task: str,
        state: QuantumState,
    ) -> dict[str, float]:
        """Score every hypothesis in a QuantumState."""
        scores: dict[str, float] = {}
        for h in state.hypotheses:
            scores[h.label] = self.evaluate_hypothesis(task, h)
        return scores

    def create_evaluator(self, task: str):  # noqa: ANN201
        """Return an evaluator function compatible with interference_filter.

        Usage::

            evaluator = quantum_llm.create_evaluator("build a web app")
            new_state = interference_filter(state, evaluators=[evaluator])
        """

        def _llm_evaluator(h: Hypothesis) -> float:
            return self.evaluate_hypothesis(task, h)

        return _llm_evaluator

    # ------------------------------------------------------------------
    # Evolve state using LLM scores
    # ------------------------------------------------------------------

    def evolve_state(
        self,
        task: str,
        superposition: Superposition,
    ) -> QuantumState:
        """Score all hypotheses with the LLM and update amplitudes."""
        scores = self.evaluate_all(task, superposition.state)

        def scoring_fn(h: Hypothesis) -> float:
            return scores.get(h.label, 0.5)

        return superposition.evolve(scoring_fn)

    # ------------------------------------------------------------------
    # Chat-style reasoning
    # ------------------------------------------------------------------

    def chat_reason(
        self,
        task: str,
        context: dict[str, Any] | None = None,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        """Run a single-turn chat reasoning step.

        Returns a dict with the LLM's response and metadata.
        """
        context = context or {}
        context_str = ""
        if context:
            context_parts = [f"- {k}: {v}" for k, v in context.items() if k != "original_task"]
            if context_parts:
                context_str = "\n\nContext:\n" + "\n".join(context_parts)

        messages = [
            {"role": "system", "content": _CHAT_REASON_SYSTEM},
            {"role": "user", "content": f"Task: {task}{context_str}"},
        ]

        if hasattr(self.provider, "chat"):
            result = self.provider.chat(
                messages=messages,
                max_tokens=max_tokens,
                temperature=self.default_temperature,
            )
        else:
            prompt = "\n".join(
                f"[{m['role'].title()}] {m['content']}" for m in messages
            )
            prompt += "\n[Assistant]"
            result = self.provider.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=self.default_temperature,
            )

        return {
            "response": result.text,
            "tokens_used": result.tokens_used,
            "finish_reason": result.finish_reason,
        }

    # ------------------------------------------------------------------
    # Scoring via token probabilities
    # ------------------------------------------------------------------

    def score_text(self, prompt: str, text: str) -> float:
        """Score text using the provider's token-probability scoring."""
        return self.provider.score(prompt, text)

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_hypotheses(text: str, n: int) -> list[dict[str, Any]]:
        """Parse hypotheses from LLM output (JSON or plain text)."""
        text = text.strip()

        # --- Try JSON first ---
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                if isinstance(data, list):
                    hypotheses: list[dict[str, Any]] = []
                    for item in data[:n]:
                        if not isinstance(item, dict):
                            continue
                        label = item.get(
                            "label", f"hypothesis-{len(hypotheses)}"
                        )
                        steps = item.get("steps", [])
                        if not isinstance(steps, list):
                            steps = [str(steps)]
                        hypotheses.append({
                            "label": str(label),
                            "steps": [str(s) for s in steps],
                        })
                    if hypotheses:
                        return hypotheses
            except json.JSONDecodeError:
                pass

        # --- Fallback: parse numbered/bulleted plain text ---
        return QuantumLLM._parse_plain_text_hypotheses(text, n)

    @staticmethod
    def _parse_plain_text_hypotheses(
        text: str, n: int,
    ) -> list[dict[str, Any]]:
        """Extract hypotheses from numbered lists or paragraphs."""
        heading_re = re.compile(
            r"(?:^|\n)"
            r"\s*(?:#+\s*|\*\*|\d+[\.\):]\s*)"
            r"([^\n]{3,80})",
        )
        bullet_re = re.compile(r"^\s*[-*•]\s+(.+)", re.MULTILINE)
        numbered_step_re = re.compile(
            r"^\s*(?:\d+[\.\)]|[a-z][\.\)])\s+(.+)", re.MULTILINE,
        )

        headings = heading_re.findall(text)

        if headings:
            hypotheses: list[dict[str, Any]] = []
            for i, heading in enumerate(headings[:n]):
                label = re.sub(r"[\*#]+", "", heading).strip()
                start = text.find(heading) + len(heading)
                end = (
                    text.find(headings[i + 1])
                    if i + 1 < len(headings)
                    else len(text)
                )
                section = text[start:end]
                steps = bullet_re.findall(section)
                if not steps:
                    steps = numbered_step_re.findall(section)
                if not steps:
                    steps = [
                        s.strip()
                        for s in section.strip().split("\n")
                        if s.strip()
                    ][:5]
                hypotheses.append({
                    "label": label[:60],
                    "steps": [s.strip() for s in steps[:8]],
                })
            if hypotheses:
                return hypotheses

        paragraphs = [
            p.strip() for p in text.split("\n\n") if p.strip()
        ]
        if len(paragraphs) >= 2:
            results: list[dict[str, Any]] = []
            for i, para in enumerate(paragraphs[:n]):
                lines = [ln.strip() for ln in para.split("\n") if ln.strip()]
                label = lines[0][:60] if lines else f"approach-{i + 1}"
                steps = lines[1:] if len(lines) > 1 else lines
                results.append({"label": label, "steps": steps[:8]})
            if results:
                return results

        return []

    @staticmethod
    def _parse_score(text: str) -> float:
        """Extract a float score from LLM output."""
        text = text.strip()
        match = re.search(r"(\d+\.?\d*)", text)
        if match:
            score = float(match.group(1))
            return max(0.0, min(1.0, score))
        return 0.5

    @staticmethod
    def _fallback_hypotheses(task: str, n: int) -> list[dict[str, Any]]:
        """Generate simple heuristic hypotheses when LLM parsing fails."""
        strategies = [
            ("direct", [f"Directly execute: {task}"]),
            ("step-by-step", [f"Analyse: {task}", f"Plan: {task}", f"Execute: {task}"]),
            ("divide", [f"Break down: {task}", "Solve each part", "Combine results"]),
            ("iterative", ["Draft solution", "Evaluate quality", "Refine", "Finalise"]),
            ("research-first", ["Research context", f"Plan for: {task}", "Execute", "Verify"]),
        ]
        results: list[dict[str, Any]] = []
        for label, steps in strategies[:n]:
            results.append({"label": label, "steps": steps})
        return results

    def __repr__(self) -> str:
        return f"QuantumLLM(provider={self.provider!r})"
