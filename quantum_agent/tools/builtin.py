"""Built-in tools available to all quantum agents."""

from __future__ import annotations

from typing import Any

from quantum_agent.tools.registry import tool


@tool(
    name="think",
    description="Internal reasoning step",
    keywords=["think", "reason", "analyse"],
)
async def think_tool(params: dict[str, Any]) -> dict[str, Any]:
    """Structured thinking — breaks input into observations and conclusions."""
    text = params.get("input", "")
    words = text.split()
    observations = [
        f"Token group {i}: {' '.join(words[i : i + 5])}"
        for i in range(0, min(len(words), 15), 5)
    ]
    return {
        "thought": f"Analysed '{text[:80]}...' ({len(words)} tokens)",
        "observations": observations,
    }


@tool(
    name="search_memory",
    description="Search agent memory",
    keywords=["search", "memory", "recall", "remember"],
)
async def search_memory_tool(params: dict[str, Any]) -> dict[str, Any]:
    """Placeholder — agents override this with their own memory instance."""
    query = params.get("input", "")
    return {"results": [], "query": query}


@tool(
    name="calculate",
    description="Perform arithmetic",
    keywords=["calculate", "math", "compute", "number"],
)
async def calculate_tool(params: dict[str, Any]) -> dict[str, Any]:
    """Safe arithmetic evaluator."""
    expr = params.get("input", "0")
    allowed = set("0123456789+-*/.(). ")
    if not all(c in allowed for c in expr):
        return {"error": "Expression contains disallowed characters"}
    try:
        result = eval(expr, {"__builtins__": {}})  # noqa: S307
        return {"expression": expr, "result": result}
    except Exception as e:
        return {"error": str(e)}


@tool(
    name="summarize",
    description="Summarize text",
    keywords=["summarize", "summary", "brief", "shorten"],
)
async def summarize_tool(params: dict[str, Any]) -> dict[str, Any]:
    """Simple extractive summary — returns the first few sentences."""
    text = params.get("input", "")
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    summary = ". ".join(sentences[:3]) + "." if sentences else text[:200]
    return {
        "summary": summary,
        "original_length": len(text),
        "summary_length": len(summary),
    }
