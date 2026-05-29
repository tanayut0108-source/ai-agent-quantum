"""Dynamic tool registry with decorator-based registration.

Agents can discover and invoke tools at runtime.  Tools are async
callables with metadata (name, description, keywords).
"""

from __future__ import annotations

import functools
from collections.abc import Awaitable, Callable
from typing import Any

ToolFn = Callable[..., Awaitable[dict[str, Any]]]

_GLOBAL_TOOLS: dict[str, dict[str, Any]] = {}


def tool(
    name: str,
    description: str = "",
    keywords: list[str] | None = None,
) -> Callable[[ToolFn], ToolFn]:
    """Decorator to register a function as a tool."""

    def decorator(fn: ToolFn) -> ToolFn:
        _GLOBAL_TOOLS[name] = {
            "name": name,
            "description": description,
            "keywords": keywords or [],
            "fn": fn,
        }

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
            return await fn(*args, **kwargs)

        return wrapper

    return decorator


class ToolRegistry:
    """Per-agent tool registry.  Falls back to the global registry."""

    def __init__(self) -> None:
        self._local: dict[str, dict[str, Any]] = {}

    def register(
        self,
        name: str,
        fn: ToolFn,
        description: str = "",
        keywords: list[str] | None = None,
    ) -> None:
        self._local[name] = {
            "name": name,
            "description": description,
            "keywords": keywords or [],
            "fn": fn,
        }

    def get_tool(self, name: str) -> dict[str, Any] | None:
        return self._local.get(name) or _GLOBAL_TOOLS.get(name)

    def list_tools(self) -> list[str]:
        names = set(self._local.keys()) | set(_GLOBAL_TOOLS.keys())
        return sorted(names)

    async def execute(self, name: str, params: dict[str, Any]) -> dict[str, Any]:
        tool_info = self.get_tool(name)
        if not tool_info:
            return {"error": f"Tool '{name}' not found", "status": "error"}
        try:
            result = await tool_info["fn"](params)
            return {**result, "status": "ok"}
        except Exception as e:
            return {"error": str(e), "status": "error"}

    def __repr__(self) -> str:
        return f"ToolRegistry(local={len(self._local)}, global={len(_GLOBAL_TOOLS)})"
