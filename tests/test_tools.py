"""Tests for tool registry."""

import asyncio

from quantum_agent.tools.registry import ToolRegistry, tool


@tool(name="test_echo", description="Echo tool", keywords=["echo"])
async def echo_tool(params):
    return {"echo": params.get("input", "")}


class TestToolRegistry:
    def test_global_registration(self):
        registry = ToolRegistry()
        assert "test_echo" in registry.list_tools()

    def test_local_registration(self):
        registry = ToolRegistry()

        async def my_tool(params):
            return {"result": "ok"}

        registry.register("my_tool", my_tool, keywords=["test"])
        assert "my_tool" in registry.list_tools()

    def test_execute(self):
        registry = ToolRegistry()
        result = asyncio.run(registry.execute("test_echo", {"input": "hello"}))
        assert result["echo"] == "hello"
        assert result["status"] == "ok"

    def test_execute_missing_tool(self):
        registry = ToolRegistry()
        result = asyncio.run(registry.execute("nonexistent", {}))
        assert result["status"] == "error"

    def test_get_tool(self):
        registry = ToolRegistry()
        info = registry.get_tool("test_echo")
        assert info is not None
        assert info["name"] == "test_echo"
