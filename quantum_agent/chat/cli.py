"""Interactive CLI chat — run with ``python -m quantum_agent.chat``."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Quantum Agent interactive chat",
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Model name for Ollama (e.g. tinyllama) or path to GGUF file",
    )
    parser.add_argument(
        "--backend", "-b",
        choices=["auto", "ollama", "llama"],
        default="auto",
        help="Backend: 'ollama', 'llama' (GGUF), or 'auto' (default)",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--ctx", type=int, default=2048,
        help="Context window size (default: 2048)",
    )
    parser.add_argument(
        "--threads", type=int, default=0,
        help="CPU threads for llama.cpp (0 = auto)",
    )
    parser.add_argument(
        "--gpu-layers", type=int, default=0,
        help="GPU layers to offload (0 = CPU only, llama.cpp)",
    )
    parser.add_argument(
        "--quantum", action="store_true",
        help="Enable quantum hypothesis mode",
    )
    parser.add_argument(
        "--ternary", action="store_true",
        help="Enable ternary mode: AI answers -1, 0, or 1 only",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Generation temperature (default: 0.7)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512,
        help="Max tokens per response (default: 512)",
    )
    parser.add_argument(
        "--system", type=str, default="",
        help="Custom system prompt",
    )
    parser.add_argument(
        "--memory", type=str, default="",
        help="Path to memory file (default: ~/.quantum_agent_memory.json)",
    )
    args = parser.parse_args(argv)

    backend = _detect_backend(args.backend, args.model)

    llm = _load_ollama(args) if backend == "ollama" else _load_llama(args)

    from quantum_agent.chat.knowledge import KnowledgeStore
    from quantum_agent.chat.session import ChatSession

    knowledge = KnowledgeStore(path=args.memory or None)
    print(f"Memory: {knowledge.count} memories loaded")

    session = ChatSession(
        provider=llm,
        system_prompt=args.system,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        quantum_mode=args.quantum,
        ternary_mode=args.ternary,
        knowledge=knowledge,
    )

    mode = "ternary" if args.ternary else ("quantum" if args.quantum else "standard")
    print(f"\nQuantum Agent Chat ({mode} mode)")
    print("Type 'quit' or 'exit' to end")
    print("Type '/reset' to clear history")
    print("Type '/quantum on' or '/quantum off' to toggle mode")
    print("Type '/ternary on' or '/ternary off' to toggle ternary mode")
    print("Type '/remember ...' to store knowledge")
    print("Type '/memories' to list stored knowledge")
    print("Type '/forget N' to remove memory by index")
    print("Type '/info' for session info")
    print("-" * 40)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("Bye!")
            break

        if user_input == "/reset":
            session.reset()
            print("History cleared.")
            continue

        if user_input == "/quantum on":
            session.quantum_mode = True
            print("Quantum mode: ON")
            continue

        if user_input == "/quantum off":
            session.quantum_mode = False
            print("Quantum mode: OFF")
            continue

        if user_input == "/ternary on":
            session.ternary_mode = True
            print("Ternary mode: ON (-1 / 0 / 1)")
            continue

        if user_input == "/ternary off":
            session.ternary_mode = False
            print("Ternary mode: OFF")
            continue

        if user_input == "/info":
            info = session.get_context_summary()
            for key, val in info.items():
                print(f"  {key}: {val}")
            continue

        if user_input.startswith("/remember "):
            text = user_input[10:].strip()
            if text:
                result = session.remember(text)
                print(result)
            else:
                print("Usage: /remember <text>")
            continue

        if user_input == "/memories":
            memories = session.list_memories()
            if memories:
                for m in memories:
                    print(f"  {m}")
            else:
                print("  No memories stored.")
            continue

        if user_input.startswith("/memories "):
            query = user_input[10:].strip()
            memories = session.list_memories(query)
            if memories:
                for m in memories:
                    print(f"  {m}")
            else:
                print(f"  No memories matching '{query}'.")
            continue

        if user_input.startswith("/forget "):
            try:
                idx = int(user_input[8:].strip())
                result = session.forget(idx)
                print(result)
            except ValueError:
                print("Usage: /forget <index>")
            continue

        if user_input.startswith("/"):
            print(f"Unknown command: {user_input}")
            continue

        reply = session.send(user_input)
        print(f"\nAgent: {reply.content}")


def _detect_backend(backend: str, model: str) -> str:
    """Auto-detect which backend to use based on model string."""
    if backend != "auto":
        return backend
    if model.endswith(".gguf") or "/" in model or "\\" in model:
        return "llama"
    return "ollama"


def _load_ollama(args: argparse.Namespace) -> object:
    """Load an Ollama backend."""
    from quantum_agent.llm.ollama_backend import OllamaBackend

    print(f"Using Ollama backend: {args.model}")
    llm = OllamaBackend(model=args.model, base_url=args.ollama_url)
    llm.load()
    print(f"Model ready: {llm.model_info().name}")
    return llm


def _load_llama(args: argparse.Namespace) -> object:
    """Load a llama.cpp GGUF backend."""
    try:
        from quantum_agent.llm.llama_backend import LlamaBackend, LlamaConfig
    except ImportError:
        print("Error: llama-cpp-python is required for GGUF models.")
        print("Install: pip install llama-cpp-python")
        sys.exit(1)

    config = LlamaConfig(
        model_path=args.model,
        n_ctx=args.ctx,
        n_threads=args.threads,
        n_gpu_layers=args.gpu_layers,
    )

    print(f"Loading GGUF model: {args.model}")
    llm = LlamaBackend(config=config)
    llm.load()
    print(f"Model loaded: {llm.model_info().name}")
    if llm._is_chat_model:
        print("Chat model detected")
    return llm


if __name__ == "__main__":
    main()
