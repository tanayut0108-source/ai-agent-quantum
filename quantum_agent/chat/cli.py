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
        help="Path to GGUF model file",
    )
    parser.add_argument(
        "--ctx", type=int, default=2048,
        help="Context window size (default: 2048)",
    )
    parser.add_argument(
        "--threads", type=int, default=0,
        help="CPU threads (0 = auto)",
    )
    parser.add_argument(
        "--gpu-layers", type=int, default=0,
        help="GPU layers to offload (0 = CPU only)",
    )
    parser.add_argument(
        "--quantum", action="store_true",
        help="Enable quantum hypothesis mode",
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
    args = parser.parse_args(argv)

    try:
        from quantum_agent.llm import LlamaBackend
    except ImportError:
        print("Error: llama-cpp-python is required.")
        print("Install: pip install llama-cpp-python")
        sys.exit(1)

    from quantum_agent.chat.session import ChatSession
    from quantum_agent.llm.llama_backend import LlamaConfig

    config = LlamaConfig(
        model_path=args.model,
        n_ctx=args.ctx,
        n_threads=args.threads,
        n_gpu_layers=args.gpu_layers,
    )

    print(f"Loading model: {args.model}")
    llm = LlamaBackend(config=config)
    llm.load()
    print(f"Model loaded: {llm.model_info().name}")
    if llm._is_chat_model:
        print("Chat model detected")

    session = ChatSession(
        provider=llm,
        system_prompt=args.system,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        quantum_mode=args.quantum,
    )

    mode = "quantum" if args.quantum else "standard"
    print(f"\nQuantum Agent Chat ({mode} mode)")
    print("Type 'quit' or 'exit' to end")
    print("Type '/reset' to clear history")
    print("Type '/quantum on' or '/quantum off' to toggle mode")
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

        if user_input == "/info":
            info = session.get_context_summary()
            for key, val in info.items():
                print(f"  {key}: {val}")
            continue

        if user_input.startswith("/"):
            print(f"Unknown command: {user_input}")
            continue

        reply = session.send(user_input)
        print(f"\nAgent: {reply.content}")


if __name__ == "__main__":
    main()
