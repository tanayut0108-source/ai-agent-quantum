"""Auto-setup: install Ollama, pull model, start server, launch chat.

Usage::

    python -m quantum_agent.setup --model gemma2:9b
    python -m quantum_agent.setup --model tinyllama --fusion
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request


def _is_termux() -> bool:
    return os.path.isdir("/data/data/com.termux")


def _ollama_installed() -> bool:
    return shutil.which("ollama") is not None


def _ollama_running(base_url: str = "http://localhost:11434") -> bool:
    try:
        urllib.request.urlopen(f"{base_url}/api/tags", timeout=3)
        return True
    except Exception:
        return False


def _run(cmd: list[str], *, check: bool = True) -> int:
    print(f"  > {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if check and result.returncode != 0:
        print(f"  [WARN] Command exited with code {result.returncode}")
    return result.returncode


def _install_ollama() -> bool:
    print("\n[1/4] Installing Ollama...")
    if _ollama_installed():
        print("  Ollama already installed!")
        return True

    system = platform.system().lower()
    if _is_termux():
        print("  Detected Termux — installing via install script...")
        code = _run(["bash", "-c", "curl -fsSL https://ollama.com/install.sh | sh"])
        if code != 0:
            print("  Trying alternative: pkg install golang && go install...")
            _run(["pkg", "install", "-y", "golang"])
            _run(["go", "install", "github.com/ollama/ollama@latest"])
    elif system == "linux":
        _run(["bash", "-c", "curl -fsSL https://ollama.com/install.sh | sh"])
    elif system == "darwin":
        print("  macOS: please install from https://ollama.com/download")
        return False
    else:
        print(f"  Unsupported platform: {system}")
        print("  Please install Ollama manually: https://ollama.com/download")
        return False

    return _ollama_installed()


def _start_server(base_url: str = "http://localhost:11434") -> bool:
    print("\n[2/4] Starting Ollama server...")
    if _ollama_running(base_url):
        print("  Server already running!")
        return True

    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    for i in range(15):
        time.sleep(2)
        if _ollama_running(base_url):
            print("  Server started!")
            return True
        print(f"  Waiting... ({i + 1})")

    print("  [ERROR] Server did not start in time.")
    return False


def _pull_model(model: str) -> bool:
    print(f"\n[3/4] Pulling model: {model}")
    print("  (This may take a while for large models)")
    code = _run(["ollama", "pull", model])
    return code == 0


def _launch_chat(
    model: str,
    *,
    fusion: bool = False,
    quantum: bool = False,
    ternary: bool = False,
    base_url: str = "http://localhost:11434",
) -> None:
    print("\n[4/4] Launching chat...")
    print("=" * 40)

    argv = ["--model", model, "--ollama-url", base_url]
    if fusion:
        argv.append("--fusion")
    elif quantum:
        argv.append("--quantum")
    elif ternary:
        argv.append("--ternary")

    from quantum_agent.chat.cli import main as chat_main

    chat_main(argv)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Quantum Agent — auto-setup and launch",
    )
    parser.add_argument(
        "--model", "-m",
        default="tinyllama",
        help="Ollama model name (default: tinyllama)",
    )
    parser.add_argument(
        "--fusion", action="store_true",
        help="Start in fusion mode (recommended)",
    )
    parser.add_argument(
        "--quantum", action="store_true",
        help="Start in quantum mode",
    )
    parser.add_argument(
        "--ternary", action="store_true",
        help="Start in ternary mode",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--skip-install", action="store_true",
        help="Skip Ollama installation check",
    )
    args = parser.parse_args(argv)

    print("=" * 40)
    print("  Quantum Agent — Auto Setup")
    print("=" * 40)

    if not args.skip_install and not _install_ollama():
        print("\n[ERROR] Could not install Ollama.")
        print("Install manually: https://ollama.com/download")
        sys.exit(1)

    if not _start_server(args.ollama_url):
        print("\n[ERROR] Could not start Ollama server.")
        print("Try running 'ollama serve' manually in another terminal.")
        sys.exit(1)

    if not _pull_model(args.model):
        print(f"\n[ERROR] Could not pull model: {args.model}")
        sys.exit(1)

    _launch_chat(
        args.model,
        fusion=args.fusion,
        quantum=args.quantum,
        ternary=args.ternary,
        base_url=args.ollama_url,
    )


if __name__ == "__main__":
    main()
