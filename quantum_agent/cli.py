"""CLI interface for AI Agent Quantum."""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from quantum_agent.core.qubit import Hypothesis, QuantumState
from quantum_agent.orchestrator.quantum_orchestrator import (
    OrchestratorConfig,
    QuantumOrchestrator,
)
from quantum_agent.reasoning.quantum_reasoning import QuantumReasoner

app = typer.Typer(
    name="quantum-agent",
    help="AI Agent Quantum — quantum-inspired multi-agent AI framework",
    add_completion=False,
)
console = Console()


@app.command()
def run(
    task: str = typer.Argument(..., help="The task to execute"),
    max_cycles: int = typer.Option(3, "--cycles", "-c", help="Max plan-execute-critique cycles"),
    verbose: bool = typer.Option(True, "--verbose/--quiet", "-v/-q"),
) -> None:
    """Run the quantum orchestrator on a task."""
    config = OrchestratorConfig(max_cycles=max_cycles, verbose=verbose)
    orchestrator = QuantumOrchestrator(config=config)

    console.print(
        Panel(
            f"[bold cyan]Task:[/] {task}\n"
            f"[bold cyan]Max cycles:[/] {max_cycles}",
            title="[bold magenta]AI Agent Quantum[/]",
            border_style="bright_blue",
        )
    )

    result = asyncio.run(orchestrator.run(task))
    _render_result(result, verbose)


@app.command()
def reason(
    question: str = typer.Argument(..., help="Question to reason about"),
    hypotheses: str = typer.Option(
        "",
        "--hypotheses",
        "-H",
        help="Comma-separated hypothesis labels",
    ),
) -> None:
    """Run quantum reasoning on a question with hypotheses."""
    if hypotheses:
        hyp_list = [{"label": h.strip()} for h in hypotheses.split(",")]
    else:
        hyp_list = [
            {"label": "approach_A"},
            {"label": "approach_B"},
            {"label": "approach_C"},
        ]

    def score_fn(h: Hypothesis) -> float:
        return 0.5 + 0.1 * len(h.label)

    reasoner = QuantumReasoner()
    result = reasoner.reason(hyp_list, [score_fn])

    console.print(Panel(f"[bold cyan]Question:[/] {question}", title="Quantum Reasoning"))
    console.print(f"[bold green]Answer:[/] {result.answer}")
    console.print(f"[bold green]Confidence:[/] {result.confidence:.4f}")
    trace_str = " -> ".join(f"{e:.3f}" for e in result.entropy_trace)
    console.print(f"[bold green]Entropy trace:[/] {trace_str}")

    if result.alternatives:
        table = Table(title="Alternatives")
        table.add_column("Label")
        table.add_column("Probability")
        for alt in result.alternatives:
            table.add_row(alt["label"], f"{alt['probability']:.4f}")
        console.print(table)


@app.command()
def demo() -> None:
    """Run a demonstration of all quantum agent capabilities."""
    console.print(
        Panel(
            "[bold]Demonstrating AI Agent Quantum capabilities[/]\n\n"
            "1. Quantum State Superposition\n"
            "2. Interference Filtering\n"
            "3. Quantum Annealing\n"
            "4. Multi-Agent Orchestration\n"
            "5. Quantum Memory",
            title="[bold magenta]AI Agent Quantum Demo[/]",
            border_style="bright_magenta",
        )
    )

    _demo_superposition()
    _demo_interference()
    _demo_annealing()
    _demo_memory()
    _demo_orchestrator()


def _demo_superposition() -> None:
    console.print("\n[bold cyan]━━━ 1. Quantum State Superposition ━━━[/]")
    state = QuantumState(
        [
            Hypothesis("REST API", {"type": "web"}),
            Hypothesis("GraphQL", {"type": "web"}),
            Hypothesis("gRPC", {"type": "rpc"}),
            Hypothesis("WebSocket", {"type": "realtime"}),
        ]
    )
    console.print(f"Initial state: {state}")
    console.print(f"Entropy: {state.entropy():.4f}")

    state.amplify("GraphQL", 2.0)
    console.print(f"After amplifying GraphQL: {state.probabilities()}")

    state.dampen("gRPC", 0.3)
    console.print(f"After dampening gRPC: {state.probabilities()}")

    result = state.measure()
    console.print(f"[bold green]Collapsed to:[/] {result.label}")


def _demo_interference() -> None:
    from quantum_agent.core.interference import interference_filter

    console.print("\n[bold cyan]━━━ 2. Quantum Interference ━━━[/]")
    state = QuantumState(
        [
            Hypothesis("solution_A", {"quality": "high"}),
            Hypothesis("solution_B", {"quality": "medium"}),
            Hypothesis("solution_C", {"quality": "low"}),
        ]
    )

    evaluators = [
        lambda h: 0.9 if "A" in h.label else (0.5 if "B" in h.label else 0.1),
        lambda h: 0.8 if "A" in h.label else (0.6 if "B" in h.label else 0.2),
    ]

    console.print(f"Before: {state.probabilities()}")
    new_state = interference_filter(state, evaluators)
    console.print(f"After interference: {new_state.probabilities()}")
    console.print("[bold green]High-quality solutions amplified, low-quality dampened[/]")


def _demo_annealing() -> None:
    from quantum_agent.reasoning.annealing import quantum_anneal

    console.print("\n[bold cyan]━━━ 3. Quantum Annealing ━━━[/]")

    labels = ["plan_A", "plan_B", "plan_C", "plan_D"]
    data_map = {
        "plan_A": {"cost": 10, "quality": 0.9},
        "plan_B": {"cost": 5, "quality": 0.7},
        "plan_C": {"cost": 20, "quality": 0.95},
        "plan_D": {"cost": 8, "quality": 0.85},
    }

    def energy(label: str, data: dict) -> float:
        return data.get("cost", 10) / 20.0 - data.get("quality", 0.5)

    best = quantum_anneal(labels, data_map, energy, iterations=200)
    console.print(f"[bold green]Best plan found:[/] {best} (data={data_map[best]})")


def _demo_memory() -> None:
    from quantum_agent.memory.quantum_memory import QuantumMemory

    console.print("\n[bold cyan]━━━ 4. Quantum Memory ━━━[/]")
    mem = QuantumMemory()
    mem.store("python web framework", "Use FastAPI for async APIs", tags=["python", "web"])
    mem.store("database choice", "PostgreSQL for relational data", tags=["database", "sql"])
    mem.store("caching strategy", "Redis with TTL-based expiry", tags=["cache", "redis"])

    results = mem.recall("python web api", top_k=2)
    console.print("Query: 'python web api'")
    for r in results:
        console.print(f"  [bold green]{r['key']}[/]: {r['value']} (relevance={r['relevance']:.4f})")


def _demo_orchestrator() -> None:
    console.print("\n[bold cyan]━━━ 5. Multi-Agent Orchestration ━━━[/]")
    orchestrator = QuantumOrchestrator()

    result = asyncio.run(
        orchestrator.run("Design a REST API for a todo application")
    )

    tree = Tree("[bold magenta]Orchestration Result[/]")
    tree.add(f"[bold]Status:[/] {result['status']}")
    tree.add(f"[bold]Cycles:[/] {result['total_cycles']}")

    cycles_branch = tree.add("[bold]Cycle Details[/]")
    for c in result["cycles"]:
        cycles_branch.add(
            f"Cycle {c['cycle']}: plan={c['plan_label']}, "
            f"verdict={c['verdict']}, score={c.get('avg_score', 0):.2f}"
        )

    tree.add(f"[bold]Memory entries:[/] {result['memory_size']}")
    tree.add(f"[bold]Entangled keys:[/] {list(result['entanglement_state'].keys())}")

    console.print(tree)


def _render_result(result: dict, verbose: bool) -> None:
    console.print(
        Panel(
            f"[bold green]Status:[/] {result['status']}\n"
            f"[bold green]Total cycles:[/] {result['total_cycles']}",
            title="[bold green]Result[/]",
            border_style="green",
        )
    )

    if verbose and result.get("cycles"):
        table = Table(title="Cycle Summary")
        table.add_column("Cycle", style="cyan")
        table.add_column("Plan", style="magenta")
        table.add_column("Steps", style="yellow")
        table.add_column("Verdict", style="green")
        table.add_column("Score", style="blue")
        table.add_column("Duration", style="dim")

        for c in result["cycles"]:
            table.add_row(
                str(c["cycle"]),
                c.get("plan_label", "?"),
                str(c.get("steps_completed", 0)),
                c.get("verdict", "?"),
                f"{c.get('avg_score', 0):.2f}",
                f"{c.get('duration', 0):.3f}s",
            )
        console.print(table)

    if verbose:
        console.print("\n[dim]Entanglement state:[/]", result.get("entanglement_state", {}))


if __name__ == "__main__":
    app()
