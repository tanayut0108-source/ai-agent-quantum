# AI Agent Quantum

> A quantum-inspired multi-agent AI framework with superposition reasoning, agent entanglement, and quantum annealing for intelligent decision-making.

```
   ╔═══════════════════════════════════════╗
   ║        AI  AGENT  QUANTUM            ║
   ║   ψ = α|plan⟩ + β|execute⟩ + γ|reflect⟩  ║
   ╚═══════════════════════════════════════╝
```

## Overview

AI Agent Quantum applies concepts from quantum computing — **superposition**, **entanglement**, **interference**, and **quantum annealing** — as powerful models for AI agent reasoning and decision-making.

Instead of following a single chain-of-thought, agents explore multiple hypotheses **simultaneously in superposition**, let evaluators create **constructive/destructive interference**, and use **simulated quantum annealing** to find optimal plans.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   QuantumOrchestrator                    │
│  ┌──────────┐   ┌───────────┐   ┌──────────┐           │
│  │ Planner  │──▶│ Executor  │──▶│  Critic  │           │
│  │  Agent   │   │  Agent    │   │  Agent   │           │
│  └────┬─────┘   └─────┬─────┘   └────┬─────┘           │
│       │               │               │                 │
│       └───────────────┼───────────────┘                 │
│              EntanglementBus                            │
│  ┌──────────────────────────────────────────────┐       │
│  │  Shared State • Events • Implicit Coordination│      │
│  └──────────────────────────────────────────────┘       │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐          │
│  │ Quantum  │  │ Quantum  │  │    Tool      │          │
│  │ Memory   │  │ Reasoner │  │  Registry    │          │
│  └──────────┘  └──────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────┘
```

### Core Concepts

| Quantum Concept | Implementation | Purpose |
|---|---|---|
| **Superposition** | `QuantumState` holds multiple weighted hypotheses | Explore many solutions simultaneously |
| **Measurement** | `state.measure()` collapses to one hypothesis | Make a final decision |
| **Amplitude Amplification** | `state.amplify()` (Grover-style) | Boost promising solutions |
| **Interference** | `interference_filter()` with multi-evaluators | Good ideas reinforce, bad ones cancel |
| **Entanglement** | `EntanglementBus` pub/sub between agents | Implicit agent coordination |
| **Quantum Annealing** | `quantum_anneal()` simulated annealing optimizer | Find optimal plans in discrete spaces |
| **Decoherence** | Memory amplitude decay over time | Naturally forget stale information |

### Agents

- **PlannerAgent** — Generates candidate plans in superposition, uses quantum annealing to find the optimal plan.
- **ExecutorAgent** — Executes plan steps, invoking tools and tracking progress.
- **CriticAgent** — Evaluates execution quality using quantum interference, provides accept/refine/reject verdicts.

### Plan-Execute-Critique Loop

```
                    ┌──────────┐
           ┌───────│  PLANNER  │◀────────┐
           │       │ (anneal)  │         │
           │       └──────────┘         │
           ▼                            │ refine
   ┌──────────────┐              ┌──────────┐
   │   EXECUTOR   │─────────────▶│  CRITIC  │
   │ (step-by-step)│              │(interfere)│
   └──────────────┘              └──────────┘
                                       │
                                  accept │ reject
                                       ▼
                                   [RESULT]
```

## Installation

```bash
# Clone the repository
git clone https://github.com/tanayut0108-source/ai-agent-quantum.git
cd ai-agent-quantum

# Install with pip
pip install -e ".[dev]"
```

## Quick Start

### CLI

```bash
# Run the full orchestrator on a task
quantum-agent run "Design a REST API for a todo application"

# Run quantum reasoning
quantum-agent reason "What's the best database for this use case?" \
  --hypotheses "PostgreSQL,MongoDB,Redis,DynamoDB"

# Run the interactive demo
quantum-agent demo
```

### Python API

```python
import asyncio
from quantum_agent.orchestrator import QuantumOrchestrator

async def main():
    orchestrator = QuantumOrchestrator()
    result = await orchestrator.run("Build a recommendation engine")
    print(f"Status: {result['status']}")
    print(f"Cycles: {result['total_cycles']}")
    for cycle in result['cycles']:
        print(f"  Plan: {cycle['plan_label']}, Verdict: {cycle['verdict']}")

asyncio.run(main())
```

### Quantum Reasoning

```python
from quantum_agent.core import QuantumState, Hypothesis
from quantum_agent.core.interference import interference_filter

# Create a superposition of hypotheses
state = QuantumState([
    Hypothesis("microservices", {"scalability": "high", "complexity": "high"}),
    Hypothesis("monolith", {"scalability": "low", "complexity": "low"}),
    Hypothesis("serverless", {"scalability": "high", "complexity": "medium"}),
])

# Apply interference from multiple evaluators
evaluators = [
    lambda h: 0.9 if h.data.get("scalability") == "high" else 0.3,
    lambda h: 0.8 if h.data.get("complexity") != "high" else 0.4,
]
result = interference_filter(state, evaluators)

# See the probabilities after interference
print(result.probabilities())
# {'serverless': 0.52, 'microservices': 0.31, 'monolith': 0.17}
```

### Memory with Quantum Decoherence

```python
from quantum_agent.memory import QuantumMemory

memory = QuantumMemory(capacity=1000, decay_rate=0.001)
memory.store("python web", "Use FastAPI", tags=["python", "api"])
memory.store("database", "PostgreSQL", tags=["sql"])

# Retrieve with amplitude-weighted relevance
results = memory.recall("python api framework", top_k=3)
for r in results:
    print(f"{r['key']}: {r['value']} (relevance={r['relevance']:.4f})")
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest -v

# Run linter
ruff check .

# Run type checker
mypy quantum_agent/
```

## Project Structure

```
ai-agent-quantum/
├── quantum_agent/
│   ├── core/              # Quantum primitives
│   │   ├── qubit.py       # Hypothesis & QuantumState
│   │   ├── superposition.py # Superposition manager
│   │   ├── entanglement.py  # Agent entanglement bus
│   │   └── interference.py  # Constructive/destructive filtering
│   ├── agents/            # Autonomous agents
│   │   ├── base.py        # BaseQuantumAgent
│   │   ├── planner.py     # Plan generation (annealing)
│   │   ├── executor.py    # Step-by-step execution
│   │   └── critic.py      # Quality evaluation (interference)
│   ├── orchestrator/      # Multi-agent coordination
│   │   └── quantum_orchestrator.py
│   ├── reasoning/         # Reasoning engine
│   │   ├── annealing.py   # Simulated quantum annealing
│   │   └── quantum_reasoning.py
│   ├── memory/            # Quantum-inspired memory
│   │   └── quantum_memory.py
│   ├── tools/             # Pluggable tool system
│   │   ├── registry.py    # Tool registry & decorator
│   │   └── builtin.py     # Built-in tools
│   └── cli.py             # CLI interface
├── tests/                 # Comprehensive test suite
├── pyproject.toml
└── README.md
```

## License

MIT
