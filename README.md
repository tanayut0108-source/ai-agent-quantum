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

## Web UI

AI Agent Quantum includes a **built-in web interface** — open `http://localhost:8000` in your browser or phone.

Features:
- **Run Task** — Execute the Plan-Execute-Critique orchestrator
- **Quantum Reasoning** — Superposition-based reasoning with entropy collapse
- **Superposition** — Create, amplify, dampen, and collapse quantum states
- **Memory** — Store and recall with quantum decoherence
- **Interference** — Apply constructive/destructive interference
- **Health indicator** — Live server status

### Start the Server

```bash
# Option 1: Direct
quantum-agent-api

# Option 2: With auto-reload (development)
uvicorn quantum_agent.api:app --reload --host 0.0.0.0 --port 8000

# Option 3: Docker
docker compose up
```

Open `http://localhost:8000` for the web UI, or `/docs` for Swagger API docs.

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/run` | Run the full Plan-Execute-Critique orchestrator |
| `POST` | `/reason` | Quantum reasoning with hypotheses |
| `POST` | `/superposition` | Create and manipulate quantum states |
| `POST` | `/memory/store` | Store a value in quantum memory |
| `POST` | `/memory/recall` | Recall from memory |
| `GET` | `/memory/snapshot` | View all memory entries |
| `POST` | `/interference` | Apply quantum interference |

### Example API Calls

```bash
# Run the orchestrator
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"task": "Design a REST API", "max_cycles": 2}'

# Quantum reasoning
curl -X POST http://localhost:8000/reason \
  -H "Content-Type: application/json" \
  -d '{"question": "Best database?", "hypotheses": ["PostgreSQL", "MongoDB", "Redis"]}'

# Health check
curl http://localhost:8000/health
```

## Docker

```bash
# Build and run
docker compose up -d

# Or build manually
docker build -t quantum-agent .
docker run -p 8000:8000 quantum-agent
```

## Deploy (Free Options)

Deploy the API so you can access it from your phone or anywhere:

### Railway (Recommended)
```bash
# Install Railway CLI, then:
railway login
railway init
railway up
```

### Render (Recommended — Free, 24/7)
1. Go to [render.com](https://render.com) → New → **Blueprint**
2. Connect your GitHub repo — Render auto-detects `render.yaml`
3. Click **Apply** — done! Your API runs 24/7 for free

Or manually: New Web Service → build: `pip install .` → start: `uvicorn quantum_agent.api:app --host 0.0.0.0 --port $PORT`

### Fly.io
```bash
fly launch
fly deploy
```

## CI/CD

GitHub Actions runs automatically on every push and PR:
- **Lint** — `ruff check`
- **Test** — `pytest` on Python 3.11 & 3.12
- **Docker** — build and health check

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
│   ├── static/            # Web UI frontend
│   │   └── index.html     # Single-page web app
│   ├── api.py             # FastAPI Web API
│   └── cli.py             # CLI interface
├── tests/                 # Comprehensive test suite
├── render.yaml            # Render.com deploy config
├── Dockerfile
├── docker-compose.yml
├── .github/workflows/ci.yml  # GitHub Actions CI
├── pyproject.toml
└── README.md
```

## License

MIT
