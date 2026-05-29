"""FastAPI Web API for AI Agent Quantum.

Start the server::

    quantum-agent-api            # default: http://0.0.0.0:8000
    uvicorn quantum_agent.api:app --reload  # dev mode

API docs are available at ``/docs`` (Swagger UI) and ``/redoc``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from quantum_agent import __version__
from quantum_agent.core.interference import interference_filter
from quantum_agent.core.qubit import Hypothesis, QuantumState
from quantum_agent.memory.quantum_memory import QuantumMemory
from quantum_agent.orchestrator.quantum_orchestrator import (
    OrchestratorConfig,
    QuantumOrchestrator,
)
from quantum_agent.reasoning.quantum_reasoning import QuantumReasoner

app = FastAPI(
    title="AI Agent Quantum API",
    description=(
        "Quantum-inspired multi-agent AI framework — "
        "superposition reasoning, agent entanglement, and quantum annealing."
    ),
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class RunRequest(BaseModel):
    task: str = Field(..., description="The task to execute", min_length=1)
    max_cycles: int = Field(3, description="Max plan-execute-critique cycles", ge=1, le=10)
    auto_retry: bool = Field(True, description="Automatically retry on 'refine' verdict")


class RunResponse(BaseModel):
    task: str
    status: str
    total_cycles: int
    cycles: list[dict[str, Any]]
    final_plan: dict[str, Any]
    final_execution: dict[str, Any]
    final_review: dict[str, Any]
    entanglement_state: dict[str, Any]
    memory_size: int
    duration_seconds: float


class ReasonRequest(BaseModel):
    question: str = Field(..., description="The question to reason about", min_length=1)
    hypotheses: list[str] = Field(
        default_factory=lambda: ["approach_A", "approach_B", "approach_C"],
        description="Candidate hypothesis labels",
        min_length=1,
    )
    max_rounds: int = Field(5, description="Max reasoning rounds", ge=1, le=20)


class ReasonResponse(BaseModel):
    question: str
    answer: str
    confidence: float
    alternatives: list[dict[str, Any]]
    reasoning_steps: list[dict[str, Any]]
    entropy_trace: list[float]


class SuperpositionRequest(BaseModel):
    hypotheses: list[dict[str, Any]] = Field(
        ...,
        description="List of hypothesis dicts, each must have a 'label' key",
        min_length=1,
    )
    amplify: str | None = Field(None, description="Label to amplify")
    amplify_factor: float = Field(2.0, description="Amplification factor", gt=0)
    dampen: str | None = Field(None, description="Label to dampen")
    dampen_factor: float = Field(0.5, description="Dampening factor", gt=0, le=1)
    collapse: bool = Field(False, description="Whether to collapse (measure) the state")


class SuperpositionResponse(BaseModel):
    probabilities: dict[str, float]
    entropy: float
    top_3: list[dict[str, Any]]
    collapsed_to: str | None = None
    size: int


class MemoryStoreRequest(BaseModel):
    key: str = Field(..., min_length=1)
    value: Any
    relevance: float = Field(1.0, ge=0, le=1)
    tags: list[str] = Field(default_factory=list)


class MemoryRecallRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=50)


class HealthResponse(BaseModel):
    status: str
    version: str
    components: dict[str, str]


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

_memory = QuantumMemory(capacity=500)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/", tags=["info"], include_in_schema=False)
async def root() -> FileResponse:
    """Serve the web UI."""
    return FileResponse(str(_STATIC_DIR / "index.html"))


@app.get("/api/info", tags=["info"])
async def api_info() -> dict[str, str]:
    return {
        "name": "AI Agent Quantum API",
        "version": __version__,
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["info"])
async def health() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        version=__version__,
        components={
            "orchestrator": "ready",
            "reasoner": "ready",
            "memory": f"{_memory.size()} entries",
        },
    )


@app.post("/run", response_model=RunResponse, tags=["orchestrator"])
async def run_task(request: RunRequest) -> RunResponse:
    """Run the full Plan-Execute-Critique orchestrator on a task."""
    start = time.monotonic()
    config = OrchestratorConfig(
        max_cycles=request.max_cycles,
        auto_retry=request.auto_retry,
    )
    orchestrator = QuantumOrchestrator(config=config)
    result = await orchestrator.run(request.task)
    duration = round(time.monotonic() - start, 3)

    return RunResponse(
        task=request.task,
        status=result["status"],
        total_cycles=result["total_cycles"],
        cycles=result["cycles"],
        final_plan=result["final_plan"],
        final_execution=result["final_execution"],
        final_review=result["final_review"],
        entanglement_state=result["entanglement_state"],
        memory_size=result["memory_size"],
        duration_seconds=duration,
    )


@app.post("/reason", response_model=ReasonResponse, tags=["reasoning"])
async def reason(request: ReasonRequest) -> ReasonResponse:
    """Run quantum reasoning on a question with candidate hypotheses."""
    hyp_list = [{"label": h} for h in request.hypotheses]

    def score_fn(h: Hypothesis) -> float:
        return 0.5 + 0.1 * len(h.label)

    reasoner = QuantumReasoner(max_rounds=request.max_rounds)
    result = reasoner.reason(hyp_list, [score_fn])

    return ReasonResponse(
        question=request.question,
        answer=result.answer,
        confidence=result.confidence,
        alternatives=result.alternatives,
        reasoning_steps=result.reasoning_steps,
        entropy_trace=result.entropy_trace,
    )


@app.post("/superposition", response_model=SuperpositionResponse, tags=["quantum"])
async def superposition(request: SuperpositionRequest) -> SuperpositionResponse:
    """Create and manipulate a quantum state superposition."""
    for h in request.hypotheses:
        if "label" not in h:
            raise HTTPException(status_code=422, detail="Each hypothesis must have a 'label' key")

    hyps = [
        Hypothesis(
            label=h["label"],
            data={k: v for k, v in h.items() if k != "label"},
        )
        for h in request.hypotheses
    ]
    state = QuantumState(hyps)

    if request.amplify:
        state.amplify(request.amplify, request.amplify_factor)

    if request.dampen:
        state.dampen(request.dampen, request.dampen_factor)

    collapsed_to = None
    if request.collapse:
        result = state.measure()
        collapsed_to = result.label

    return SuperpositionResponse(
        probabilities=state.probabilities(),
        entropy=state.entropy(),
        top_3=[
            {"label": h.label, "probability": h.probability}
            for h in state.top_k(3)
        ],
        collapsed_to=collapsed_to,
        size=state.size,
    )


@app.post("/memory/store", tags=["memory"])
async def memory_store(request: MemoryStoreRequest) -> dict[str, str]:
    """Store a value in quantum memory."""
    _memory.store(
        key=request.key,
        value=request.value,
        relevance=request.relevance,
        tags=request.tags,
    )
    return {"status": "stored", "key": request.key, "memory_size": str(_memory.size())}


@app.post("/memory/recall", tags=["memory"])
async def memory_recall(request: MemoryRecallRequest) -> dict[str, Any]:
    """Recall from quantum memory with amplitude-weighted retrieval."""
    results = _memory.recall(request.query, top_k=request.top_k)
    return {"query": request.query, "results": results}


@app.get("/memory/snapshot", tags=["memory"])
async def memory_snapshot() -> dict[str, Any]:
    """Get a snapshot of all memory entries."""
    return {"size": _memory.size(), "entries": _memory.snapshot()}


class InterferenceRequest(BaseModel):
    hypotheses: list[dict[str, Any]] = Field(
        ..., description="List of hypothesis dicts with 'label' keys", min_length=1
    )
    scores: list[list[float]] = Field(
        ...,
        description="Evaluator scores — one inner list per evaluator, one score per hypothesis",
        min_length=1,
    )


@app.post("/interference", tags=["quantum"])
async def interference(request: InterferenceRequest) -> dict[str, Any]:
    """Apply quantum interference to a set of hypotheses."""
    hyps = [
        Hypothesis(label=h.get("label", f"h_{i}"))
        for i, h in enumerate(request.hypotheses)
    ]
    state = QuantumState(hyps)

    evaluators = []
    for score_list in request.scores:
        score_map = {
            hyps[i].label: score_list[i]
            for i in range(min(len(hyps), len(score_list)))
        }
        evaluators.append(lambda h, sm=score_map: sm.get(h.label, 0.5))

    result = interference_filter(state, evaluators)
    return {
        "probabilities": result.probabilities(),
        "top_3": [
            {"label": h.label, "probability": h.probability}
            for h in result.top_k(3)
        ],
        "entropy": result.entropy(),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def start() -> None:
    """Launch the API server (used by the ``quantum-agent-api`` script)."""
    uvicorn.run(
        "quantum_agent.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    start()
