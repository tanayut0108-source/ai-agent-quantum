"""Tests for the FastAPI Web API."""

import pytest
from fastapi.testclient import TestClient

from quantum_agent.api import app


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthAndInfo:
    def test_root(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "AI Agent Quantum API"
        assert "version" in data

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["components"]["orchestrator"] == "ready"


class TestRunEndpoint:
    def test_run_task(self, client):
        resp = client.post("/run", json={
            "task": "Build a calculator",
            "max_cycles": 1,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["task"] == "Build a calculator"
        assert data["total_cycles"] == 1
        assert data["status"] in ("accepted", "refine", "reject", "max_cycles_reached")
        assert len(data["cycles"]) == 1
        assert "final_plan" in data
        assert "duration_seconds" in data

    def test_run_empty_task_rejected(self, client):
        resp = client.post("/run", json={"task": ""})
        assert resp.status_code == 422


class TestReasonEndpoint:
    def test_reason_with_hypotheses(self, client):
        resp = client.post("/reason", json={
            "question": "Best language?",
            "hypotheses": ["Python", "Rust", "Go"],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["question"] == "Best language?"
        assert data["answer"] in ("Python", "Rust", "Go")
        assert data["confidence"] > 0
        assert len(data["entropy_trace"]) >= 2

    def test_reason_defaults(self, client):
        resp = client.post("/reason", json={"question": "Test?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] in ("approach_A", "approach_B", "approach_C")


class TestSuperpositionEndpoint:
    def test_basic_superposition(self, client):
        resp = client.post("/superposition", json={
            "hypotheses": [{"label": "A"}, {"label": "B"}, {"label": "C"}],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["probabilities"]) == 3
        assert data["size"] == 3
        assert data["entropy"] > 0
        assert data["collapsed_to"] is None

    def test_amplify_and_collapse(self, client):
        resp = client.post("/superposition", json={
            "hypotheses": [{"label": "X"}, {"label": "Y"}],
            "amplify": "X",
            "amplify_factor": 5.0,
            "collapse": True,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["collapsed_to"] is not None
        assert data["size"] == 1

    def test_missing_label_rejected(self, client):
        resp = client.post("/superposition", json={
            "hypotheses": [{"name": "no_label"}],
        })
        assert resp.status_code == 422


class TestMemoryEndpoints:
    def test_store_and_recall(self, client):
        client.post("/memory/store", json={
            "key": "test_api_key",
            "value": "test_value",
            "tags": ["test"],
        })
        resp = client.post("/memory/recall", json={
            "query": "test_api_key",
            "top_k": 3,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) >= 1
        assert data["results"][0]["key"] == "test_api_key"

    def test_memory_snapshot(self, client):
        resp = client.get("/memory/snapshot")
        assert resp.status_code == 200
        data = resp.json()
        assert "size" in data
        assert "entries" in data


class TestInterferenceEndpoint:
    def test_interference(self, client):
        resp = client.post("/interference", json={
            "hypotheses": [{"label": "good"}, {"label": "bad"}],
            "scores": [[0.9, 0.1]],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "probabilities" in data
        assert data["probabilities"]["good"] > data["probabilities"]["bad"]
