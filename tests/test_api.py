"""
Integration tests for FastAPI endpoints (/health and /api/v1/levels/simulate).
Uses starlette/fastapi TestClient.
"""

import pytest
from fastapi.testclient import TestClient

from service.api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "ai-game-level-balancing"
    assert "version" in data


def test_simulate_tutorial_level():
    payload = {
        "level_id": "lvl_tutorial_01",
        "grid": [
            [0, 0, 9],
            [0, 0, 0],
        ],
        "episodes": 40,
        "train_episodes": 120,
        "agent_type": "q_learning",
        "start_pos": [0, 0],
    }
    response = client.post("/api/v1/levels/simulate", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["level_id"] == "lvl_tutorial_01"
    assert data["solvable"] is True
    assert data["optimal_steps"] == 2
    assert data["win_rate"] >= 75.0
    assert data["difficulty_tag"] == "TOO_EASY"
    assert data["trap_rate"] == 0.0
    assert "recommendation" in data


def test_simulate_unsolvable_level():
    payload = {
        "level_id": "lvl_impossible",
        "grid": [
            [0, 1, 9],
            [0, 1, 1],
        ],
        "episodes": 20,
        "train_episodes": 50,
        "agent_type": "q_learning",
        "start_pos": [0, 0],
    }
    response = client.post("/api/v1/levels/simulate", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["level_id"] == "lvl_impossible"
    assert data["solvable"] is False
    assert data["optimal_steps"] is None
    assert data["win_rate"] == 0.0
    assert data["difficulty_tag"] == "CHURN_RISK"
    assert "unsolvable" in data["recommendation"].lower()


def test_simulate_validation_errors():
    # Grid with no goal (9)
    resp_no_goal = client.post(
        "/api/v1/levels/simulate",
        json={"level_id": "bad_grid", "grid": [[0, 0], [0, 1]]},
    )
    assert resp_no_goal.status_code == 422

    # Ragged grid (unequal row lengths)
    resp_ragged = client.post(
        "/api/v1/levels/simulate",
        json={"level_id": "ragged", "grid": [[0, 9], [0]]},
    )
    assert resp_ragged.status_code == 422

    # Invalid agent type
    resp_agent = client.post(
        "/api/v1/levels/simulate",
        json={
            "level_id": "bad_agent",
            "grid": [[0, 9]],
            "agent_type": "invalid_bot_type",
        },
    )
    assert resp_agent.status_code == 422

