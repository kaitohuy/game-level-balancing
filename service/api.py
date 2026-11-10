"""
FastAPI REST API Service for Automated Game Level Balancing.
Exposes simulation endpoints and level metrics for game designers and CI/CD pipelines.
"""

import logging
from typing import Any, Dict
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from core.metrics_engine import MetricsEngine
from service.schemas import (
    HealthCheckResponse,
    LevelSimulateRequest,
    LevelSimulateResponse,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("level-balancing-api")

app = FastAPI(
    title="AI Game Level Balancing API",
    description=(
        "Automated simulation pipeline for mobile puzzle/casual game studios. "
        "Leverages A* Pathfinding and Reinforcement Learning (Q-Learning / DQN) "
        "to simulate thousands of virtual playthroughs, predict player retention, "
        "and classify level difficulty (TOO_EASY, BALANCED, CHURN_RISK)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for internal studio dashboards and tools
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

metrics_engine = MetricsEngine()


@app.get(
    "/health",
    response_model=HealthCheckResponse,
    tags=["System"],
    summary="Health check endpoint",
)
async def health_check() -> HealthCheckResponse:
    """Check service availability."""
    return HealthCheckResponse(
        status="ok",
        service="ai-game-level-balancing",
        version="1.0.0",
    )


@app.post(
    "/api/v1/levels/simulate",
    response_model=LevelSimulateResponse,
    status_code=status.HTTP_200_OK,
    tags=["Simulation"],
    summary="Simulate and balance a game level",
)
async def simulate_level(request: LevelSimulateRequest) -> LevelSimulateResponse:
    """
    Run automated playtesting simulation for a given level grid.

    - **A* Sanity Check**: Verifies mathematical solvability and theoretical upper bound.
    - **RL Agent Playtesting**: Runs trial-and-error simulation (Q-Learning or DQN).
    - **Telemetry Extraction**: Evaluates win rate, trap trigger rate, and average steps.
    - **Difficulty Classification**: Returns `TOO_EASY`, `BALANCED`, or `CHURN_RISK`.
    """
    logger.info(
        f"Received simulation request for level '{request.level_id}' "
        f"({len(request.grid)}x{len(request.grid[0])}), episodes={request.episodes}, agent={request.agent_type}"
    )

    try:
        metrics = metrics_engine.evaluate_level(
            grid=request.grid,
            level_id=request.level_id,
            episodes=request.episodes,
            train_episodes=request.train_episodes,
            eval_epsilon=request.eval_epsilon,
            start_pos=request.start_pos,
            agent_type=request.agent_type,
        )

        logger.info(
            f"Simulation completed for '{request.level_id}': "
            f"tag={metrics.difficulty_tag}, win_rate={metrics.win_rate}%, "
            f"trap_rate={metrics.trap_rate}%, avg_steps={metrics.avg_steps}"
        )

        return LevelSimulateResponse(
            level_id=metrics.level_id,
            solvable=metrics.solvable,
            optimal_steps=metrics.optimal_steps,
            win_rate=metrics.win_rate,
            avg_steps=metrics.avg_steps,
            trap_rate=metrics.trap_rate,
            timeout_rate=metrics.timeout_rate,
            difficulty_ratio=metrics.difficulty_ratio,
            difficulty_tag=metrics.difficulty_tag,
            recommendation=metrics.recommendation,
            details=metrics.details,
        )

    except ValueError as ve:
        logger.warning(f"Validation error processing level '{request.level_id}': {ve}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error simulating level '{request.level_id}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal simulation error: {str(e)}",
        )
