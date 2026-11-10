"""
Pydantic Request and Response Schemas for the Simulation API.
"""

from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, field_validator


class LevelSimulateRequest(BaseModel):
    """Request payload for level balancing simulation."""
    level_id: str = Field(
        ...,
        description="Unique identifier for the game level",
        examples=["level_045_ice_cavern"],
    )
    grid: List[List[int]] = Field(
        ...,
        description="2D grid matrix: 0=Path, 1=Wall, 2=Trap, 9=Goal",
        examples=[
            [
                [0, 0, 0, 1, 9],
                [0, 2, 0, 1, 0],
                [0, 0, 0, 0, 0],
            ]
        ],
    )
    episodes: int = Field(
        default=100,
        ge=1,
        le=2000,
        description="Number of virtual playthrough episodes to evaluate",
    )
    train_episodes: int = Field(
        default=250,
        ge=10,
        le=5000,
        description="Number of exploration/training episodes for the agent",
    )
    agent_type: str = Field(
        default="q_learning",
        description="AI bot type: 'q_learning' (tabular) or 'dqn' (deep q-network)",
    )
    eval_epsilon: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Exploratory noise rate during evaluation (models human trial/error). Default: 0.05",
    )
    start_pos: Optional[Tuple[int, int]] = Field(
        default=None,
        description="Optional starting coordinate (row, col). Defaults to (0, 0)",
    )

    @field_validator("grid")
    @classmethod
    def validate_grid(cls, v: List[List[int]]) -> List[List[int]]:
        if not v or len(v) == 0:
            raise ValueError("Grid cannot be empty.")
        if not isinstance(v[0], list) or len(v[0]) == 0:
            raise ValueError("Grid rows cannot be empty.")

        num_cols = len(v[0])
        has_goal = False
        allowed_cells = {0, 1, 2, 9}

        for r_idx, row in enumerate(v):
            if len(row) != num_cols:
                raise ValueError(
                    f"Grid must be rectangular. Row {r_idx} has length {len(row)}, expected {num_cols}."
                )
            for c_idx, cell in enumerate(row):
                if cell not in allowed_cells:
                    raise ValueError(
                        f"Invalid cell value {cell} at ({r_idx}, {c_idx}). Allowed: {allowed_cells}"
                    )
                if cell == 9:
                    has_goal = True

        if not has_goal:
            raise ValueError("Grid must contain at least one goal cell (value 9).")

        return v

    @field_validator("agent_type")
    @classmethod
    def validate_agent_type(cls, v: str) -> str:
        valid_agents = {"q_learning", "dqn"}
        if v.lower() not in valid_agents:
            raise ValueError(f"agent_type must be one of {valid_agents}, got '{v}'")
        return v.lower()


class LevelSimulateResponse(BaseModel):
    """Response payload containing simulation telemetry and balancing recommendation."""
    level_id: str
    solvable: bool
    optimal_steps: Optional[int]
    win_rate: float
    avg_steps: float
    trap_rate: float
    timeout_rate: float
    difficulty_ratio: Optional[float]
    difficulty_tag: str
    recommendation: str
    details: Optional[Dict[str, Any]] = None


class HealthCheckResponse(BaseModel):
    """Health check response schema."""
    status: str
    service: str
    version: str
