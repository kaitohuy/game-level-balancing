"""
Unit tests for GridPuzzleEnv (Gymnasium environment).
Validates game dynamics, boundaries, collision detection, traps, goals, and timeouts.
"""

import numpy as np
import pytest

from core.environment import Action, CellType, GridPuzzleEnv


def test_env_initialization_valid():
    grid = [
        [0, 0, 9],
        [0, 1, 0],
    ]
    env = GridPuzzleEnv(grid=grid, start_pos=(0, 0), max_steps=20)
    obs, info = env.reset()

    assert env.rows == 2
    assert env.cols == 3
    assert env.agent_pos == (0, 0)
    assert np.allclose(obs, [0.0, 0.0])
    assert info["agent_pos"] == (0, 0)
    assert env.max_steps == 20


def test_env_auto_start_pos():
    # If (0, 0) is a wall, auto-detect first walkable
    grid = [
        [1, 0, 9],
        [0, 0, 0],
    ]
    env = GridPuzzleEnv(grid=grid)
    assert env.start_pos == (0, 1)


def test_env_invalid_grids():
    # 1D array
    with pytest.raises(ValueError, match="2-dimensional"):
        GridPuzzleEnv(grid=[0, 1, 9])

    # No goal
    with pytest.raises(ValueError, match="goal cell"):
        GridPuzzleEnv(grid=[[0, 0], [0, 1]])

    # Start on wall
    with pytest.raises(ValueError, match="cannot be placed on a WALL"):
        GridPuzzleEnv(grid=[[1, 9]], start_pos=(0, 0))

    # Start out of bounds
    with pytest.raises(ValueError, match="outside grid bounds"):
        GridPuzzleEnv(grid=[[0, 9]], start_pos=(5, 5))


def test_env_step_movement_and_penalty():
    grid = [
        [0, 0, 9],
        [0, 0, 0],
    ]
    env = GridPuzzleEnv(grid=grid, start_pos=(0, 0))
    env.reset()

    # Move RIGHT
    obs, reward, terminated, truncated, info = env.step(Action.RIGHT)
    assert env.agent_pos == (0, 1)
    assert reward == -1.0
    assert not terminated
    assert not truncated
    assert info["outcome"] == "step"
    assert info["step_count"] == 1


def test_env_wall_collision():
    grid = [
        [0, 1, 9],
    ]
    env = GridPuzzleEnv(grid=grid, start_pos=(0, 0))
    env.reset()

    # Try moving RIGHT into wall
    obs, reward, terminated, truncated, info = env.step(Action.RIGHT)
    assert env.agent_pos == (0, 0)  # Position blocked
    assert reward == -5.0
    assert info["outcome"] == "wall"
    assert not terminated


def test_env_boundary_collision():
    grid = [
        [0, 9],
    ]
    env = GridPuzzleEnv(grid=grid, start_pos=(0, 0))
    env.reset()

    # Try moving UP into outer boundary
    obs, reward, terminated, truncated, info = env.step(Action.UP)
    assert env.agent_pos == (0, 0)
    assert reward == -5.0
    assert info["outcome"] == "wall"
    assert not terminated


def test_env_reach_goal():
    grid = [
        [0, 9],
    ]
    env = GridPuzzleEnv(grid=grid, start_pos=(0, 0))
    env.reset()

    obs, reward, terminated, truncated, info = env.step(Action.RIGHT)
    assert env.agent_pos == (0, 1)
    assert reward == 100.0
    assert terminated
    assert not truncated
    assert info["outcome"] == "goal"


def test_env_trigger_trap():
    grid = [
        [0, 2, 9],
    ]
    env = GridPuzzleEnv(grid=grid, start_pos=(0, 0))
    env.reset()

    obs, reward, terminated, truncated, info = env.step(Action.RIGHT)
    assert env.agent_pos == (0, 1)
    assert reward == -100.0
    assert terminated
    assert not truncated
    assert info["outcome"] == "trap"


def test_env_timeout_truncation():
    grid = [
        [0, 0, 9],
    ]
    # Set max_steps to 2
    env = GridPuzzleEnv(grid=grid, start_pos=(0, 0), max_steps=2)
    env.reset()

    # Step 1: Up (bump wall)
    env.step(Action.UP)
    # Step 2: Up (bump wall again, reaching max_steps)
    obs, reward, terminated, truncated, info = env.step(Action.UP)

    assert not terminated
    assert truncated
    assert info["outcome"] == "timeout"


def test_env_render():
    grid = [
        [0, 1, 2, 9],
    ]
    env = GridPuzzleEnv(grid=grid, start_pos=(0, 0))
    env.reset()
    ascii_grid = env.render()
    assert "P" in ascii_grid
    assert "#" in ascii_grid
    assert "X" in ascii_grid
    assert "G" in ascii_grid

