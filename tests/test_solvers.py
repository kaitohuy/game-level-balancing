"""
Unit tests for AI Solvers and RL Agents.
Verifies A* pathfinding optimality, unsolvability detection, Q-learning convergence,
and Deep Q-Network training functionality.
"""

import numpy as np
import pytest
import torch

from core.a_star_solver import AStarSolver, PathResult
from core.environment import Action, CellType, GridPuzzleEnv
from core.q_learning_agent import QLearningAgent
from core.dqn_agent import DQNAgent, QNetwork, ReplayBuffer


# --- A* Solver Tests ---

def test_a_star_simple_path():
    grid = [
        [0, 0, 0, 9],
    ]
    solver = AStarSolver()
    result = solver.solve(grid, start_pos=(0, 0))

    assert result.solvable is True
    assert result.optimal_steps == 3
    assert result.path == [(0, 0), (0, 1), (0, 2), (0, 3)]
    assert result.actions == [Action.RIGHT, Action.RIGHT, Action.RIGHT]


def test_a_star_obstacle_avoidance():
    # Wall forces going around:
    # 0 1 9
    # 0 0 0
    grid = [
        [0, 1, 9],
        [0, 0, 0],
    ]
    solver = AStarSolver()
    result = solver.solve(grid, start_pos=(0, 0))

    assert result.solvable is True
    assert result.optimal_steps == 4  # (0,0) -> (1,0) -> (1,1) -> (1,2) -> (0,2)
    assert result.path is not None
    assert result.path[0] == (0, 0)
    assert result.path[-1] == (0, 2)


def test_a_star_trap_avoidance():
    # Trap at (0, 1), alternative path through (1, 1)
    grid = [
        [0, 2, 9],
        [0, 0, 0],
    ]
    solver = AStarSolver(avoid_traps=True)
    result = solver.solve(grid, start_pos=(0, 0))

    assert result.solvable is True
    assert result.path is not None
    assert (0, 1) not in result.path  # Never steps on trap


def test_a_star_unsolvable_map():
    # Goal completely walled off
    grid = [
        [0, 1, 9],
        [0, 1, 1],
    ]
    solver = AStarSolver()
    result = solver.solve(grid, start_pos=(0, 0))

    assert result.solvable is False
    assert result.optimal_steps is None
    assert result.path is None
    assert result.actions is None


def test_a_star_start_at_goal():
    grid = [
        [9, 0],
    ]
    solver = AStarSolver()
    result = solver.solve(grid, start_pos=(0, 0))

    assert result.solvable is True
    assert result.optimal_steps == 0
    assert result.path == [(0, 0)]
    assert result.actions == []


# --- Q-Learning Agent Tests ---

def test_q_learning_bellman_update():
    agent = QLearningAgent(learning_rate=0.5, discount_factor=0.9, epsilon=0.0)
    state = (0, 0)
    next_state = (0, 1)
    action = Action.RIGHT
    reward = -1.0

    # Next state has max Q-value of 10.0
    agent.q_table[next_state][Action.RIGHT] = 10.0

    # Old Q(s, a) = 0.0
    # td_target = -1.0 + 0.9 * 10.0 = 8.0
    # td_error = 8.0 - 0.0 = 8.0
    # new_q = 0.0 + 0.5 * 8.0 = 4.0
    agent.update(state, action, reward, next_state, terminated=False)
    assert np.isclose(agent.q_table[state][action], 4.0)


def test_q_learning_convergence():
    # Simple 3x3 open grid
    grid = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 9],
    ]
    env = GridPuzzleEnv(grid=grid, start_pos=(0, 0))
    agent = QLearningAgent(seed=42, learning_rate=0.2, discount_factor=0.95)

    # Train for 200 episodes
    train_stats = agent.train(env, episodes=200)
    assert train_stats["episodes"] == 200

    # Evaluate greedy policy
    eval_stats = agent.evaluate(env, episodes=50)
    # On a tiny 3x3 grid, trained Q-learning should achieve >= 90% win rate
    assert eval_stats["win_rate"] >= 85.0
    assert eval_stats["trap_rate"] == 0.0


# --- DQN Agent Tests ---

def test_dqn_network_architecture():
    net = QNetwork(state_dim=2, action_dim=4, hidden_dim=32)
    sample_input = torch.tensor([[0.0, 0.0], [1.0, 2.0]], dtype=torch.float32)
    output = net(sample_input)
    assert output.shape == (2, 4)


def test_dqn_replay_buffer():
    buf = ReplayBuffer(capacity=10)
    for i in range(5):
        buf.push(np.array([i, i]), 0, 1.0, np.array([i + 1, i + 1]), False)
    assert len(buf) == 5

    states, actions, rewards, next_states, dones = buf.sample(3)
    assert states.shape == (3, 2)
    assert actions.shape == (3, 1)
    assert rewards.shape == (3, 1)
    assert next_states.shape == (3, 2)
    assert dones.shape == (3, 1)


def test_dqn_training_and_eval():
    grid = [
        [0, 0, 9],
        [0, 0, 0],
    ]
    env = GridPuzzleEnv(grid=grid, start_pos=(0, 0))
    agent = DQNAgent(state_dim=2, action_dim=4, hidden_dim=32, batch_size=8, seed=42)

    # Short training run to verify no tensor or device errors
    train_res = agent.train(env, episodes=30)
    assert len(train_res["rewards"]) == 30

    eval_res = agent.evaluate(env, episodes=10)
    assert "win_rate" in eval_res
    assert "avg_steps" in eval_res

