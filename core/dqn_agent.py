"""
Deep Q-Network (DQN) Agent implemented with PyTorch.
Features Experience Replay and Target Network synchronization for stability.
"""

from collections import deque
import random
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from core.environment import GridPuzzleEnv


class QNetwork(nn.Module):
    """
    Multi-layer Perceptron Q-Network:
    Linear -> ReLU -> Linear -> ReLU -> Linear (Q-Values).
    """

    def __init__(self, state_dim: int = 2, action_dim: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """Experience Replay Buffer backed by collections.deque."""

    def __init__(self, capacity: int = 10000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)

        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64).unsqueeze(1),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network Agent with Target Network and Replay Buffer.
    """

    def __init__(
        self,
        state_dim: int = 2,
        action_dim: int = 4,
        hidden_dim: int = 64,
        learning_rate: float = 1e-3,
        discount_factor: float = 0.95,
        epsilon: float = 0.3,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 10000,
        batch_size: int = 32,
        target_update_freq: int = 10,
        seed: Optional[int] = 42,
        device: Optional[str] = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Networks
        self.policy_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.SmoothL1Loss()
        self.memory = ReplayBuffer(buffer_capacity)
        self.total_steps = 0

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        """Select action using epsilon-greedy strategy."""
        if not greedy and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return int(q_values.argmax(dim=1).item())

    def update(self) -> Optional[float]:
        """Sample mini-batch from replay buffer and update policy network."""
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Q(s, a)
        current_q = self.policy_net(states).gather(1, actions)

        # Target Q = r + gamma * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            target_q = rewards + self.gamma * next_q * (1.0 - dones)

        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return float(loss.item())

    def sync_target_network(self) -> None:
        """Copy weights from policy net to target net."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self) -> None:
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(
        self,
        env: GridPuzzleEnv,
        episodes: int = 200,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Train DQN agent on the provided environment."""
        episode_rewards: List[float] = []
        episode_lengths: List[int] = []
        losses: List[float] = []
        outcomes: List[str] = []

        for ep in range(episodes):
            obs, _ = env.reset()
            total_reward = 0.0
            done = False
            last_outcome = "timeout"

            while not done:
                action = self.select_action(obs, greedy=False)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                last_outcome = info.get("outcome", "unknown")

                self.memory.push(obs, action, reward, next_obs, terminated)
                obs = next_obs
                total_reward += reward
                self.total_steps += 1

                loss_val = self.update()
                if loss_val is not None:
                    losses.append(loss_val)

            self.decay_epsilon()
            if (ep + 1) % self.target_update_freq == 0:
                self.sync_target_network()

            episode_rewards.append(total_reward)
            episode_lengths.append(env.step_count)
            outcomes.append(last_outcome)

            if verbose and (ep + 1) % 25 == 0:
                recent_wins = outcomes[-25:].count("goal")
                print(f"[DQN] Ep {ep+1}/{episodes} | Epsilon: {self.epsilon:.3f} | Win Rate: {recent_wins/25*100:.1f}%")

        return {
            "episodes": episodes,
            "rewards": episode_rewards,
            "lengths": episode_lengths,
            "losses": losses,
            "outcomes": outcomes,
            "final_epsilon": self.epsilon,
        }

    def evaluate(
        self,
        env: GridPuzzleEnv,
        episodes: int = 100,
        eval_epsilon: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Evaluate DQN agent performance.

        Args:
            env: GridPuzzleEnv instance
            episodes: Number of test playthroughs
            eval_epsilon: Human exploratory variance (defaults to 0.05).
                         Set to 0.0 for pure greedy deterministic evaluation.
        """
        wins = 0
        traps = 0
        timeouts = 0
        step_counts: List[int] = []
        outcomes: List[str] = []

        for _ in range(episodes):
            obs, _ = env.reset()
            done = False
            last_outcome = "timeout"

            while not done:
                if eval_epsilon > 0.0 and random.random() < eval_epsilon:
                    action = random.randint(0, self.action_dim - 1)
                else:
                    action = self.select_action(obs, greedy=True)
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                last_outcome = info.get("outcome", "unknown")

            step_counts.append(env.step_count)
            outcomes.append(last_outcome)
            if last_outcome == "goal":
                wins += 1
            elif last_outcome == "trap":
                traps += 1
            else:
                timeouts += 1

        win_rate = (wins / episodes) * 100.0
        trap_rate = (traps / episodes) * 100.0
        timeout_rate = (timeouts / episodes) * 100.0
        avg_steps = float(np.mean(step_counts)) if step_counts else 0.0

        return {
            "episodes": episodes,
            "wins": wins,
            "traps": traps,
            "timeouts": timeouts,
            "win_rate": win_rate,
            "trap_rate": trap_rate,
            "timeout_rate": timeout_rate,
            "avg_steps": avg_steps,
            "step_counts": step_counts,
            "outcomes": outcomes,
        }
