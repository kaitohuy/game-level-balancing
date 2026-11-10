"""
Tabular Q-Learning Agent.
Simulates a human player learning a casual puzzle level through trial, error,
and exploratory behavior.
"""

from collections import defaultdict
import random
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from core.environment import Action, GridPuzzleEnv


class QLearningAgent:
    """
    Tabular Q-Learning Agent with epsilon-greedy exploration and decay.

    Formula:
        Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 0.3,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: Optional[int] = 42,
    ):
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.initial_epsilon = epsilon

        # Q-table: state (r, c) -> np.ndarray of shape (4,)
        self.q_table: Dict[Tuple[int, int], np.ndarray] = defaultdict(lambda: np.zeros(4, dtype=np.float32))

        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def get_q_values(self, state: Tuple[int, int]) -> np.ndarray:
        """Retrieve Q-values for all 4 actions in given state."""
        return self.q_table[state]

    def select_action(self, state: Tuple[int, int], greedy: bool = False) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: (row, col) coordinates.
            greedy: If True, always pick greedy action (evaluation mode).
        """
        if not greedy and random.random() < self.epsilon:
            return random.randint(0, 3)

        q_vals = self.get_q_values(state)
        # Random tie-breaking among best actions
        max_val = np.max(q_vals)
        best_actions = np.where(np.isclose(q_vals, max_val))[0]
        return int(random.choice(best_actions))

    def update(
        self,
        state: Tuple[int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int],
        terminated: bool,
    ) -> float:
        """
        Update Q-value using Bellman optimality equation.

        Returns:
            TD error for monitoring convergence.
        """
        old_val = self.q_table[state][action]
        if terminated:
            td_target = reward
        else:
            best_next = np.max(self.get_q_values(next_state))
            td_target = reward + self.gamma * best_next

        td_error = td_target - old_val
        self.q_table[state][action] = old_val + self.alpha * td_error
        return float(td_error)

    def decay_epsilon(self) -> None:
        """Decay exploration rate down to epsilon_min."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(
        self,
        env: GridPuzzleEnv,
        episodes: int = 300,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Train the agent on the given environment for a number of episodes.

        Returns:
            Dictionary with training telemetry (rewards, lengths, win_count).
        """
        episode_rewards: List[float] = []
        episode_lengths: List[int] = []
        outcomes: List[str] = []

        for ep in range(episodes):
            _, info = env.reset(seed=(self.seed + ep if self.seed is not None else None))
            state = env.get_state_tuple()
            total_reward = 0.0
            done = False
            last_outcome = "timeout"

            while not done:
                action = self.select_action(state, greedy=False)
                _, reward, terminated, truncated, step_info = env.step(action)
                next_state = env.get_state_tuple()

                self.update(state, action, reward, next_state, terminated)

                state = next_state
                total_reward += reward
                done = terminated or truncated
                last_outcome = step_info.get("outcome", "unknown")

            self.decay_epsilon()
            episode_rewards.append(total_reward)
            episode_lengths.append(env.step_count)
            outcomes.append(last_outcome)

            if verbose and (ep + 1) % 50 == 0:
                recent_wins = outcomes[-50:].count("goal")
                print(f"[Q-Learning] Ep {ep+1}/{episodes} | Epsilon: {self.epsilon:.3f} | Win Rate: {recent_wins/50*100:.1f}%")

        return {
            "episodes": episodes,
            "rewards": episode_rewards,
            "lengths": episode_lengths,
            "outcomes": outcomes,
            "final_epsilon": self.epsilon,
        }

    def evaluate(
        self,
        env: GridPuzzleEnv,
        episodes: int = 100,
        eval_epsilon: float = 0.05,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate agent performance.

        Args:
            env: GridPuzzleEnv instance
            episodes: Number of evaluation episodes
            eval_epsilon: Human exploratory variance (defaults to 0.05, matching epsilon_min).
                         Set to 0.0 for pure deterministic greedy evaluation.
            seed: Optional seed for evaluation runs

        Returns:
            Dictionary containing win_rate, trap_rate, avg_steps, and details.
        """
        wins = 0
        traps = 0
        timeouts = 0
        step_counts: List[int] = []
        all_outcomes: List[str] = []

        eval_seed = seed if seed is not None else (self.seed + 10000 if self.seed is not None else None)

        for ep in range(episodes):
            run_seed = (eval_seed + ep) if eval_seed is not None else None
            _, _ = env.reset(seed=run_seed)
            state = env.get_state_tuple()
            done = False
            last_outcome = "timeout"

            while not done:
                if eval_epsilon > 0.0 and random.random() < eval_epsilon:
                    action = random.randint(0, 3)
                else:
                    action = self.select_action(state, greedy=True)
                _, _, terminated, truncated, step_info = env.step(action)
                state = env.get_state_tuple()
                done = terminated or truncated
                last_outcome = step_info.get("outcome", "unknown")

            step_counts.append(env.step_count)
            all_outcomes.append(last_outcome)
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
            "outcomes": all_outcomes,
        }
