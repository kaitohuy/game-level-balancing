"""
Game Balancing & Telemetry Engine.
Runs automated playtesting simulations, extracts telemetry metrics,
and classifies level difficulty for casual game designers.
"""

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from core.a_star_solver import AStarSolver, PathResult
from core.environment import GridPuzzleEnv
from core.q_learning_agent import QLearningAgent
from core.dqn_agent import DQNAgent


class DifficultyTag(str, Enum):
    """Level difficulty classification tags."""
    CHURN_RISK = "CHURN_RISK"      # Frustrating/impassable, high drop-off
    BALANCED = "BALANCED"          # Healthy casual challenge
    TOO_EASY = "TOO_EASY"          # Trivial/tutorial difficulty


@dataclass
class LevelMetrics:
    """Simulation metrics and difficulty classification for a level."""
    level_id: str
    solvable: bool
    optimal_steps: Optional[int]
    win_rate: float
    trap_rate: float
    timeout_rate: float
    avg_steps: float
    difficulty_ratio: Optional[float]
    difficulty_tag: str
    recommendation: str
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MetricsEngine:
    """
    Simulates virtual playthroughs, computes metrics, and classifies difficulty.
    """

    def __init__(
        self,
        churn_win_threshold: float = 30.0,
        churn_trap_threshold: float = 50.0,
        easy_win_threshold: float = 75.0,
        easy_ratio_tolerance: float = 1.35,
    ):
        self.churn_win_threshold = churn_win_threshold
        self.churn_trap_threshold = churn_trap_threshold
        self.easy_win_threshold = easy_win_threshold
        self.easy_ratio_tolerance = easy_ratio_tolerance
        self.a_star_solver = AStarSolver(avoid_traps=True)

    def classify_difficulty(
        self,
        solvable: bool,
        win_rate: float,
        trap_rate: float,
        difficulty_ratio: Optional[float],
    ) -> Tuple[DifficultyTag, str]:
        """
        Classify level difficulty based on spec rules:
        - If W < 30% or T > 50%: CHURN_RISK (Overly frustrating, needs nerf).
        - If 30% <= W <= 75%: BALANCED (Optimal engagement for casual players).
        - If W > 75% and Dr approx 1.0: TOO_EASY (Suitable only for tutorial levels).
        """
        if not solvable:
            return (
                DifficultyTag.CHURN_RISK,
                "Critical: Level is mathematically unsolvable. No path exists to the goal.",
            )

        if win_rate < self.churn_win_threshold or trap_rate > self.churn_trap_threshold:
            reasons = []
            if win_rate < self.churn_win_threshold:
                reasons.append(f"Win rate {win_rate:.1f}% < {self.churn_win_threshold}%")
            if trap_rate > self.churn_trap_threshold:
                reasons.append(f"Trap rate {trap_rate:.1f}% > {self.churn_trap_threshold}%")
            return (
                DifficultyTag.CHURN_RISK,
                f"Churn Risk: Level is overly punishing ({', '.join(reasons)}). Recommend reducing traps or widening corridors.",
            )

        ratio = difficulty_ratio if difficulty_ratio is not None else 1.0

        if win_rate > self.easy_win_threshold and ratio <= self.easy_ratio_tolerance:
            return (
                DifficultyTag.TOO_EASY,
                f"Too Easy: High win rate ({win_rate:.1f}%) and near-optimal pathing (Dr={ratio:.2f}). Suitable primarily for tutorial stages.",
            )

        return (
            DifficultyTag.BALANCED,
            f"Balanced: Optimal engagement curve (Win Rate={win_rate:.1f}%, Trap Rate={trap_rate:.1f}%, Dr={ratio:.2f}).",
        )

    def evaluate_level(
        self,
        grid: Union[List[List[int]], np.ndarray],
        level_id: str = "level_01",
        episodes: int = 100,
        train_episodes: int = 250,
        eval_epsilon: float = 0.05,
        start_pos: Optional[Tuple[int, int]] = None,
        agent_type: str = "q_learning",
        agent_instance: Optional[Union[QLearningAgent, DQNAgent]] = None,
        seed: int = 42,
    ) -> LevelMetrics:
        """
        Run end-to-end evaluation pipeline on a level grid.

        Steps:
        1. Run A* solver for solvability verification and optimal upper bound.
        2. If unsolvable, report immediately.
        3. Train RL agent (simulating trial & error learning).
        4. Run N test playthroughs to collect telemetry.
        5. Compute metrics & output automated difficulty tag.
        """
        grid_arr = np.array(grid, dtype=np.int32)

        # 1. Classical A* Upper Bound
        path_result: PathResult = self.a_star_solver.solve(grid_arr, start_pos=start_pos)

        if not path_result.solvable or path_result.optimal_steps is None:
            tag, rec = self.classify_difficulty(
                solvable=False,
                win_rate=0.0,
                trap_rate=0.0,
                difficulty_ratio=None,
            )
            return LevelMetrics(
                level_id=level_id,
                solvable=False,
                optimal_steps=None,
                win_rate=0.0,
                trap_rate=0.0,
                timeout_rate=100.0,
                avg_steps=0.0,
                difficulty_ratio=None,
                difficulty_tag=tag.value,
                recommendation=rec,
                details={"optimal_path": None},
            )

        optimal_steps = path_result.optimal_steps

        # 2. Simulation Environment
        env = GridPuzzleEnv(grid=grid_arr, start_pos=start_pos)

        # 3. Agent Training & Evaluation
        if agent_instance is not None:
            agent = agent_instance
            eval_results = agent.evaluate(env, episodes=episodes, eval_epsilon=eval_epsilon)
        elif agent_type == "dqn":
            agent = DQNAgent(seed=seed)
            agent.train(env, episodes=train_episodes)
            eval_results = agent.evaluate(env, episodes=episodes, eval_epsilon=eval_epsilon)
        else:
            # Default to Tabular Q-Learning
            agent = QLearningAgent(seed=seed)
            agent.train(env, episodes=train_episodes)
            eval_results = agent.evaluate(env, episodes=episodes, eval_epsilon=eval_epsilon)

        win_rate = float(eval_results["win_rate"])
        trap_rate = float(eval_results["trap_rate"])
        timeout_rate = float(eval_results["timeout_rate"])
        avg_steps = float(eval_results["avg_steps"])

        # 4. Difficulty Ratio Dr = S_bar / S_optimal
        difficulty_ratio = (
            float(avg_steps / optimal_steps) if optimal_steps > 0 else 1.0
        )

        # 5. Automated Difficulty Classification
        tag, rec = self.classify_difficulty(
            solvable=True,
            win_rate=win_rate,
            trap_rate=trap_rate,
            difficulty_ratio=difficulty_ratio,
        )

        return LevelMetrics(
            level_id=level_id,
            solvable=True,
            optimal_steps=optimal_steps,
            win_rate=round(win_rate, 2),
            trap_rate=round(trap_rate, 2),
            timeout_rate=round(timeout_rate, 2),
            avg_steps=round(avg_steps, 2),
            difficulty_ratio=round(difficulty_ratio, 2),
            difficulty_tag=tag.value,
            recommendation=rec,
            details={
                "optimal_path": path_result.path,
                "agent_type": agent_type,
                "episodes": episodes,
            },
        )
