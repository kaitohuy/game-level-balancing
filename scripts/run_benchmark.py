"""
Benchmark & CLI Demonstration Script for Game Level Balancing Platform.
Tests multiple level layouts (Tutorial, Balanced, Churn Risk, Unsolvable),
prints telemetry tables, and generates visualization charts.
"""

import argparse
import os
import sys
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.environment import GridPuzzleEnv
from core.metrics_engine import MetricsEngine, LevelMetrics


# Pre-defined Benchmark Levels representing realistic studio cases
BENCHMARK_LEVELS: Dict[str, Dict[str, Any]] = {
    "Level_01_Tutorial": {
        "description": "Open straight path with zero traps, easy navigation.",
        "expected": "TOO_EASY",
        "grid": [
            [0, 0, 0, 0, 9],
            [0, 0, 0, 0, 0],
        ],
        "start_pos": (0, 0),
        "train_episodes": 150,
        "eval_epsilon": 0.05,
    },
    "Level_02_Balanced": {
        "description": "Standard casual level with hazard turns and engaging challenge.",
        "expected": "BALANCED",
        "grid": [
            [0, 2, 2, 2, 2, 2],
            [0, 2, 2, 2, 2, 2],
            [0, 0, 0, 0, 0, 2],
            [2, 2, 2, 2, 0, 2],
            [2, 2, 2, 2, 0, 9],
        ],
        "start_pos": (0, 0),
        "train_episodes": 180,
        "eval_epsilon": 0.08,
    },
    "Level_03_Churn_Risk_Traps": {
        "description": "Treacherous gauntlet with deceptive traps causing excessive drop-off.",
        "expected": "CHURN_RISK",
        "grid": [
            [0, 2, 2, 2, 2, 2, 2],
            [0, 2, 2, 2, 2, 2, 2],
            [0, 0, 0, 0, 0, 0, 2],
            [2, 2, 2, 2, 2, 0, 2],
            [2, 2, 2, 2, 2, 0, 2],
            [2, 2, 2, 2, 2, 0, 9],
        ],
        "start_pos": (0, 0),
        "train_episodes": 70,
        "eval_epsilon": 0.08,
    },
    "Level_04_Broken_Unsolvable": {
        "description": "Sanity check: Goal completely blocked off by impenetrable walls.",
        "expected": "CHURN_RISK (Unsolvable)",
        "grid": [
            [0, 0, 1, 1, 9],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 0, 0],
        ],
        "start_pos": (0, 0),
        "train_episodes": 50,
        "eval_epsilon": 0.05,
    },
}


def print_banner() -> None:
    print("=" * 78)
    print(" 🎮 AI-DRIVEN GAME LEVEL BALANCING PLATFORM: BENCHMARK SUITE")
    print(" Rocket Game Studio - Candidate: Nguyen Doan Huy (github.com/kaitohuy)")
    print("=" * 78)


def run_benchmark(
    episodes: int = 100,
    agent_type: str = "q_learning",
    save_plot: bool = True,
    output_image: str = "benchmark_results.png",
) -> List[LevelMetrics]:
    """Execute simulation across all benchmark levels."""
    engine = MetricsEngine()
    results: List[LevelMetrics] = []

    print_banner()
    print(f"Configuration: agent={agent_type}, eval_episodes={episodes}\n")

    for level_id, data in BENCHMARK_LEVELS.items():
        print(f"[*] Simulating '{level_id}' - {data['description']}")
        metrics = engine.evaluate_level(
            grid=data["grid"],
            level_id=level_id,
            episodes=episodes,
            train_episodes=data.get("train_episodes", 200),
            eval_epsilon=data.get("eval_epsilon", 0.05),
            start_pos=data.get("start_pos"),
            agent_type=agent_type,
        )
        results.append(metrics)
        print(f"    -> Result: Solvable={metrics.solvable} | Tag={metrics.difficulty_tag} | "
              f"Win={metrics.win_rate}% | Trap={metrics.trap_rate}% | "
              f"AvgSteps={metrics.avg_steps} | OptSteps={metrics.optimal_steps}")
        print(f"    -> Feedback: {metrics.recommendation}\n")

    # Print summary table
    print_summary_table(results)

    if save_plot:
        plot_benchmark_results(results, output_path=output_image)

    return results


def print_summary_table(results: List[LevelMetrics]) -> None:
    """Format and print benchmark results as a clean table."""
    header = (
        f"{'Level ID':<26} | {'Solv':<4} | {'Opt':<4} | "
        f"{'Win%':<6} | {'Trap%':<6} | {'AvgStep':<7} | {'Dr':<5} | {'Classification':<12}"
    )
    separator = "-" * len(header)
    print(separator)
    print(header)
    print(separator)

    for m in results:
        solv_str = "Yes" if m.solvable else "No"
        opt_str = str(m.optimal_steps) if m.optimal_steps is not None else "N/A"
        dr_str = f"{m.difficulty_ratio:.2f}" if m.difficulty_ratio is not None else "N/A"
        row = (
            f"{m.level_id:<26} | {solv_str:<4} | {opt_str:<4} | "
            f"{m.win_rate:>5.1f}% | {m.trap_rate:>5.1f}% | {m.avg_steps:>7.2f} | {dr_str:>5} | {m.difficulty_tag:<12}"
        )
        print(row)
    print(separator)
    print()


def plot_benchmark_results(results: List[LevelMetrics], output_path: str = "benchmark_results.png") -> None:
    """Generate visualization bar chart for win rate, trap rate, and difficulty ratio."""
    level_names = [r.level_id.replace("Level_", "L") for r in results]
    win_rates = [r.win_rate for r in results]
    trap_rates = [r.trap_rate for r in results]
    difficulty_ratios = [r.difficulty_ratio if r.difficulty_ratio is not None else 0.0 for r in results]

    x = np.arange(len(level_names))
    width = 0.25

    fig, ax1 = plt.subplots(figsize=(11, 5.5))

    rects1 = ax1.bar(x - width, win_rates, width, label="Win Rate (%)", color="#2ca02c")
    rects2 = ax1.bar(x, trap_rates, width, label="Trap Rate (%)", color="#d62728")

    ax1.set_ylabel("Rate (%)", fontsize=11)
    ax1.set_ylim(0, 115)
    ax1.set_xticks(x)
    ax1.set_xticklabels(level_names, rotation=15, ha="right", fontsize=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    rects3 = ax2.bar(x + width, difficulty_ratios, width, label="Difficulty Ratio (Dr)", color="#1f77b4", alpha=0.7)
    ax2.set_ylabel("Difficulty Ratio (Steps / Optimal)", fontsize=11, color="#1f77b4")
    ax2.set_ylim(0, max(max(difficulty_ratios, default=2.0) * 1.3, 4.0))

    # Add tag annotations on top of bars
    for idx, r in enumerate(results):
        color = "#2ca02c" if r.difficulty_tag == "TOO_EASY" else ("#ff7f0e" if r.difficulty_tag == "BALANCED" else "#d62728")
        ax1.text(idx, 102, r.difficulty_tag, ha="center", va="bottom", fontsize=8, fontweight="bold", color=color,
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=color, alpha=0.8))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax1.set_title("Game Level Balancing Simulation Benchmark")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[+] Benchmark chart saved to '{output_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Level Balancing Benchmark Runner")
    parser.add_argument("--episodes", type=int, default=100, help="Playthrough evaluation episodes")
    parser.add_argument("--agent", type=str, default="q_learning", choices=["q_learning", "dqn"], help="RL Agent algorithm")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    parser.add_argument("--output", type=str, default="benchmark_results.png", help="Path to save chart")

    args = parser.parse_args()
    run_benchmark(
        episodes=args.episodes,
        agent_type=args.agent,
        save_plot=not args.no_plot,
        output_image=args.output,
    )
