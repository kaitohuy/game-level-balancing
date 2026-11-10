# 📄 SPECIFICATION: AI-DRIVEN GAME LEVEL BALANCING & TESTING PLATFORM

> **Target Recruiter / Company Context:** Rocket Game Studio (Hanoi, Vietnam)  
> **Target Role:** AI Engineer (Game AI & Automation)  
> **Domain:** Casual / Puzzle / Hybrid-Casual Mobile Games  
> **Author & Candidate:** Nguyen Doan Huy (github.com/kaitohuy)

---

## 1. AGENT ROLE & PERSONA

You are a **Senior Applied AI Engineer** specializing in **Game AI, Reinforcement Learning, and Simulation Systems**. 
Your core competencies:
* **Languages & Core:** Python 3.10+, Object-Oriented Programming (OOP), Clean Architecture, NumPy.
* **AI & Algorithms:** A* Pathfinding (Heuristics, Graph Search), Reinforcement Learning (Gymnasium/OpenAI Gym, Q-Learning, Deep Q-Networks with PyTorch).
* **System & Deployment:** FastAPI, Pydantic, RESTful API design, Docker, Docker Compose, Logging & Metrics Engineering.
* **Game Domain Understanding:** Casual/Puzzle game loops, Level Design Balancing, Player Retention, Churn Risk Analysis.

Your mission is to autonomously design, implement, test, and document an end-to-end **AI Game Level Balancing System** that solves a critical pain point for casual/puzzle game studios.

---

## 2. BUSINESS CONTEXT & PROBLEM STATEMENT

### The Studio's Pain Point:
In mobile game studios like Rocket Game Studio, Game Designers create hundreds or thousands of levels for puzzle/casual games (Match-3, Grid Puzzles, Maze Runners). 
* **Manual QA Playtesting is Slow & Costly:** Human testers cannot thoroughly playtest 1,000+ levels across every edge case.
* **Player Churn Risk:** If Level 45 is too punishing, players get frustrated and uninstall the game. If Level 80 is trivially easy, players get bored.
* **The Solution:** An automated simulation pipeline using **AI Agents (Classical A* + Reinforcement Learning)** that can play thousands of virtual games in seconds, extract telemetry metrics, and automatically classify level difficulty (`TOO_EASY`, `BALANCED`, `CHURN_RISK`).

---

## 3. PROJECT ARCHITECTURE & REPOSITORY STRUCTURE

The repository must follow a clean, modular microservice structure:

```text
ai-game-level-balancing/
├── core/
│   ├── __init__.py
│   ├── environment.py         # Gymnasium-compliant Grid Puzzle Environment
│   ├── a_star_solver.py       # Classical AI: Solvability check & shortest path
│   ├── q_learning_agent.py    # Tabular Q-Learning Bot (Human trial-and-error simulation)
│   ├── dqn_agent.py           # Deep Q-Network (PyTorch) for large-scale state spaces
│   └── metrics_engine.py      # Telemetry collector & difficulty classifier
├── service/
│   ├── __init__.py
│   ├── api.py                 # FastAPI application serving simulation endpoints
│   └── schemas.py             # Pydantic request/response schemas
├── tests/
│   ├── test_environment.py    # Unit tests for game rules and collision
│   ├── test_solvers.py        # Verification of A* and RL convergence
│   └── test_api.py            # API integration tests
├── scripts/
│   └── run_benchmark.py       # CLI script to test sample levels and plot results
├── Dockerfile                 # Container build file
├── docker-compose.yml         # Container orchestration
├── requirements.txt           # Production dependencies
└── README.md                  # Comprehensive documentation with diagrams
```

---

## 4. DETAILED COMPONENT SPECIFICATIONS

### Component 1: Game Environment (`core/environment.py`)
* Must implement a custom environment adhering to the **Gymnasium** API (`reset()`, `step(action)`).
* **Grid State Encoding:**
  * `0`: Walkable path.
  * `1`: Wall / Obstacle (impassable).
  * `2`: Trap / Hazard (terminates episode, penalty: `-100`).
  * `9`: Goal / Level Complete (terminates episode, reward: `+100`).
* **Movement Dynamics:** 4 discrete actions (`0: UP`, `1: DOWN`, `2: LEFT`, `3: RIGHT`).
* **Step Penalty:** Deduct `-1` per step to encourage the agent to find time-optimal routes. Max steps capped at `rows * cols * 2` to prevent infinite loops.

### Component 2: Classical AI Solver (`core/a_star_solver.py`)
* Implements the **A\* Search Algorithm** using a priority queue (`heapq`).
* Uses **Manhattan Distance** heuristic: $h(n) = |x_n - x_{goal}| + |y_n - y_{goal}|$.
* **Primary Role:** 
  1. Act as a **Sanity Checker**: Immediately flag broken/impossible maps where no path to the goal exists ($A^* = \text{None}$).
  2. Provide a **Theoretical Upper Bound** (the mathematically optimal minimum moves required).

### Component 3: Reinforcement Learning Bots
#### A. Tabular Q-Learning (`core/q_learning_agent.py`)
* Implements Bellman equation updates:
  $$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$
* Uses $\epsilon$-greedy exploration with decay: starts at $\epsilon = 0.3$, decays to $\epsilon_{min} = 0.05$.
* **Primary Role:** Simulates a human player learning a level via trial and error, getting caught in deceptive traps that classical search might skip.

#### B. Deep Q-Network (`core/dqn_agent.py`)
* Built with **PyTorch** (`torch.nn`).
* Multi-layer perceptron (Linear $\rightarrow$ ReLU $\rightarrow$ Linear $\rightarrow$ Q-Values).
* Experience Replay Buffer (`collections.deque`) and target synchronization to stabilize training on larger grids.

### Component 4: Game Balancing & Telemetry Engine (`core/metrics_engine.py`)
* Runs $N$ test playthroughs (default: $N = 100$) using the trained agent.
* Computes telemetry:
  * **Win Rate ($W$):** $\frac{\text{Wins}}{N} \times 100\%$
  * **Trap Rate ($T$):** $\frac{\text{Traps Triggered}}{N} \times 100\%$
  * **Average Steps ($\bar{S}$):** Mean steps to finish or terminate.
  * **Difficulty Ratio ($D_r$):** $\frac{\bar{S}}{\text{Optimal Steps from } A^*}$
* **Automated Classification Rules:**
  * If $W < 30\%$ or $T > 50\%$: **`CHURN_RISK`** *(Overly frustrating, needs nerf)*.
  * If $30\% \le W \le 75\%$: **`BALANCED`** *(Optimal engagement for casual players)*.
  * If $W > 75\%$ and $D_r \approx 1.0$: **`TOO_EASY`** *(Suitable only for tutorial levels)*.

### Component 5: REST API Service (`service/api.py`)
* Framework: **FastAPI** with auto-generated OpenAPI/Swagger docs.
* **Endpoint:** `POST /api/v1/levels/simulate`
  * **Request:** JSON containing `level_id`, 2D grid matrix, simulation iteration count (`episodes`).
  * **Response:** JSON containing `solvable`, `optimal_steps`, `win_rate`, `avg_steps`, `trap_rate`, and `difficulty_tag`.
* **Health Check:** `GET /health`

### Component 6: Containerization (`Dockerfile` & `docker-compose.yml`)
* Multi-stage or slim Python 3.10/3.11 base image.
* Standardized port exposure (`8000`).
* Must build and run cleanly with a single command: `docker-compose up --build`.

---

## 5. QUALITY & ENGINEERING CONSTRAINTS (STRICT)

1. **Production-Grade Code Only:**
   * No dummy mocks, no hardcoded random return values.
   * Proper Python type hints (`from typing import List, Tuple, Dict, Optional`).
   * Clean exception handling for malformed grids (e.g., no goal found, empty matrix).
2. **Reproducibility:** Seed all random generators (`random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)`).
3. **Documentation:**
   * Provide a polished `README.md` with a Mermaid architectural diagram, installation steps, and sample cURL requests.
   * Include a runnable CLI demonstration script in `scripts/run_benchmark.py` that visualizes benchmark outputs.

---

## 6. DEFINITION OF DONE (ACCEPTANCE CRITERIA)

The Agent has completed the task successfully when:
- [ ] All Python files are created in the appropriate modular folder structure.
- [ ] Running `pytest` passes 100% of unit tests for the environment, solvers, and API.
- [ ] Running `python scripts/run_benchmark.py` successfully executes simulations on both a Tutorial level and a Hard/Trap level, correctly printing calculated metrics.
- [ ] The FastAPI service boots cleanly and handles `POST /api/v1/levels/simulate` requests via cURL or Swagger UI.
- [ ] The `README.md` file is comprehensive, well-formatted, and ready to be showcased on GitHub.