"""
Classical AI Solver using A* Search Algorithm.
Used for sanity-checking level solvability and computing theoretical optimal move bounds.
"""

from dataclasses import dataclass
import heapq
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np

from core.environment import Action, CellType


@dataclass
class PathResult:
    """Result of an A* pathfinding search."""
    solvable: bool
    optimal_steps: Optional[int]
    path: Optional[List[Tuple[int, int]]]
    actions: Optional[List[int]]


class AStarSolver:
    """
    A* Graph Search Solver with Manhattan Distance Heuristic.

    Provides:
    - Solvability verification (flags broken/impossible maps)
    - Theoretical minimum steps required (mathematical upper bound on efficiency)
    """

    def __init__(self, avoid_traps: bool = True):
        self.avoid_traps = avoid_traps

    @staticmethod
    def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
        """Calculate L1 Manhattan distance heuristic between two points."""
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def solve(
        self,
        grid: Union[List[List[int]], np.ndarray],
        start_pos: Optional[Tuple[int, int]] = None,
        goal_pos: Optional[Tuple[int, int]] = None,
    ) -> PathResult:
        """
        Execute A* search on the given grid from start_pos to goal_pos.

        Args:
            grid: 2D array / list of lists containing cell types.
            start_pos: (row, col) starting coordinate. Defaults to (0, 0) or first walkable.
            goal_pos: Target (row, col) coordinate. If None, targets the nearest goal cell (9).

        Returns:
            PathResult with solvability, optimal step count, coordinate path, and action sequence.
        """
        grid_arr = np.array(grid, dtype=np.int32)
        rows, cols = grid_arr.shape

        # Locate goals
        goals = [tuple(g) for g in np.argwhere(grid_arr == CellType.GOAL)]
        if not goals:
            return PathResult(solvable=False, optimal_steps=None, path=None, actions=None)

        # Determine start position
        if start_pos is None:
            if grid_arr[0, 0] != CellType.WALL:
                start_pos = (0, 0)
            else:
                walkable = np.argwhere(grid_arr != CellType.WALL)
                if len(walkable) == 0:
                    return PathResult(solvable=False, optimal_steps=None, path=None, actions=None)
                start_pos = tuple(walkable[0])

        start_r, start_c = start_pos
        if not (0 <= start_r < rows and 0 <= start_c < cols):
            return PathResult(solvable=False, optimal_steps=None, path=None, actions=None)
        if grid_arr[start_r, start_c] == CellType.WALL:
            return PathResult(solvable=False, optimal_steps=None, path=None, actions=None)

        # Check if already at goal
        if start_pos in goals:
            return PathResult(solvable=True, optimal_steps=0, path=[start_pos], actions=[])

        # Priority Queue entries: (f_score, g_score, tie_breaker, current_node)
        # Directions and action mapping
        moves = [
            ((-1, 0), Action.UP),
            ((1, 0), Action.DOWN),
            ((0, -1), Action.LEFT),
            ((0, 1), Action.RIGHT),
        ]

        def min_heuristic(pos: Tuple[int, int]) -> int:
            if goal_pos is not None:
                return self.manhattan_distance(pos, goal_pos)
            return min(self.manhattan_distance(pos, g) for g in goals)

        open_set: List[Tuple[int, int, int, Tuple[int, int]]] = []
        counter = 0
        h_start = min_heuristic(start_pos)
        heapq.heappush(open_set, (h_start, 0, counter, start_pos))

        came_from: Dict[Tuple[int, int], Tuple[Tuple[int, int], int]] = {}
        g_scores: Dict[Tuple[int, int], int] = {start_pos: 0}
        closed_set: Set[Tuple[int, int]] = set()

        reached_goal: Optional[Tuple[int, int]] = None

        while open_set:
            f, g, _, current = heapq.heappop(open_set)

            if current in closed_set:
                continue
            closed_set.add(current)

            # Check goal condition
            if (goal_pos is not None and current == goal_pos) or (goal_pos is None and current in goals):
                reached_goal = current
                break

            r, c = current
            for (dr, dc), action in moves:
                nr, nc = r + dr, c + dc
                neighbor = (nr, nc)

                # Boundary check
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue

                cell_val = grid_arr[nr, nc]

                # Impassable check
                if cell_val == CellType.WALL:
                    continue
                if self.avoid_traps and cell_val == CellType.TRAP:
                    continue

                tentative_g = g + 1
                if neighbor in g_scores and tentative_g >= g_scores[neighbor]:
                    continue

                g_scores[neighbor] = tentative_g
                f_score = tentative_g + min_heuristic(neighbor)
                counter += 1
                came_from[neighbor] = (current, int(action))
                heapq.heappush(open_set, (f_score, tentative_g, counter, neighbor))

        if reached_goal is None:
            return PathResult(solvable=False, optimal_steps=None, path=None, actions=None)

        # Reconstruct path and actions
        path = [reached_goal]
        actions = []
        curr = reached_goal
        while curr in came_from:
            prev_node, act = came_from[curr]
            path.append(prev_node)
            actions.append(act)
            curr = prev_node

        path.reverse()
        actions.reverse()
        optimal_steps = len(actions)

        return PathResult(
            solvable=True,
            optimal_steps=optimal_steps,
            path=path,
            actions=actions,
        )

