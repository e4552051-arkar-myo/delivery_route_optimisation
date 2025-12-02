"""
Simple grid-world environment wrapper for Q-learning.

The environment exposes:
- reset() -> starting state
- step(action) -> (next_state, reward, done)

Actions:
    0: up
    1: down
    2: left
    3: right
"""

from __future__ import annotations

from typing import Tuple, Set, Dict, List


class GridEnvironment:
    def __init__(
        self,
        graph: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
        start: Tuple[int, int],
        goal: Tuple[int, int],
        obstacles: Set[Tuple[int, int]],
        rows: int,
        cols: int,
    ) -> None:
        self.graph = graph
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.rows = rows
        self.cols = cols
        self.position: Tuple[int, int] = start

    def reset(self) -> Tuple[int, int]:
        """Reset the agent to the start state."""
        self.position = self.start
        return self.position

    def _is_valid(self, pos: Tuple[int, int]) -> bool:
        r, c = pos
        return (
            0 <= r < self.rows
            and 0 <= c < self.cols
            and pos not in self.obstacles
        )

    def step(self, action: int) -> tuple[Tuple[int, int], float, bool]:
        """
        Take an action in the grid.

        Returns:
            next_state, reward, done
        """
        r, c = self.position
        moves = {
            0: (r - 1, c),  # up
            1: (r + 1, c),  # down
            2: (r, c - 1),  # left
            3: (r, c + 1),  # right
        }

        next_pos = moves[action]

        # Invalid move: penalise and stay in place
        if not self._is_valid(next_pos):
            reward = -10.0
            done = False
            return self.position, reward, done

        # Valid move
        self.position = next_pos

        if next_pos == self.goal:
            return next_pos, 100.0, True

        # Small step penalty to encourage shorter paths
        return next_pos, -1.0, False