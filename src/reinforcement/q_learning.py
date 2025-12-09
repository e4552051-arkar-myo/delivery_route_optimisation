"""
Tabular Q-learning agent for the grid environment.
"""

from __future__ import annotations

import random
from typing import Tuple

import numpy as np


class QLearningAgent:
    def __init__(
        self,
        rows: int,
        cols: int,
        learning_rate: float = 0.1,
        discount: float = 0.99,
        epsilon: float = 0.2,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.lr = learning_rate
        self.gamma = discount

        # Exploration parameters
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table: [rows, cols, actions]
        self.q_table = np.zeros((rows, cols, 4), dtype=float)

    def _state_to_idx(self, state: Tuple[int, int]) -> Tuple[int, int]:
        return state

    def choose_action(self, state: Tuple[int, int]) -> int:
        """ε-greedy action selection."""
        r, c = self._state_to_idx(state)
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        return int(np.argmax(self.q_table[r, c]))

    def update(
        self,
        state: Tuple[int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int],
    ) -> None:
        """Q-learning update rule."""
        r, c = self._state_to_idx(state)
        nr, nc = self._state_to_idx(next_state)

        current_q = self.q_table[r, c, action]
        max_next_q = float(np.max(self.q_table[nr, nc]))

        target = reward + self.gamma * max_next_q
        self.q_table[r, c, action] = current_q + self.lr * (target - current_q)

    def decay_epsilon(self) -> None:
        """Decay epsilon after each episode."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min