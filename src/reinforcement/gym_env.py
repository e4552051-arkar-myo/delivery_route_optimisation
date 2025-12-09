import numpy as np
import gymnasium as gym
from gymnasium import spaces


class GridEnv(gym.Env):
    """
    Gymnasium-compatible grid environment for delivery route optimisation.

    Observation: np.array([row, col])  (2 integers)
    Action space: 0 = up, 1 = down, 2 = left, 3 = right
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, rows, cols, obstacles, start, goal):
        super().__init__()

        self.rows = rows
        self.cols = cols
        self.start = start
        self.goal = goal
        self.obstacles = set(obstacles)

        # Actions: 4 directions
        self.action_space = spaces.Discrete(4)

        # Observations: (row, col)
        self.observation_space = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([rows - 1, cols - 1]),
            dtype=np.int32
        )

        # Movement map
        self.moves = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1),   # right
        }

        self.agent_pos = tuple(self.start)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = tuple(self.start)
        obs = np.array(self.agent_pos, dtype=np.int32)
        return obs, {}

    def step(self, action):
        r, c = self.agent_pos
        dr, dc = self.moves[action]
        nr, nc = r + dr, c + dc

        # Out of bounds or obstacle → no movement
        if (
            nr < 0 or nr >= self.rows or
            nc < 0 or nc >= self.cols or
            (nr, nc) in self.obstacles
        ):
            next_pos = (r, c)
        else:
            next_pos = (nr, nc)

        self.agent_pos = next_pos

        # Rewards
        if next_pos == self.goal:
            reward = 100
            terminated = True
        else:
            reward = -1
            terminated = False

        truncated = False  # no time limit here

        obs = np.array(self.agent_pos, dtype=np.int32)
        return obs, reward, terminated, truncated, {}

    def render(self):
        grid = np.full((self.rows, self.cols), ".", dtype=str)
        for (r, c) in self.obstacles:
            grid[r, c] = "#"

        ar, ac = self.agent_pos
        gr, gc = self.goal

        grid[ar, ac] = "A"
        grid[gr, gc] = "G"

        for row in grid:
            print(" ".join(row))