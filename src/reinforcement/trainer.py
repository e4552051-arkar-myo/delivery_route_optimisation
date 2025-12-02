"""
Training loop and helper functions for the Q-learning agent.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

from src.reinforcement.environment import GridEnvironment
from src.reinforcement.q_learning import QLearningAgent


def train_q_learning(
    graph: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    start: Tuple[int, int],
    goal: Tuple[int, int],
    obstacles: Set[Tuple[int, int]],
    rows: int,
    cols: int,
    episodes: int = 500,
    max_steps_per_episode: int = 500,
) -> Tuple[QLearningAgent, List[float]]:
    """
    Train a Q-learning agent on the grid world.

    Args:
        graph: Grid adjacency list (not used directly, but kept for symmetry).
        start: Start cell (row, col).
        goal: Goal cell (row, col).
        obstacles: Set of blocked cells.
        rows: Number of rows in grid.
        cols: Number of columns in grid.
        episodes: Number of training episodes.
        max_steps_per_episode: Maximum steps per episode.

    Returns:
        agent: Trained QLearningAgent.
        episode_rewards: List of total reward per episode.
    """
    env = GridEnvironment(
        graph=graph,
        start=start,
        goal=goal,
        obstacles=obstacles,
        rows=rows,
        cols=cols,
    )
    agent = QLearningAgent(rows=rows, cols=cols)

    episode_rewards: List[float] = []

    for _ in range(episodes):
        state = env.reset()
        total_reward = 0.0

        for _ in range(max_steps_per_episode):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)

            state = next_state
            total_reward += reward

            if done:
                break

        episode_rewards.append(total_reward)

    return agent, episode_rewards


def extract_path(
    agent: QLearningAgent,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    obstacles: Set[Tuple[int, int]],
    rows: int,
    cols: int,
    max_steps: int = 500,
) -> List[Tuple[int, int]]:
    """
    Use the trained Q-table to greedily follow the best actions from start to goal.

    Args:
        agent: Trained QLearningAgent.
        start: Start cell (row, col).
        goal: Goal cell (row, col).
        obstacles: Set of blocked cells.
        rows: Number of rows in the grid.
        cols: Number of columns in the grid.
        max_steps: Safety cap on the number of steps.

    Returns:
        path: Sequence of visited cells from start towards goal.
    """
    path: List[Tuple[int, int]] = [start]
    current = start

    for _ in range(max_steps):
        if current == goal:
            break

        r, c = current
        # Greedy policy: choose the best action (no ε exploration)
        action = int(agent.q_table[r, c].argmax())

        moves = {
            0: (r - 1, c),
            1: (r + 1, c),
            2: (r, c - 1),
            3: (r, c + 1),
        }
        nxt = moves[action]

        if nxt == current or nxt in obstacles:
            break

        path.append(nxt)
        current = nxt

    return path