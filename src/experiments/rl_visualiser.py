"""
Visualisation utilities for reinforcement learning metrics.

Currently supports plotting the per-episode reward curve for Q-learning.
"""

from __future__ import annotations

from typing import List
from pathlib import Path

import matplotlib.pyplot as plt


def plot_reward_curve(rewards: List[float]) -> None:
    """
    Plot and save the reward curve across episodes.

    Args:
        rewards: Total reward obtained in each training episode.
    """
    out_dir = Path("data/plots/rl")
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title("Q-Learning Reward Curve")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "reward_curve.png")
    plt.close()