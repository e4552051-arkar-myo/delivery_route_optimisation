"""
PyMaze-style grid map visualisation.

This module provides a clear, human-readable visualisation of:
- grid structure
- obstacles
- explored nodes
- final path
- start and goal positions

Used to complement heatmap visualisations.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def draw_grid_map(
    rows: int,
    cols: int,
    obstacles: set,
    start: tuple,
    goal: tuple,
    path: list | None = None,
    explored: set | None = None,
    filename: str = "grid_map.png",
) -> None:
    """
    Draw a PyMaze-style grid map and save it as an image.

    Colours:
    - White  : free cell
    - Black  : obstacle
    - Yellow : explored nodes
    - Red    : final path
    - Blue   : start
    - Green  : goal
    """

    # Base grid (0 = free, 1 = obstacle)
    grid = np.zeros((rows, cols))
    for r, c in obstacles:
        grid[r, c] = 1

    fig, ax = plt.subplots(figsize=(cols / 2, rows / 2))

    # Draw free cells and obstacles
    ax.imshow(grid, cmap="gray_r")

    # Draw explored nodes
    if explored:
        for r, c in explored:
            ax.add_patch(
                plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="yellow", alpha=0.4)
            )

    # Draw final path
    if path:
        pr = [p[0] for p in path]
        pc = [p[1] for p in path]
        ax.plot(pc, pr, color="red", linewidth=2)

    # Draw start and goal
    ax.scatter(start[1], start[0], color="blue", s=100, label="Start")
    ax.scatter(goal[1], goal[0], color="green", s=100, label="Goal")

    # Grid lines
    ax.set_xticks(np.arange(-0.5, cols, 1))
    ax.set_yticks(np.arange(-0.5, rows, 1))
    ax.grid(color="black", linewidth=0.5)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)

    ax.legend(loc="upper right")

    # Save figure
    out_dir = Path("data/plots/maps")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / filename
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()