"""
Heatmap visualisation utilities for search algorithms.

Takes a visited_map (node -> expansion count) and renders it as a 2D heatmap
based on the grid dimensions.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def save_heatmap(
    visited_map: Dict[Tuple[int, int], int],
    rows: int,
    cols: int,
    title: str,
    filename: str,
) -> None:
    """
    Generate and save a heatmap image.

    Args:
        visited_map: Map from (row, col) to visitation count.
        rows: Number of rows in the grid.
        cols: Number of columns in the grid.
        title: Title for the plot.
        filename: Output filename (PNG).
    """
    heat = np.zeros((rows, cols), dtype=float)
    for (r, c), count in visited_map.items():
        if 0 <= r < rows and 0 <= c < cols:
            heat[r, c] = count

    out_dir = Path("data/plots/heatmaps")
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.imshow(heat, cmap="viridis", origin="upper", interpolation="nearest")
    plt.colorbar(label="Expansion count")
    plt.title(title)
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.tight_layout()
    plt.savefig(out_dir / filename)
    plt.close()