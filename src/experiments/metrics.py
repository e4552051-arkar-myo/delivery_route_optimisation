# src/experiments/metrics.py

from dataclasses import dataclass

@dataclass
class SearchMetrics:
    algorithm: str
    heuristic: str | None
    grid_size: tuple[int, int]
    obstacle_ratio: float
    runtime: float
    expanded: int
    path_cost: float