# src/graph/grid_builder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
import random

from typing import Optional, Iterable

Node = Tuple[int, int]
Graph = Dict[Node, List[Tuple[Node, float]]]


@dataclass
class GridConfig:
    rows: int
    cols: int
    obstacle_ratio: float = 0.2
    seed: int | None = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.obstacle_ratio < 1.0):
            raise ValueError("obstacle_ratio must be in [0.0, 1.0).")
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("rows and cols must be positive.")


def _generate_obstacles(cfg: GridConfig, protected: Optional[Iterable[Node]] = None) -> Set[Node]:
    rng = random.Random(cfg.seed)
    protected_set = set(protected) if protected else set()

    total_cells = cfg.rows * cfg.cols
    num_obstacles = int(total_cells * cfg.obstacle_ratio)

    all_cells = [(r, c) for r in range(cfg.rows) for c in range(cfg.cols)]
    # remove protected cells so they can’t become obstacles
    candidate_cells = [cell for cell in all_cells if cell not in protected_set]

    # cap k so sample never fails on very small grids / high ratios
    k = min(num_obstacles, len(candidate_cells))
    obstacles = set(rng.sample(candidate_cells, k=k)) if k > 0 else set()
    return obstacles


def build_grid_graph(cfg: GridConfig, protected: Optional[Iterable[Node]] = None) -> Tuple[Graph, Set[Node]]:
    """
    Build a 4-connected grid graph where each free cell is a node and edges
    connect orthogonal neighbors with unit cost.

    Args:
        cfg: GridConfig defining grid dimensions, obstacle ratio, and RNG seed.
        protected: Optional iterable of node coordinates that must remain free.

    Returns:
        Tuple[Graph, Set[Node]]: adjacency mapping of free nodes to lists of
        (neighbor, cost) pairs, and the set of obstacle nodes.
    """
    obstacles = _generate_obstacles(cfg, protected=protected)
    graph: Graph = {}

    def is_free(r: int, c: int) -> bool:
        return 0 <= r < cfg.rows and 0 <= c < cfg.cols and (r, c) not in obstacles

    directions = [
        (-1, 0),  # up
        (1, 0),   # down
        (0, -1),  # left
        (0, 1),   # right
    ]

    for r in range(cfg.rows):
        for c in range(cfg.cols):
            if not is_free(r, c):
                continue
            node = (r, c)
            neighbors: List[Tuple[Node, float]] = []
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if is_free(nr, nc):
                    neighbors.append(((nr, nc), 1.0))  # unit cost
            graph[node] = neighbors

    return graph, obstacles