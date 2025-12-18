"""
Experiment runner for comparing UCS, Greedy, and A* search algorithms.

Runs multiple grid sizes, obstacle ratios, and random seeds, and logs
runtime, expanded nodes, and path cost to a single CSV file.
"""

from __future__ import annotations

from time import perf_counter
from typing import Dict, List, Tuple

from pathlib import Path
import csv

from src.graph.grid_builder import GridConfig, build_grid_graph
from src.algorithms.ucs import uniform_cost_search
from src.algorithms.greedy import greedy_search
from src.algorithms.astar import a_star_search

from src.heuristics.grid.manhattan import manhattan
from src.heuristics.grid.euclidean import euclidean
from src.heuristics.grid.chebyshev import chebyshev
from src.heuristics.grid.octile import octile


RESULTS_PATH = Path("data/results/experiment_results.csv")
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

HEURISTICS = {
    "manhattan": manhattan,
    "euclidean": euclidean,
    "chebyshev": chebyshev,
    "octile": octile,
}


def log_result(row: List) -> None:
    write_header = not RESULTS_PATH.exists()
    with open(RESULTS_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                [
                    "algorithm",
                    "heuristic",
                    "rows",
                    "cols",
                    "obstacle_ratio",
                    "seed",
                    "runtime",
                    "expanded",
                    "path_cost",
                ]
            )
        writer.writerow(row)


def run_one_experiment(
    rows: int,
    cols: int,
    obstacle_ratio: float,
    seed: int,
    algorithm: str,
    heuristic_name: str | None = None,
) -> None:
    cfg = GridConfig(rows, cols, obstacle_ratio, seed)
    graph, obstacles = build_grid_graph(cfg)
    start = (0, 0)
    goal = (rows - 1, cols - 1)

    if algorithm == "ucs":
        def fn():
            return uniform_cost_search(graph, start, goal)
    elif algorithm in ("greedy", "astar"):
        h = HEURISTICS[heuristic_name]
        if algorithm == "greedy":
            def fn():
                return greedy_search(graph, start, goal, h)
        else:
            def fn():
                return a_star_search(graph, start, goal, h)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    t0 = perf_counter()
    path, cost, expanded, _ = fn()
    runtime = perf_counter() - t0

    log_result(
        [
            algorithm,
            heuristic_name,
            rows,
            cols,
            obstacle_ratio,
            seed,
            runtime,
            expanded,
            cost,
        ]
    )


def run_full_suite() -> None:
    grid_sizes = [(20, 20), (30, 30), (40, 40)]
    obstacle_ratios = [0.1, 0.2, 0.3]
    seeds = [1, 7, 13, 21, 42]
    algorithms = ["ucs", "greedy", "astar"]

    for rows, cols in grid_sizes:
        for ratio in obstacle_ratios:
            for seed in seeds:
                # UCS (no heuristic)
                run_one_experiment(rows, cols, ratio, seed, "ucs")

                # Greedy and A* with each heuristic
                for h_name in HEURISTICS.keys():
                    run_one_experiment(rows, cols, ratio, seed, "greedy", h_name)
                    run_one_experiment(rows, cols, ratio, seed, "astar", h_name)

    print("[info] Experiment suite complete.")


if __name__ == "__main__":
    run_full_suite()