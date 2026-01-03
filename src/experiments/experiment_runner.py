"""
Experiment Runner (Grid)

Runs UCS, Greedy, and A* on multiple grid sizes / obstacle ratios / seeds.
Writes a single, consistent CSV file:

data/results/experiment_results.csv

Design goals:
- Clean CSV schema (no pandas ParserError)
- Always logs numeric path_cost (or NaN)
- Supports scaling experiments up to 100x100
- Regenerates random grids until start->goal is reachable (keeps experiments usable)
- Optionally skips UCS on very large grids (realistic + academically defensible)
"""

from __future__ import annotations

import csv
from pathlib import Path
from time import perf_counter
from typing import Callable, Dict, Optional, Tuple
from collections import deque

import math

from src.graph.grid_builder import GridConfig, build_grid_graph
from src.algorithms.ucs import uniform_cost_search
from src.algorithms.greedy import greedy_search
from src.algorithms.astar import a_star_search

# Grid heuristics
from src.heuristics.grid.manhattan import manhattan
from src.heuristics.grid.euclidean import euclidean
from src.heuristics.grid.chebyshev import chebyshev
from src.heuristics.grid.octile import octile


RESULTS_PATH = Path("data/results/experiment_results.csv")
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

HEURISTICS: Dict[str, Callable[[Tuple[int, int], Tuple[int, int]], float]] = {
    "manhattan": manhattan,
    "euclidean": euclidean,
    "chebyshev": chebyshev,
    "octile": octile,
}

# Single schema for the whole project (visualiser depends on this)
FIELDNAMES = [
    "algorithm",
    "heuristic",
    "rows",
    "cols",
    "obstacle_ratio",
    "seed",
    "runtime",
    "expanded",
    "path_cost",
    "found",
    "used_seed",
]


def obstacle_ratio_for(size: int) -> float:
    """
    Slightly lower obstacle density for larger grids to reduce disconnected instances.
    Keep it simple and explainable in the report.
    """
    if size <= 50:
        return 0.20
    if size <= 70:
        return 0.18
    return 0.15  # e.g., 100x100


def is_reachable(graph, start, goal) -> bool:
    """
    Fast BFS reachability check. Returns True if goal is reachable from start.
    """
    if start not in graph or goal not in graph:
        return False

    q = deque([start])
    seen = {start}

    while q:
        n = q.popleft()
        if n == goal:
            return True
        for nb, _ in graph.get(n, []):
            if nb not in seen:
                seen.add(nb)
                q.append(nb)

    return False


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def build_solvable_instance(
    rows: int,
    cols: int,
    obstacle_ratio: float,
    base_seed: int,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    max_tries: int = 40,
):
    """
    Regenerate grid with different seeds until start->goal is reachable.
    Returns (graph, obstacles, used_seed). Raises RuntimeError if not found.
    """
    for i in range(max_tries):
        cfg_try = GridConfig(
            rows=rows,
            cols=cols,
            obstacle_ratio=obstacle_ratio,
            seed=base_seed + i,
        )

        # If your build_grid_graph supports protected=, use it.
        # Otherwise fallback to plain call.
        try:
            graph, obstacles = build_grid_graph(cfg_try, protected={start, goal})
        except TypeError:
            graph, obstacles = build_grid_graph(cfg_try)

        if is_reachable(graph, start, goal):
            return graph, obstacles, cfg_try.seed

    raise RuntimeError(
        f"Failed to generate a solvable grid after {max_tries} tries "
        f"(rows={rows}, cols={cols}, obstacle_ratio={obstacle_ratio}, base_seed={base_seed})."
    )


def run_one_experiment(
    writer: csv.DictWriter,
    rows: int,
    cols: int,
    obstacle_ratio: float,
    seed: int,
    algorithm: str,
    heuristic_name: Optional[str] = None,
    ensure_solvable: bool = True,
) -> None:
    start = (0, 0)
    goal = (rows - 1, cols - 1)

    # Build a usable grid
    try:
        if ensure_solvable:
            graph, obstacles, used_seed = build_solvable_instance(
                rows=rows,
                cols=cols,
                obstacle_ratio=obstacle_ratio,
                base_seed=seed,
                start=start,
                goal=goal,
            )
        else:
            cfg = GridConfig(rows=rows, cols=cols, obstacle_ratio=obstacle_ratio, seed=seed)
            try:
                graph, obstacles = build_grid_graph(cfg, protected={start, goal})
            except TypeError:
                graph, obstacles = build_grid_graph(cfg)
            used_seed = seed
    except RuntimeError:
        # Log an unsolved case cleanly
        writer.writerow(
            {
                "algorithm": algorithm,
                "heuristic": heuristic_name or "",
                "rows": rows,
                "cols": cols,
                "obstacle_ratio": obstacle_ratio,
                "seed": seed,
                "runtime": float("nan"),
                "expanded": 0,
                "path_cost": float("nan"),
                "found": 0,
                "used_seed": "",
            }
        )
        return

    # Choose algorithm runner
    if algorithm == "ucs":
        heuristic_name = ""  # keep CSV clean
        def fn():
            return uniform_cost_search(graph, start, goal)

    elif algorithm in ("greedy", "astar"):
        if not heuristic_name:
            raise ValueError(f"{algorithm} requires heuristic_name")

        hfn = HEURISTICS[heuristic_name]

        if algorithm == "greedy":
            def fn():
                return greedy_search(graph, start, goal, hfn)
        else:
            def fn():
                return a_star_search(graph, start, goal, hfn)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # Run + time
    t0 = perf_counter()
    path, cost, expanded, _visited = fn()
    runtime = perf_counter() - t0

    found = 1 if path else 0
    path_cost = _safe_float(cost) if found else float("nan")

    writer.writerow(
        {
            "algorithm": algorithm,
            "heuristic": heuristic_name or "",
            "rows": rows,
            "cols": cols,
            "obstacle_ratio": obstacle_ratio,
            "seed": seed,
            "runtime": float(runtime),
            "expanded": int(expanded),
            "path_cost": float(path_cost),
            "found": int(found),
            "used_seed": used_seed,
        }
    )


def run_full_suite() -> None:
    # Scaling up to 100 with smooth increments (good for plots and discussion)
    grid_sizes = [(20, 20), (30, 30), (40, 40), (50, 50), (70, 70), (100, 100)]

    # For MSc: keep seeds modest so you can run quickly, but still claim robustness
    seeds = [1, 7, 13, 21, 42]

    ensure_solvable = True

    # Optional (recommended): UCS becomes expensive on huge grids
    ucs_max_size = 70

    # Overwrite each run to avoid mixing old CSV formats
    with open(RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for rows, cols in grid_sizes:
            ratio = obstacle_ratio_for(rows)

            for seed in seeds:
                # UCS
                if rows <= ucs_max_size:
                    run_one_experiment(
                        writer=writer,
                        rows=rows,
                        cols=cols,
                        obstacle_ratio=ratio,
                        seed=seed,
                        algorithm="ucs",
                        ensure_solvable=ensure_solvable,
                    )

                # Greedy + A* with each heuristic
                for h_name in HEURISTICS.keys():
                    run_one_experiment(
                        writer=writer,
                        rows=rows,
                        cols=cols,
                        obstacle_ratio=ratio,
                        seed=seed,
                        algorithm="greedy",
                        heuristic_name=h_name,
                        ensure_solvable=ensure_solvable,
                    )
                    run_one_experiment(
                        writer=writer,
                        rows=rows,
                        cols=cols,
                        obstacle_ratio=ratio,
                        seed=seed,
                        algorithm="astar",
                        heuristic_name=h_name,
                        ensure_solvable=ensure_solvable,
                    )

    print("[info] Experiment suite complete.")
    print(f"[info] Results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    run_full_suite()