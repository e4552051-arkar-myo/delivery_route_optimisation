# src/experiments/experiment_runner.py

from time import perf_counter

from src.graph.grid_builder import GridConfig, build_grid_graph
from src.algorithms.ucs import uniform_cost_search
from src.algorithms.greedy import greedy_search
from src.algorithms.astar import a_star_search

from src.heuristics.manhattan import manhattan
from src.heuristics.euclidean import euclidean
from src.heuristics.chebyshev import chebyshev
from src.heuristics.octile import octile

from src.experiments.metrics import SearchMetrics
from src.experiments.logger import  log_experiment

HEURISTICS = {
    "manhattan": manhattan,
    "euclidean": euclidean,
    "chebyshev": chebyshev,
    "octile": octile,
}

def run_one_experiment(rows, cols, obstacle_ratio, algorithm, heuristic_name=None):
    cfg = GridConfig(rows, cols, obstacle_ratio, seed=42)
    graph, obstacles = build_grid_graph(cfg)
    start = (0, 0)
    goal = (rows - 1, cols - 1)

    # Select algorithm
    if algorithm == "ucs":
        fn = lambda: uniform_cost_search(graph, start, goal)
    elif algorithm == "greedy":
        h = HEURISTICS[heuristic_name]
        fn = lambda: greedy_search(graph, start, goal, h)
    else:
        h = HEURISTICS[heuristic_name]
        fn = lambda: a_star_search(graph, start, goal, h)

    # Time execution
    t0 = perf_counter()
    path, cost, expanded = fn()
    runtime = perf_counter() - t0

    metrics = SearchMetrics(
        algorithm=algorithm,
        heuristic=heuristic_name,
        grid_size=(rows, cols),
        obstacle_ratio=obstacle_ratio,
        runtime=runtime,
        expanded=expanded,
        path_cost=cost,
    )

    log_experiment(metrics)

    return metrics


def run_full_suite():
    grid_sizes = [(20, 20), (30, 30), (40, 40)]
    obstacle_ratios = [0.1, 0.2, 0.3]
    algorithms = ["ucs", "greedy", "astar"]

    for rows, cols in grid_sizes:
        for ratio in obstacle_ratios:

            # UCS first (no heuristic)
            run_one_experiment(rows, cols, ratio, "ucs")

            # Then Greedy + A*
            for h in HEURISTICS:
                run_one_experiment(rows, cols, ratio, "greedy", h)
                run_one_experiment(rows, cols, ratio, "astar", h)

    print("[info] Experiment suite complete.")

if __name__ == "__main__":
    print("Sprint 3 experiment runner started!")
    run_full_suite()