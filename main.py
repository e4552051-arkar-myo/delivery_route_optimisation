# main.py
import argparse

from src.graph.grid_builder import GridConfig, build_grid_graph
from src.algorithms.ucs import uniform_cost_search
from src.algorithms.greedy import greedy_search
from src.algorithms.astar import a_star_search
from src.utils.timer import Timer

from src.heuristics.manhattan import manhattan
from src.heuristics.euclidean import euclidean
from src.heuristics.chebyshev import chebyshev
from src.heuristics.octile import octile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Delivery Route Optimisation")

    parser.add_argument("--rows", type=int, default=10)
    parser.add_argument("--cols", type=int, default=10)
    parser.add_argument("--obstacle-ratio", type=float, default=0.2)

    parser.add_argument("--start", type=int, nargs=2, default=(0, 0))
    parser.add_argument("--goal", type=int, nargs=2, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--algo",
        type=str,
        default="ucs",
        choices=["ucs", "greedy", "astar"],
        help="Which search algorithm to run."
    )

    parser.add_argument(
        "--heuristic",
        type=str,
        default="manhattan",
        choices=["manhattan", "euclidean", "chebyshev", "octile"],
        help="Heuristic function (used for Greedy and A*)."
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(args)

    rows, cols = args.rows, args.cols
    start = tuple(args.start)

    if args.goal is None:
        goal = (rows - 1, cols - 1)
    else:
        goal = tuple(args.goal)

    cfg = GridConfig(
        rows=rows,
        cols=cols,
        obstacle_ratio=args.obstacle_ratio,
        seed=args.seed,
    )

    print(f"[info] Building {rows}x{cols} grid (obstacle_ratio={args.obstacle_ratio})...")
    graph, obstacles = build_grid_graph(cfg)

    # print ("[check] graph", graph)
    # print ("[check] obstacles", obstacles)

    if start not in graph:
        raise ValueError(f"Start cell {start} is blocked.")
    if goal not in graph:
        raise ValueError(f"Goal cell {goal} is blocked.")

    print(f"[info] Number of free cells: {len(graph)}")
    print(f"[info] Number of obstacles: {len(obstacles)}")

    # Map heuristic name → function
    heuristics = {
        "manhattan": manhattan,
        "euclidean": euclidean,
        "chebyshev": chebyshev,
        "octile": octile,
    }

    algo_name = args.algo
    hfn = heuristics[args.heuristic]

    print(f"[info] Running algorithm: {algo_name}")

    # Run the selected algorithm
    with Timer(f"{algo_name.upper()} search"):
        if algo_name == "ucs":
            path, cost, expanded = uniform_cost_search(graph, start, goal)
        elif algo_name == "greedy":
            path, cost, expanded = greedy_search(graph, start, goal, hfn)
        else:  # astar
            path, cost, expanded = a_star_search(graph, start, goal, hfn)

    # Print results
    if not path:
        print("[result] No path found.")
    else:
        print(f"[result] Path found with cost={cost}")
        print(f"[result] Path length (nodes): {len(path)}")
        print(f"[result] Nodes expanded: {expanded}")
        print(f"[result] First 10 nodes: {path[:10]}")


if __name__ == "__main__":
    main()