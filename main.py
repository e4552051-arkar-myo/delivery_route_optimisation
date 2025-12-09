"""
Entry point for the Delivery Route Optimisation project.

Supports:
- Uniform Cost Search (UCS)
- Greedy Best-First Search
- A* Search
- Q-Learning (model-free RL baseline)

It parses CLI arguments, builds the grid world, runs the selected algorithm,
and optionally generates visualisations such as heatmaps and learning curves.
"""

from __future__ import annotations

import argparse
from typing import Tuple

from src.graph.grid_builder import GridConfig, build_grid_graph
from src.algorithms.ucs import uniform_cost_search
from src.algorithms.greedy import greedy_search
from src.algorithms.astar import a_star_search
from src.utils.timer import Timer

from src.heuristics.manhattan import manhattan
from src.heuristics.euclidean import euclidean
from src.heuristics.chebyshev import chebyshev
from src.heuristics.octile import octile

from src.reinforcement.trainer import train_q_learning, extract_path as q_extract_path
from src.experiments.heatmap_visualiser import save_heatmap
from src.experiments.rl_visualiser import plot_reward_curve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid-based delivery route optimisation using classical search and Q-learning."
    )

    parser.add_argument("--rows", type=int, default=20, help="Number of grid rows.")
    parser.add_argument("--cols", type=int, default=20, help="Number of grid columns.")
    parser.add_argument(
        "--obstacle-ratio",
        type=float,
        default=0.2,
        help="Fraction of blocked cells (0.0–0.6 recommended).",
    )
    parser.add_argument(
        "--start",
        type=int,
        nargs=2,
        metavar=("SR", "SC"),
        default=(0, 0),
        help="Start cell (row col), 0-based.",
    )
    parser.add_argument(
        "--goal",
        type=int,
        nargs=2,
        metavar=("GR", "GC"),
        default=None,
        help="Goal cell (row col), 0-based. Default is bottom-right.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for obstacle placement.",
    )

    parser.add_argument(
        "--algo",
        type=str,
        default="ucs",
        choices=["ucs", "greedy", "astar", "qlearning"],
        help="Which algorithm to run.",
    )
    parser.add_argument(
        "--heuristic",
        type=str,
        default="manhattan",
        choices=["manhattan", "euclidean", "chebyshev", "octile"],
        help="Heuristic for Greedy/A* (ignored for UCS and Q-learning).",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
        help="Number of training episodes for Q-learning.",
    )

    parser.add_argument(
        "--rl-mode",
        type=str,
        default="classic",
        choices=["classic", "gym"],
        help="RL mode: classic Q-learning or Gymnasium-based Q-learning."
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(args)

    rows, cols = args.rows, args.cols
    start: Tuple[int, int] = tuple(args.start)

    if args.goal is None:
        goal: Tuple[int, int] = (rows - 1, cols - 1)
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

    if start not in graph:
        raise ValueError(f"Start cell {start} is blocked or out of bounds.")
    if goal not in graph:
        raise ValueError(f"Goal cell {goal} is blocked or out of bounds.")

    print(f"[info] Number of free cells: {len(graph)}")
    print(f"[info] Number of obstacles: {len(obstacles)}")

    heuristics = {
        "manhattan": manhattan,
        "euclidean": euclidean,
        "chebyshev": chebyshev,
        "octile": octile,
    }

    algo_name = args.algo

    if algo_name == "qlearning":

        if args.rl_mode == "classic":
            print(f"[info] Training CLASSIC Q-learning agent for {args.episodes} episodes...")
            with Timer("Classic Q-learning"):
                agent, rewards = train_q_learning(
                    graph=graph,
                    start=start,
                    goal=goal,
                    obstacles=obstacles,
                    rows=rows,
                    cols=cols,
                    episodes=args.episodes,
                )

        else:  # Gym mode
            print(f"[info] Training GYM Q-learning agent for {args.episodes} episodes...")
            from src.reinforcement.gym_trainer import train_gym_qlearning
            with Timer("Gym Q-learning"):
                agent, rewards = train_gym_qlearning(
                    rows=rows,
                    cols=cols,
                    obstacles=obstacles,
                    start=start,
                    goal=goal,
                    episodes=args.episodes,
                )

        # Plot reward curve
        plot_reward_curve(rewards)
        print("[info] Reward curve saved.")

        # Extract path
        path = q_extract_path(
            agent=agent,
            start=start,
            goal=goal,
            obstacles=obstacles,
            rows=rows,
            cols=cols,
        )

        # Report Q-learning result
        if not path:
            print("[result] Q-learning did not find a valid path.")
        else:
            print(f"[result] Q-learning path length (nodes): {len(path)}")
            print(f"[result] Q-learning path cost (steps): {len(path) - 1}")
            print(f"[result] First 10 nodes: {path[:10]}")
        
        return

    # Classical search algorithms
    else:
        hfn = heuristics[args.heuristic]
        print(f"[info] Running algorithm: {algo_name}")

        with Timer(f"{algo_name.upper()} search"):
            if algo_name == "ucs":
                path, cost, expanded, visited_map = uniform_cost_search(graph, start, goal)
            elif algo_name == "greedy":
                path, cost, expanded, visited_map = greedy_search(graph, start, goal, hfn)
            else:  # astar
                path, cost, expanded, visited_map = a_star_search(graph, start, goal, hfn)

        if not path:
            print("[result] No path found.")
        else:
            print(f"[result] Path found with cost={cost}")
            print(f"[result] Path length (nodes): {len(path)}")
            print(f"[result] Nodes expanded: {expanded}")
            print(f"[result] First 10 nodes: {path[:10]}")

            heatmap_title = f"{algo_name.upper()} exploration heatmap"
            heatmap_file = f"{algo_name}_heatmap.png"
            save_heatmap(
                visited_map=visited_map,
                rows=rows,
                cols=cols,
                title=heatmap_title,
                filename=heatmap_file,
            )
            print(f"[info] Heatmap saved as data/plots/heatmaps/{heatmap_file}")


if __name__ == "__main__":
    main()