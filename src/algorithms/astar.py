# src/algorithms/astar.py

from __future__ import annotations
import heapq
from typing import Dict, List, Tuple, Hashable, Callable
from src.utils.path_reconstruction import reconstruct_path

Node = Hashable
Graph = Dict[Node, List[Tuple[Node, float]]]
HeuristicFn = Callable[[Node, Node], float]


def a_star_search(
    graph: Graph,
    start: Node,
    goal: Node,
    heuristic: HeuristicFn
) -> Tuple[List[Node], float, int]:
    """
    A* Graph Search.
    f(n) = g(n) + h(n)
    Guaranteed optimal if heuristic is admissible & consistent.
    """
    frontier = []
    heapq.heappush(frontier, (0.0, start))

    g_cost = {start: 0.0}
    parent = {}
    expanded = 0

    while frontier:
        f, node = heapq.heappop(frontier)
        expanded += 1

        if node == goal:
            path = reconstruct_path(parent, start, goal)
            return path, g_cost[node], expanded

        for neighbor, edge_cost in graph.get(node, []):
            new_g = g_cost[node] + edge_cost

            if new_g < g_cost.get(neighbor, float("inf")):
                g_cost[neighbor] = new_g
                parent[neighbor] = node

                f_val = new_g + heuristic(neighbor, goal)
                heapq.heappush(frontier, (f_val, neighbor))

    return [], float("inf"), expanded