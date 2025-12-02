"""
A* Search implementation.

A* uses a combination of the path cost so far g(n) and a heuristic h(n)
to guide the search towards the goal while maintaining optimality when
the heuristic is admissible and consistent.
"""

from __future__ import annotations

from typing import Callable, Dict, Hashable, List, Tuple
import heapq

from src.utils.path_reconstruction import reconstruct_path

Node = Hashable
Graph = Dict[Node, List[Tuple[Node, float]]]
Heuristic = Callable[[Node, Node], float]


def a_star_search(
    graph: Graph,
    start: Node,
    goal: Node,
    heuristic: Heuristic,
) -> Tuple[List[Node], float, int, Dict[Node, int]]:
    """
    Perform A* search on the given graph.

    Args:
        graph: Adjacency list mapping node -> list of (neighbor, edge_cost).
        start: Start node.
        goal: Goal node.
        heuristic: Heuristic function h(n, goal).

    Returns:
        path: List of nodes from start to goal (empty if no path).
        cost: Total path cost.
        expanded: Number of nodes popped from the frontier.
        visited_map: Map of node -> expansion count (for heatmaps).
    """
    frontier: List[Tuple[float, Node]] = []
    heapq.heappush(frontier, (0.0, start))

    g_cost: Dict[Node, float] = {start: 0.0}
    parent: Dict[Node, Node] = {}
    expanded = 0
    visited_map: Dict[Node, int] = {}

    while frontier:
        f, node = heapq.heappop(frontier)
        expanded += 1
        visited_map[node] = visited_map.get(node, 0) + 1

        if node == goal:
            path = reconstruct_path(parent, start, goal)
            return path, g_cost[node], expanded, visited_map

        for neighbor, edge_cost in graph.get(node, []):
            tentative_g = g_cost[node] + edge_cost
            if tentative_g < g_cost.get(neighbor, float("inf")):
                g_cost[neighbor] = tentative_g
                parent[neighbor] = node
                f_new = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(frontier, (f_new, neighbor))

    return [], float("inf"), expanded, visited_map