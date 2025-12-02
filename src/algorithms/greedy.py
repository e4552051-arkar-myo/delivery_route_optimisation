"""
Greedy Best-First Search implementation.

This algorithm uses only the heuristic h(n) to decide which node to expand next.
It is very fast but does not guarantee optimal paths. It returns the path,
cost (path length), number of expanded nodes, and a visitation map for heatmaps.
"""

from __future__ import annotations

from typing import Callable, Dict, Hashable, List, Tuple
import heapq

from src.utils.path_reconstruction import reconstruct_path

Node = Hashable
Graph = Dict[Node, List[Tuple[Node, float]]]
Heuristic = Callable[[Node, Node], float]


def greedy_search(
    graph: Graph,
    start: Node,
    goal: Node,
    heuristic: Heuristic,
) -> Tuple[List[Node], float, int, Dict[Node, int]]:
    """
    Perform Greedy Best-First Search on the given graph.

    Args:
        graph: Adjacency list mapping node -> list of (neighbor, edge_cost).
        start: Start node.
        goal: Goal node.
        heuristic: Heuristic function h(n, goal).

    Returns:
        path: List of nodes from start to goal (empty if no path).
        cost: Total path cost (here: sum of edge costs along the path).
        expanded: Number of nodes popped from the frontier.
        visited_map: Map of node -> expansion count (for heatmaps).
    """
    frontier: List[Tuple[float, Node]] = []
    heapq.heappush(frontier, (heuristic(start, goal), start))

    parent: Dict[Node, Node] = {}
    visited: Dict[Node, bool] = {}
    expanded = 0
    visited_map: Dict[Node, int] = {}

    while frontier:
        _, node = heapq.heappop(frontier)

        if visited.get(node, False):
            continue

        visited[node] = True
        expanded += 1
        visited_map[node] = visited_map.get(node, 0) + 1

        if node == goal:
            path = reconstruct_path(parent, start, goal)
            # Compute cost as path length - 1 (moves) or use graph if weighted
            cost = float(len(path) - 1)
            return path, cost, expanded, visited_map

        for neighbor, _ in graph.get(node, []):
            if not visited.get(neighbor, False):
                parent[neighbor] = node
                priority = heuristic(neighbor, goal)
                heapq.heappush(frontier, (priority, neighbor))

    return [], float("inf"), expanded, visited_map