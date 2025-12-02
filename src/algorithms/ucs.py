"""
Uniform Cost Search (UCS) implementation.

This module implements UCS on a weighted graph represented as an adjacency list.
It returns the optimal path (if one exists), the total path cost, the number of
expanded nodes, and a visitation map suitable for heatmap visualisation.
"""

from __future__ import annotations

from typing import Dict, Hashable, List, Tuple
import heapq

from src.utils.path_reconstruction import reconstruct_path

Node = Hashable
Graph = Dict[Node, List[Tuple[Node, float]]]


def uniform_cost_search(
    graph: Graph,
    start: Node,
    goal: Node,
) -> Tuple[List[Node], float, int, Dict[Node, int]]:
    """
    Perform Uniform Cost Search (graph-search variant).

    Args:
        graph: Adjacency list mapping node -> list of (neighbor, edge_cost).
        start: Start node.
        goal: Goal node.

    Returns:
        path: List of nodes from start to goal (empty if no path).
        cost: Total path cost (float("inf") if no path).
        expanded: Number of nodes popped from the frontier.
        visited_map: Map of node -> expansion count (for heatmaps).
    """
    frontier: List[Tuple[float, Node]] = []
    heapq.heappush(frontier, (0.0, start))

    best_cost: Dict[Node, float] = {start: 0.0}
    parent: Dict[Node, Node] = {}
    expanded = 0
    visited_map: Dict[Node, int] = {}

    while frontier:
        cost, node = heapq.heappop(frontier)
        expanded += 1

        # Track how many times each node is expanded
        visited_map[node] = visited_map.get(node, 0) + 1

        # Skip stale entries
        if cost > best_cost.get(node, float("inf")):
            continue

        if node == goal:
            path = reconstruct_path(parent, start, goal)
            return path, cost, expanded, visited_map

        for neighbor, edge_cost in graph.get(node, []):
            new_cost = cost + edge_cost
            if new_cost < best_cost.get(neighbor, float("inf")):
                best_cost[neighbor] = new_cost
                parent[neighbor] = node
                heapq.heappush(frontier, (new_cost, neighbor))

    # No path found
    return [], float("inf"), expanded, visited_map