# src/algorithms/ucs.py
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
) -> Tuple[List[Node], float, int]:
    """
    Uniform Cost Search (graph search variant).

    Args:
        graph: adjacency list {node: [(neighbor, cost), ...]}.
        start: starting node.
        goal: goal node.

    Returns:
        path: list of nodes from start to goal (empty if no path).
        cost: total path cost (float("inf") if no path).
        expanded: number of nodes popped from the frontier.
    """
    frontier: List[Tuple[float, Node]] = []
    heapq.heappush(frontier, (0.0, start))

    # Best cost found so far to each node.
    best_cost: Dict[Node, float] = {start: 0.0}
    parent: Dict[Node, Node] = {}
    expanded = 0

    while frontier:
        cost, node = heapq.heappop(frontier)
        expanded += 1

        # If this is a stale entry, skip it.
        if cost > best_cost.get(node, float("inf")):
            continue

        if node == goal:
            path = reconstruct_path(parent, start, goal)
            return path, cost, expanded

        for neighbor, edge_cost in graph.get(node, []):
            new_cost = cost + edge_cost
            if new_cost < best_cost.get(neighbor, float("inf")):
                best_cost[neighbor] = new_cost
                parent[neighbor] = node
                heapq.heappush(frontier, (new_cost, neighbor))

    # No path
    return [], float("inf"), expanded