# src/algorithms/greedy.py

from __future__ import annotations
import heapq
from typing import Hashable, Dict, List, Tuple, Callable
from src.utils.path_reconstruction import reconstruct_path

Node = Hashable
Graph = Dict[Node, List[Tuple[Node, float]]]
HeuristicFn = Callable[[Node, Node], float]


def greedy_search(
    graph: Graph,
    start: Node,
    goal: Node,
    heuristic: HeuristicFn
) -> Tuple[List[Node], float, int]:
    """
    Greedy Best-First Search.
    Expands the node that appears closest to the goal according to h(n).
    Not optimal, not complete in graphs with obstacles.
    """
    frontier = []
    heapq.heappush(frontier, (heuristic(start, goal), start))

    parent = {}
    visited = set()
    expanded = 0

    while frontier:
        _, node = heapq.heappop(frontier)
        expanded += 1

        if node in visited:
            continue
        visited.add(node)

        if node == goal:
            path = reconstruct_path(parent, start, goal)
            return path, float(len(path) - 1), expanded

        for neighbor, _ in graph.get(node, []):
            if neighbor not in visited:
                parent[neighbor] = node
                heapq.heappush(frontier, (heuristic(neighbor, goal), neighbor))

    return [], float("inf"), expanded