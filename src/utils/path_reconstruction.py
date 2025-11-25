# src/utils/path_reconstruction.py
from __future__ import annotations

from typing import Dict, Hashable, List

Node = Hashable


def reconstruct_path(
    parent: Dict[Node, Node],
    start: Node,
    goal: Node,
) -> List[Node]:
    """
    Reconstruct a path from `start` to `goal` using a parent map.

    Args:
        parent: mapping {child: parent}.
        start: start node.
        goal: goal node.

    Returns:
        List of nodes from start to goal (inclusive), or [] if unreachable.
    """
    if goal != start and goal not in parent:
        return []

    cur = goal
    path = [cur]
    while cur != start:
        cur = parent[cur]
        path.append(cur)

    path.reverse()
    return path