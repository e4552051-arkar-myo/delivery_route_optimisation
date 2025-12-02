from src.algorithms.greedy import greedy_search


def int_heuristic(a, b) -> float:
    """
    Simple heuristic for integer-labelled graphs.
    Uses absolute difference between node IDs.
    """
    return abs(a - b)


def test_greedy_finds_a_path_but_not_optimal():
    # Diamond-shaped graph
    graph = {
        0: [(1, 1.0), (2, 1.0)],
        1: [(0, 1.0), (3, 10.0)],
        2: [(0, 1.0), (3, 1.0)],
        3: [(1, 10.0), (2, 1.0)],
    }

    path, cost, expanded, visited_map = greedy_search(graph, 0, 3, int_heuristic)

    assert len(path) > 0
    # Should at least reach the goal
    assert path[0] == 0
    assert path[-1] == 3

    # Greedy does not guarantee optimality; cost should be >= 2
    assert cost >= 2.0

    # Heatmap / visited_map sanity check
    assert isinstance(visited_map, dict)
    assert len(visited_map) >= 1