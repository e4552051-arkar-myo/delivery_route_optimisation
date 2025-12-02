from src.algorithms.astar import a_star_search


def int_heuristic(a, b) -> float:
    """
    Simple heuristic for integer-labelled graphs.
    Uses absolute difference between node IDs.
    """
    return abs(a - b)


def test_astar_optimal_path_simple():
    # Graph: 0 -1- 1 -1- 2 (straight line)
    graph = {
        0: [(1, 1.0)],
        1: [(0, 1.0), (2, 1.0)],
        2: [(1, 1.0)],
    }

    path, cost, expanded, visited_map = a_star_search(graph, 0, 2, int_heuristic)

    assert path == [0, 1, 2]
    assert cost == 2.0
    assert expanded >= 1

    assert isinstance(visited_map, dict)
    assert len(visited_map) >= 1


def test_astar_optimal_in_weighted_graph():
    # Weighted graph where a naive greedy may fail:
    #
    # 0 -1- 1 -10- 3
    #  \           ^
    #   \-3- 2 ----|
    #
    graph = {
        0: [(1, 1.0), (2, 3.0)],
        1: [(0, 1.0), (3, 10.0)],
        2: [(0, 3.0), (3, 1.0)],
        3: [(1, 10.0), (2, 1.0)],
    }

    path, cost, expanded, visited_map = a_star_search(graph, 0, 3, int_heuristic)

    # Optimal path is 0 -> 2 -> 3, cost = 4.0
    assert path == [0, 2, 3]
    assert cost == 4.0

    assert isinstance(visited_map, dict)
    assert len(visited_map) >= 1