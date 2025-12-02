from src.algorithms.ucs import uniform_cost_search

def test_ucs_on_simple_line():
    # Graph: 0 -1- 1 -1- 2
    graph = {
        0: [(1, 1.0)],
        1: [(0, 1.0), (2, 1.0)],
        2: [(1, 1.0)],
    }

    path, cost, expanded, visited_map = uniform_cost_search(graph, 0, 2)

    assert path == [0, 1, 2]
    assert cost == 2.0
    assert expanded >= 1
    assert isinstance(visited_map, dict)
    assert len(visited_map) >= 1