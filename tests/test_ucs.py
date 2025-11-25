# tests/test_ucs.py
from src.algorithms.ucs import uniform_cost_search


def test_ucs_on_simple_line():
    # Graph: 0 -1- 1 -1- 2
    graph = {
        0: [(1, 1.0)],
        1: [(0, 1.0), (2, 1.0)],
        2: [(1, 1.0)],
    }

    path, cost, expanded = uniform_cost_search(graph, 0, 2)
    assert path == [0, 1, 2]
    assert cost == 2.0
    assert expanded > 0