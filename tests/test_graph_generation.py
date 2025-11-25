# tests/test_graph_generation.py
from src.graph.grid_builder import GridConfig, build_grid_graph


def test_build_grid_graph_basic():
    cfg = GridConfig(rows=3, cols=3, obstacle_ratio=0.0, seed=0)
    graph, obstacles = build_grid_graph(cfg)

    # 3x3 grid, no obstacles
    assert len(obstacles) == 0
    assert len(graph) == 9
    # Check neighbors of center cell (1,1): up, down, left, right
    neighbors = dict(graph[(1, 1)])
    assert (0, 1) in neighbors
    assert (2, 1) in neighbors
    assert (1, 0) in neighbors
    assert (1, 2) in neighbors