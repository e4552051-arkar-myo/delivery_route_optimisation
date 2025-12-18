"""
Example: Run routing algorithms on a real-world OSM map (Middlesbrough).

This file demonstrates real-world applicability.
It tests multiple heuristics (zero, euclidean, manhattan) for A* and Greedy.
"""

from src.osm.osm_loader import load_place
from src.osm.osm_graph_builder import build_graph_from_osm
from src.osm.osm_visualiser import plot_osm_route

from src.algorithms.astar import a_star_search
from src.algorithms.ucs import uniform_cost_search
from src.algorithms.greedy import greedy_search

from src.heuristics.osm.euclidean_osm import euclidean_osm
from src.heuristics.osm.manhattan_osm import manhattan_osm

import networkx as nx


def run_example():
    print("[info] Loading OSM road network for Middlesbrough...")
    G = load_place("Middlesbrough, UK")
    graph = build_graph_from_osm(G)

    # Fixed start & goal for reproducibility
    start = 703646761
    goal = 581121633

    print("Start:", start)
    print("Goal:", goal)

    # Heuristic factory functions
    heuristics = {
        "zero": (lambda _G: (lambda a, b: 0)),
        "euclidean": euclidean_osm,
        "manhattan": manhattan_osm,
    }

    # UCS (no heuristic)
    print("\n[UCS] Running Uniform Cost Search...")
    ucs_path, ucs_cost, _, _ = uniform_cost_search(graph, start, goal)
    print("UCS path length:", len(ucs_path))
    print("UCS cost:", ucs_cost)

    ucs_route = nx.shortest_path(G, start, goal, weight="length")
    plot_osm_route(G, ucs_route, filename="ucs_route.png")

    # A* + Greedy for each heuristic
    for h_name, h_factory in heuristics.items():
        h = h_factory(G)

        # ---- A* ----
        print(f"\n[A*] Running A* with {h_name} heuristic...")
        a_path, a_cost, _, _ = a_star_search(graph, start, goal, h)
        print("A* path length:", len(a_path))
        print("A* cost:", a_cost)

        # Visualise (use OSMnx route for accurate drawing)
        a_route = nx.shortest_path(G, start, goal, weight="length")
        plot_osm_route(G, a_route, filename=f"astar_{h_name}.png")

        # ---- Greedy ----
        print(f"\n[Greedy] Running Greedy with {h_name} heuristic...")
        g_path, g_cost, _, _ = greedy_search(graph, start, goal, h)
        print("Greedy path length:", len(g_path))
        print("Greedy cost:", g_cost)

        g_route = nx.shortest_path(G, start, goal, weight="length")
        plot_osm_route(G, g_route, filename=f"greedy_{h_name}.png")


if __name__ == "__main__":
    run_example()