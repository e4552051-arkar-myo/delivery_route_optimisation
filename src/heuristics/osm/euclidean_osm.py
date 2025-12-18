"""
Euclidean heuristic for OpenStreetMap graphs.
Uses node coordinates (longitude, latitude).
"""

from math import sqrt


def euclidean_osm(G):
    """
    Returns a heuristic function h(u, v) for OSM graphs.
    """

    def heuristic(u, v):
        ux, uy = G.nodes[u]["x"], G.nodes[u]["y"]
        vx, vy = G.nodes[v]["x"], G.nodes[v]["y"]

        dx = ux - vx
        dy = uy - vy
        return sqrt(dx * dx + dy * dy)

    return heuristic