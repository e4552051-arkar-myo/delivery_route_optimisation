"""
Manhattan-distance heuristic for OpenStreetMap graphs.

This heuristic estimates distance using the sum of absolute
differences in latitude and longitude.
"""

def manhattan_osm(G):
    """
    Returns a Manhattan-distance heuristic function for OSM graphs.

    h(u, v) = |lat_u - lat_v| + |lon_u - lon_v|
    """

    def h(u, v):
        ux, uy = G.nodes[u]["x"], G.nodes[u]["y"]
        vx, vy = G.nodes[v]["x"], G.nodes[v]["y"]

        return abs(ux - vx) + abs(uy - vy)

    return h