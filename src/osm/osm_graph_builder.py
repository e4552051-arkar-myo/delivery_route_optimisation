"""
Convert OSM road network into a weighted graph compatible
with existing search algorithms.
"""

def build_graph_from_osm(G):
    """
    Convert OSMnx graph into adjacency-list format.
    """
    graph = {}

    for u, v, data in G.edges(data=True):
        weight = data.get("length", 1.0)

        graph.setdefault(u, []).append((v, weight))
        graph.setdefault(v, []).append((u, weight))  # undirected

    return graph