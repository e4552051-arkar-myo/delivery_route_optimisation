"""
OSM visualisation utilities.

Plots real-world routes on OpenStreetMap using OSMnx.
"""

import osmnx as ox
from pathlib import Path


def plot_osm_route(G, route, filename="osm_route.png"):
    """
    Plot a route on a real-world OSM map.

    - Grey: road network
    - Red : route
    """

    fig, ax = ox.plot_graph_route(
        G,
        route,
        route_color="red",
        route_linewidth=4,
        node_size=0,
        bgcolor="white",
        show=False,
        close=False,
    )

    out_dir = Path("data/plots/osm")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / filename
    fig.savefig(out_path, bbox_inches="tight")
    fig.clf()

    print(f"[info] OSM route map saved to {out_path}")