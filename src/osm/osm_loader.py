"""
OSM loader module.

Responsible only for loading OpenStreetMap data.
"""

import osmnx as ox


def load_place(place_name: str):
    """
    Load road network for a given place.
    """
    return ox.graph_from_place(place_name, network_type="drive")


def load_from_file(osm_file: str):
    """
    Load road network from a local .osm file.
    """
    return ox.graph_from_xml(osm_file)