# src/heuristics/custom.py

def custom_heuristic(a, b) -> float:
    """
    Placeholder for future custom heuristic.
    You might use traffic density, road quality, or region weights.
    """
    ar, ac = a
    br, bc = b
    return abs(ar - br) + abs(ac - bc)