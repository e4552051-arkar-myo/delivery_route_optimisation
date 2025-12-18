# src/heuristics/octile.py

import math

def octile(a, b) -> float:
    ar, ac = a
    br, bc = b
    dx = abs(ar - br)
    dy = abs(ac - bc)
    return dx + dy - min(dx, dy) + math.sqrt(2) * min(dx, dy)