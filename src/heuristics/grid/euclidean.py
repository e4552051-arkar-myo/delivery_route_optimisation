# src/heuristics/euclidean.py

import math

def euclidean(a, b) -> float:
    ar, ac = a
    br, bc = b
    return math.sqrt((ar - br)** 2 + (ac - bc)** 2)