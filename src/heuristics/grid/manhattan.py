# src/heuristics/manhattan.py

def manhattan(a, b) -> float:
    ar, ac = a
    br, bc = b
    return abs(ar - br) + abs(ac - bc)