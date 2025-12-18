# src/heuristics/chebyshev.py

def chebyshev(a, b) -> float:
    ar, ac = a
    br, bc = b
    return max(abs(ar - br), abs(ac - bc))