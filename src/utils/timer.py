# src/utils/timer.py
from __future__ import annotations

import time
from typing import Optional


class Timer:
    """
    Context manager for timing code blocks.

    Example:
        with Timer("UCS"):
            run_ucs()
    """

    def __init__(self, label: str = "Timer") -> None:
        self.label = label
        self.elapsed: Optional[float] = None

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        end = time.perf_counter()
        self.elapsed = end - self._start
        print(f"[time] {self.label}: {self.elapsed:.6f} seconds")