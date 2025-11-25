# src/experiments/logger.py

import csv
from pathlib import Path
from src.experiments.metrics import SearchMetrics

RESULT_DIR = Path("data/results")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

def log_experiment(metrics: SearchMetrics):
    path = RESULT_DIR / "experiment_results.csv"
    write_header = not path.exists()

    with open(path, "a", newline="") as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow([
                "algorithm", "heuristic",
                "rows", "cols",
                "obstacle_ratio",
                "runtime",
                "expanded",
                "path_cost"
            ])

        writer.writerow([
            metrics.algorithm,
            metrics.heuristic,
            metrics.grid_size[0],
            metrics.grid_size[1],
            metrics.obstacle_ratio,
            metrics.runtime,
            metrics.expanded,
            metrics.path_cost
        ])