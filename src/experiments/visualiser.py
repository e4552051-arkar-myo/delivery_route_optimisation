# src/experiments/visualiser.py

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_results(csv_file: str, y_column: str, title: str, output: str):
    path = Path("data/results") / csv_file
    df = pd.read_csv(path)

    plt.figure(figsize=(12, 6))
    for algo in df["algorithm"].unique():
        subset = df[df["algorithm"] == algo]
        plt.plot(subset["rows"], subset[y_column], marker="o", label=algo)

    plt.title(title)
    plt.xlabel("Grid size (rows)")
    plt.ylabel(y_column)
    plt.legend()
    plt.grid(True)

    out_path = Path("data/plots") / output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()