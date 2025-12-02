# src/experiments/combined_visualiser.py

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS = Path("data/results/experiment_results.csv")
PLOTS_DIR = Path("data/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def generate_combined_plots():
    df = pd.read_csv(RESULTS)

    plt.figure(figsize=(18, 12))

    # Runtime plot
    plt.subplot(3, 1, 1)
    for algo in df["algorithm"].unique():
        subset = df[df["algorithm"] == algo]
        plt.plot(subset["rows"], subset["runtime"], marker="o", label=algo)
    plt.title("Runtime Comparison Across Algorithms")
    plt.xlabel("Grid Size (rows)")
    plt.ylabel("Runtime")
    plt.grid(True)
    plt.legend()

    # Node expansion plot
    plt.subplot(3, 1, 2)
    for algo in df["algorithm"].unique():
        subset = df[df["algorithm"] == algo]
        plt.plot(subset["rows"], subset["expanded"], marker="o", label=algo)
    plt.title("Node Expansion Comparison")
    plt.xlabel("Grid Size (rows)")
    plt.ylabel("Nodes Expanded")
    plt.grid(True)
    plt.legend()

    # Path cost plot
    plt.subplot(3, 1, 3)
    for algo in df["algorithm"].unique():
        subset = df[df["algorithm"] == algo]
        plt.plot(subset["rows"], subset["path_cost"], marker="o", label=algo)
    plt.title("Path Cost Comparison")
    plt.xlabel("Grid Size (rows)")
    plt.ylabel("Path Cost")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    out_path = PLOTS_DIR / "combined_algorithm_performance.png"
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Combined plot saved to: {out_path}")


if __name__ == "__main__":
    generate_combined_plots()