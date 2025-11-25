import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS = Path("data/results/experiment_results.csv")
PLOTS_DIR = Path("data/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def generate_combined_plots():
    print(PLOTS_DIR)
    df = pd.read_csv(RESULTS)
    print(df)
    plt.figure(figsize=(18, 12))

    # 1. Runtime subplot
    plt.subplot(3, 1, 1)
    for algo in df["algorithm"].unique():
        subset = df[df["algorithm"] == algo]
        plt.plot(subset["rows"], subset["runtime"], marker="o", label=algo)

    plt.title("Runtime Comparison Across Algorithms", fontsize=14)
    plt.xlabel("Grid Size (rows)")
    plt.ylabel("Runtime (seconds)")
    plt.grid(True)
    plt.legend()

    # 2. Node Expansion subplot
    plt.subplot(3, 1, 2)
    for algo in df["algorithm"].unique():
        subset = df[df["algorithm"] == algo]
        plt.plot(subset["rows"], subset["expanded"], marker="o", label=algo)

    plt.title("Node Expansion Comparison", fontsize=14)
    plt.xlabel("Grid Size (rows)")
    plt.ylabel("Nodes Expanded")
    plt.grid(True)
    plt.legend()

    # 3. Path Cost subplot
    plt.subplot(3, 1, 3)
    for algo in df["algorithm"].unique():
        subset = df[df["algorithm"] == algo]
        plt.plot(subset["rows"], subset["path_cost"], marker="o", label=algo)

    plt.title("Path Cost Comparison", fontsize=14)
    plt.xlabel("Grid Size (rows)")
    plt.ylabel("Path Cost")
    plt.grid(True)
    plt.legend()

    # Save final combined plot
    out_path = PLOTS_DIR / "combined_algorithm_performance.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    print(f"[info] Combined chart saved to: {out_path}")

if __name__ == "__main__":
    generate_combined_plots()