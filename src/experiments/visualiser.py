from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

RESULTS = Path("data/results/experiment_results.csv")
PLOTS_DIR = Path("data/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def generate_combined_plots():
    if not RESULTS.exists():
        raise FileNotFoundError(f"Missing results file: {RESULTS}")

    # Robust read
    df = pd.read_csv(RESULTS, engine="python", on_bad_lines="skip")

    # ---- Basic validation ----
    required = {"algorithm", "rows", "runtime", "expanded", "path_cost"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"experiment_results.csv missing columns: {sorted(missing)}")

    # If you have multiple settings mixed (e.g., different obstacle ratios),
    # it is better to filter to one setting for a clean plot.
    # Uncomment and set the value you want to report:
    #
    # if "obstacle_ratio" in df.columns:
    #     df = df[df["obstacle_ratio"] == 0.2]

    # ---- Aggregate to 3 lines: mean across seeds + heuristics ----
    agg = (
        df.groupby(["algorithm", "rows"], as_index=False)
        .agg(
            runtime_mean=("runtime", "mean"),
            expanded_mean=("expanded", "mean"),
            path_cost_mean=("path_cost", "mean"),
        )
        .sort_values(["algorithm", "rows"])
    )

    # ---- Plotting ----
    plt.figure(figsize=(14, 10))

    # 1) Runtime
    ax1 = plt.subplot(3, 1, 1)
    for algo in ["ucs", "astar", "greedy"]:
        if algo in set(agg["algorithm"]):
            s = agg[agg["algorithm"] == algo].sort_values("rows")
            ax1.plot(s["rows"], s["runtime_mean"], marker="o", label=algo)
    ax1.set_title("Runtime (mean across seeds and heuristics)")
    ax1.set_xlabel("Grid size (rows)")
    ax1.set_ylabel("Seconds")
    ax1.set_xticks(sorted(agg["rows"].unique()))
    ax1.grid(True)
    ax1.legend()

    # 2) Nodes Expanded
    ax2 = plt.subplot(3, 1, 2)
    for algo in ["ucs", "astar", "greedy"]:
        if algo in set(agg["algorithm"]):
            s = agg[agg["algorithm"] == algo].sort_values("rows")
            ax2.plot(s["rows"], s["expanded_mean"], marker="o", label=algo)
    ax2.set_title("Nodes expanded (mean across seeds and heuristics)")
    ax2.set_xlabel("Grid size (rows)")
    ax2.set_ylabel("Nodes expanded")
    ax2.set_xticks(sorted(agg["rows"].unique()))
    ax2.grid(True)
    ax2.legend()

    # 3) Path Cost
    ax3 = plt.subplot(3, 1, 3)
    for algo in ["ucs", "astar", "greedy"]:
        if algo in set(agg["algorithm"]):
            s = agg[agg["algorithm"] == algo].sort_values("rows")
            ax3.plot(s["rows"], s["path_cost_mean"], marker="o", label=algo)
    ax3.set_title("Path cost (mean across seeds and heuristics)")
    ax3.set_xlabel("Grid size (rows)")
    ax3.set_ylabel("Cost")
    ax3.set_xticks(sorted(agg["rows"].unique()))
    ax3.grid(True)
    ax3.legend()

    out_path = PLOTS_DIR / "combined_algorithm_performance.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"[info] Saved: {out_path}")


if __name__ == "__main__":
    generate_combined_plots()