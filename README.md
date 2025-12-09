# Delivery Route Optimisation  
Classical Search • Heuristic Search • Reinforcement Learning (Q‑Learning + Gymnasium)  
MSc Computer Science – ICA Project

This project implements and evaluates multiple search algorithms for delivery‑route optimisation on a grid world. It compares uninformed, informed and learning‑based approaches in terms of runtime, node expansion, path optimality and learning performance. The work combines classical algorithms with reinforcement learning, offering both traditional AI search and modern RL-based route learning.

The system includes:
- Uniform Cost Search (UCS) – baseline optimal uninformed search  
- Greedy Best‑First Search – fast, heuristic‑driven, non‑optimal search  
- A* Search – optimal, heuristic‑guided search  
- Q‑Learning (Classic Environment) – table‑based reinforcement learning  
- Q‑Learning (Gymnasium Environment) – optional Gym‑compatible reinforcement learning  
- Experiment Runner – automated benchmarking and CSV logging  
- Visualisers – runtime plots, heatmaps and reward curves  

---

## Project Structure

```
delivery_route_optimisation/
│
├── main.py                       # CLI entry point
├── README.md                     
├── requirements.txt              
│
├── src/
│   ├── algorithms/               # UCS, Greedy, A*
│   ├── heuristics/               # Manhattan, Euclidean, Chebyshev, Octile
│   ├── graph/                    # Grid generator and obstacle placement
│   ├── utils/                    # Timer and path reconstruction
│   ├── experiments/              # Experiment runner and visualisation tools
│   └── reinforcement/
│       ├── environment.py        # Classic Q-learning environment
│       ├── q_learning.py         # Q-learning agent
│       ├── trainer.py            # Classic Q-learning trainer
│       ├── gym_env.py            # Gymnasium-based grid environment
│       └── gym_trainer.py        # Gym-based Q-learning trainer
│
├── data/
│   ├── results/                  # experiment_results.csv (generated)
│   └── plots/                    # heatmaps, combined charts, RL reward curves
│
└── tests/                        # Unit tests for classical search and RL
```

---

## Running the Project

### 1. Create and activate a virtual environment

```
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Running Classical Search Algorithms

### Uniform Cost Search (UCS)
```
python main.py --algo ucs
```

### A* Search (Manhattan heuristic)
```
python main.py --algo astar --heuristic manhattan
```

### Greedy Best‑First Search
```
python main.py --algo greedy --heuristic euclidean
```

---

## Running Reinforcement Learning

### Q‑Learning — Classic Environment
```
python main.py --algo qlearning --rl-mode classic --episodes 500
```

### Q‑Learning — Gymnasium Environment
```
python main.py --algo qlearning --rl-mode gym --episodes 500
```

The Gymnasium mode uses the standard Gym API and supports integration with reinforcement learning libraries such as Stable‑Baselines3.

---

## Running Experiments

To generate a full benchmarking suite:

```
python -m src.experiments.experiment_runner
```

This produces a consolidated results file:

```
data/results/experiment_results.csv
```

---

## Combined Performance Plot

```
python -m src.experiments.visualiser
```

Saved to:

```
data/plots/combined_algorithm_performance.png
```

---

## Heatmap Visualisations

Each classical search algorithm (UCS, Greedy, A*) can generate a heatmap illustrating how often each grid cell was expanded during search.

Example:
```
python main.py --algo ucs
```

Heatmaps are saved under:

```
data/plots/heatmaps/
```

with filenames such as:

```
ucs_heatmap.png
greedy_heatmap.png
astar_heatmap.png
```

These heatmaps highlight the exploration characteristics of each algorithm, clearly showing the contrast between UCS’s exhaustive search, Greedy’s narrow heuristic-driven expansion and A*’s balanced efficiency.

---

## Q‑Learning Reward Curve

Both the classic and Gym‑based Q‑learning modes produce a reward curve file stored at:

```
data/plots/rl/reward_curve.png
```

This provides a visual overview of learning progression across training episodes.

---

# Algorithms Included

## Uniform Cost Search (UCS)
- Uninformed search  
- Guarantees optimality  
- High expansion cost on larger grids  

## Greedy Best‑First Search
- Uses only the heuristic h(n)  
- Very fast  
- Does not guarantee optimal paths  

## A* Search
- Uses f(n) = g(n) + h(n)  
- Optimal when using admissible heuristics  
- More efficient than UCS in most cases  

## Q‑Learning
- Reinforcement learning approach  
- Learns through trial and error  
- Performance improves with training  
- Optional Gymnasium mode provides a standard RL interface  

---

## Evaluation Metrics

Each experiment records:
- Runtime  
- Number of nodes expanded  
- Path cost  
- Heuristic used  
- Grid dimensions  
- Obstacle ratio  
- Random seed (multi‑seed experiments)  

All metrics are written to:

```
data/results/experiment_results.csv
```

---

## Summary

This project provides a detailed comparison between classical search algorithms and reinforcement learning methods applied to delivery‑route optimisation. The implementation highlights the trade‑offs between optimality, computational cost and search efficiency. The addition of Gymnasium support, heatmaps and comprehensive experimental tools enables a research‑grade evaluation suitable for MSc‑level coursework.
