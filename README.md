# Delivery Route Optimisation  
**Classical Search • Heuristic Search • Reinforcement Learning (Q-Learning)**  
MSc Computer Science – ICA Project

This project implements and evaluates multiple search algorithms for delivery‑route optimisation on a grid world. It compares uninformed, informed, and learning‑based approaches in terms of runtime, node expansion, and path optimality.

The system implements:
- **Uniform Cost Search (UCS)** – baseline optimal uninformed search  
- **Greedy Best‑First Search** – fast, heuristic‑driven, non‑optimal  
- **A\* Search** – optimal, admissible heuristic‑guided  
- **Q‑Learning (Reinforcement Learning)** – model‑free, trial‑and‑error learning  
- **Experiment Runner** – automated benchmarking suite  
- **Visualiser** – runtime, expansion, and path‑cost charts  

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
│   ├── graph/                    # Grid generator + obstacles
│   ├── utils/                    # Timer + path reconstruction
│   ├── experiments/              # Experiment runner, CSV logger, visualiser
│   └── reinforcement/            # Q-learning env + agent + trainer
│
├── data/
│   ├── results/                  # experiment_results.csv (auto‑generated)
│   └── plots/                    # combined_algorithm_performance.png (auto‑generated)
│
└── tests/                        # Unit tests for grid + algorithms
```

---

##  Running the Project

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

##  Run a Single Algorithm

### **UCS**
```bash
python main.py --algo ucs
```

### **A\*** (Manhattan heuristic)
```bash
python main.py --algo astar --heuristic manhattan
```

### **Greedy Best‑First Search**
```bash
python main.py --algo greedy --heuristic euclidean
```

### **Q‑Learning** (500 episodes)
```bash
python main.py --algo qlearning --episodes 500
```

---

## Running Experiments

Run full benchmarking suite:
```bash
python -m src.experiments.experiment_runner
```

Generates:
```
data/results/experiment_results.csv
```

---

## Generate Combined Performance Plot

```bash
python -m src.experiments.visualiser
```

Saved to:
```
data/plots/combined_algorithm_performance.png
```

---

## Generate Heatmap Visualisations

After running any search algorithm (UCS, Greedy, A*), a heatmap can be produced showing how many times each grid cell was expanded during the search.

Generate heatmaps using:

```bash
python main.py --algo ucs       # or greedy / astar
```

Heatmaps are saved automatically to:

```
data/plots/heatmaps/
```

Files are named in the format:

```
ucs_heatmap.png
greedy_heatmap.png
astar_heatmap.png
```

These heatmaps help visualise how different algorithms explore the grid, showing clear contrasts between UCS's exhaustive expansion, Greedy's narrow beam‑like search, and A*’s balanced informed exploration.
---

# Algorithms Included

## **Uniform Cost Search (UCS)**
- Uninformed search  
- Always optimal  
- Very high search cost  

## **Greedy Best‑First Search**
- Uses only heuristic h(n)  
- Extremely fast  
- Not optimal (detours common)  

## **A\* Search**
- Uses f(n) = g(n) + h(n)  
- Always optimal with admissible heuristics  
- More efficient than UCS  

## **Q‑Learning**
- Learns routes through reward signals  
- Model‑free RL  
- Not guaranteed optimal  
- Improves with training  

---

## Evaluation Metrics

Every experiment logs:
- Runtime (seconds)  
- Nodes expanded (search effort)  
- Path cost (optimality)  
- Heuristic used  
- Grid size & obstacle ratio  

All metrics are saved to:
```
data/results/experiment_results.csv
```

---

## Summary

This project provides a full experimental comparison between classical search algorithms and reinforcement learning for grid‑based delivery routing. It demonstrates trade‑offs between optimality, speed, and computational cost, and includes a complete experiment suite for reproducible evaluation.
