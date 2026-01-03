# Delivery Route Optimisation  
Classical Search • Heuristic Search • Reinforcement Learning (Q‑Learning + Gymnasium)  
MSc Computer Science – Artificial Intelligence (ICA Project)

## Overview

This project explores the **delivery route optimisation problem** using a combination of classical search algorithms, heuristic‑based search, and reinforcement learning. The main goal is to understand how different AI techniques behave when solving routing problems and to compare their performance in terms of efficiency, optimality, and learning behaviour.

The core experiments are carried out in a **grid‑based environment**, which allows full control over problem size, obstacle density, and start/goal positions. To demonstrate real‑world relevance, the project also includes an **OpenStreetMap (OSM) extension** that applies classical search algorithms to a real road network from Middlesbrough.

---

## Algorithms Implemented

The system includes the following algorithms:

- **Uniform Cost Search (UCS)** – baseline uninformed search that guarantees optimal solutions  
- **Greedy Best‑First Search** – fast heuristic‑driven search without optimality guarantees  
- **A\* Search** – optimal heuristic‑guided search using admissible heuristics  
- **Q‑Learning (Classic Environment)** – tabular reinforcement learning on a grid world  
- **Q‑Learning (Gymnasium Environment)** – optional Gym‑compatible RL implementation  
- **Experiment Runner** – automated benchmarking and CSV logging  
- **Visualisers** – heatmaps, grid maps, performance charts, and reward curves  

---

## Project Structure

```
delivery_route_optimisation/
│
├── main.py                       # Command-line entry point
├── README.md                     
├── requirements.txt              
│
├── src/
│   ├── algorithms/               # UCS, Greedy, A*
│   ├── heuristics/
│   │   ├── grid/                 # Manhattan, Euclidean, Chebyshev, Octile
│   │   └── osm/                  # OSM-specific heuristics
│   ├── graph/                    # Grid generator and obstacle placement
│   ├── utils/                    # Timing and path reconstruction utilities
│   ├── experiments/              # Experiment runner and visualisation tools
│   ├── reinforcement/
│   │   ├── environment.py        # Classic Q-learning environment
│   │   ├── q_learning.py         # Q-learning agent
│   │   ├── trainer.py            # Classic Q-learning trainer
│   │   ├── gym_env.py            # Gymnasium-based grid environment
│   │   └── gym_trainer.py        # Gym-based Q-learning trainer
│   └── osm/                      # OpenStreetMap loading, graph building and visualisation
│
├── data/
│   ├── results/                  # experiment_results.csv (generated)
│   └── plots/                    # heatmaps, grid maps, OSM routes, RL reward curves
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

## Running Classical Search Algorithms (Grid World)

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

Each run produces:
- Console output (path cost, nodes expanded)
- A **grid map** showing the computed route
- A **heatmap** illustrating node expansion behaviour

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

The Gymnasium mode follows the standard Gym API and allows future integration with external RL libraries.

---

## Running Experiments

To generate a full benchmarking suite across multiple grid sizes and algorithms:

```
python -m src.experiments.experiment_runner
```

This produces a consolidated results file:

```
data/results/experiment_results.csv
```

---

## Performance Visualisation

### Combined Performance Plot
```
python -m src.experiments.visualiser
```

Saved to:
```
data/plots/combined_algorithm_performance.png
```

This chart compares runtime, node expansion, and path cost across algorithms.

---

## Heatmap Visualisations

Each classical search algorithm (UCS, Greedy, A*) generates a heatmap showing how frequently each grid cell is expanded during search.

Heatmaps are saved under:
```
data/plots/heatmaps/
```

These visualisations clearly highlight the differences between exhaustive, heuristic‑driven, and balanced search strategies.

---

## Q‑Learning Reward Curve

Both the classic and Gym‑based Q‑learning modes generate a reward curve stored at:

```
data/plots/rl/reward_curve.png
```

This provides a clear visual representation of learning progress across episodes.

---

## OpenStreetMap Extension

The project includes an optional real‑world extension using OpenStreetMap data for Middlesbrough. Classical search algorithms (UCS, A*, Greedy) are applied to the road network, and the resulting routes are visualised directly on the map.

This extension is intended as a **demonstration of real‑world applicability**, rather than a full experimental comparison.

To run the OpenStreetMap demonstration:

```
python -m src.osm.osm_example
```
---

## Evaluation Metrics

Each experiment records:
- Runtime  
- Number of nodes expanded  
- Path cost  
- Heuristic used  
- Grid dimensions  
- Obstacle ratio  
- Random seed  

All metrics are written to:
```
data/results/experiment_results.csv
```

---

## Running Tests

Unit tests are provided to verify the correctness of grid generation, search algorithms, and reinforcement learning components.

To run all tests:

```
pytest
```
Tests are located in the `tests/` directory and cover:
- Grid graph generation
- Uniform Cost Search, Greedy, and A* correctness
- Q-learning behaviour on small environments

---

## Summary

This project provides a structured and practical comparison between classical search algorithms and reinforcement learning approaches for delivery route optimisation. The results demonstrate the efficiency of heuristic‑guided search, the trade‑off between speed and optimality, and the limitations of tabular reinforcement learning in large environments. The inclusion of comprehensive visualisation tools and a real‑world OSM extension makes the project suitable for MSc‑level assessment and further extension.
