# Quick Start Guide

This codebase supports two approaches for learning OU process dynamics with transformers:

## 1. Next-Token Prediction (Original)

Predicts the next timestep: E[X_{t+1}] and E[X²_{t+1}]

```bash
# Generate data
python scripts/data_gen.py
# Select: 1

# Train model
python scripts/train.py
# Select: 1

# Test in notebooks
jupyter notebook notebooks/02_analysis.ipynb
jupyter notebook notebooks/03_regression_plot.ipynb
```

## 2. MGF Prediction (New - Recommended)

Predicts the log conditional moment-generating function φ(s | X_L, θ)

```bash
# Generate data
python scripts/data_gen.py
# Select: 2

# Train model
python scripts/train.py
# Select: 2

# Test in notebook
jupyter notebook notebooks/06_mgf_prediction.ipynb
```

## Configuration

Edit config files in `configs/`:

- **data.yaml**: Data generation parameters
  - `generation`: Number of trajectories, sequence length
  - `theta_sampling`: For MGF - lognormal distribution parameters for relaxation rate θ
  - `mgf`: s-range and grid size

- **model.yaml**: Model architecture
  - `d_model`: 64 (embedding dimension)
  - `n_head`: 4 (attention heads)
  - `n_layers`: 2 (transformer layers)
  - `d_output`: 2 (next-token) or 50 (MGF)

- **train.yaml**: Training hyperparameters
  - `batch_size`: 64
  - `learning_rate`: 0.001
  - `epochs`: 50

## Directory Structure

```
transformer-causal-dynamics/
├── configs/           # YAML configuration files
├── src/              # Core library code
│   ├── model.py      # Transformer model
│   ├── dataset.py    # OU process generation
│   ├── mgf_dataset.py # MGF-specific data generation
│   └── inference.py  # Inference utilities
├── scripts/          # Executable scripts
│   ├── data_gen.py   # Unified data generation (menu)
│   └── train.py      # Unified training (menu)
├── notebooks/        # Jupyter analysis notebooks
├── data/            # Generated datasets
└── experiments/     # Trained model checkpoints
```

## Key Differences

| Aspect | Next-Token | MGF |
|--------|-----------|-----|
| **Training Task** | Predict X_{t+1}, X²_{t+1} | Predict φ(s) on grid |
| **Output Dim** | 2 | 50 |
| **Gamma** | Fixed | Variable (per trajectory) |
| **Use Case** | Time series prediction | Parameter inference |
| **Stability** | Numerically sensitive | Stable |

For most research purposes, **MGF prediction is recommended** as it:
- Directly learns the full conditional distribution
- Handles variable relaxation rates
- Is numerically stable
- Enables implicit parameter inference
