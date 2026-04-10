# CLAUDE.md

## Project Overview

Research project studying whether transformers can learn causal dynamics from Ornstein-Uhlenbeck (OU) stochastic processes. A transformer is trained to predict conditional moments (mean, variance) of the next step given a trajectory, effectively learning the moment-generating function (MGF) coefficients.

## Tech Stack

- Python 3.13, PyTorch, NumPy, Matplotlib, Seaborn, scikit-learn
- Quarto for documentation site (hosted on Render)
- Jupyter notebooks for analysis
- YAML configs in `configs/`

## Project Structure

```
run_pipeline.py          # Entry point: data gen -> train -> test
src/
  model.py               # SimpleTransformer with HermiteEmbedding, causal attention
  mgf_dataset.py         # OU process simulation + MGF target computation
  utils.py               # Config loading, experiment ID management
scripts/
  data_gen.py            # Standalone data generation
  train.py               # Training loop (MSE loss, Adam)
  test.py                # Theta/sequence-length sweep evaluation
configs/
  data.yaml              # Physics params (theta, mu, sigma, dt), trajectory structure
  model.yaml             # Architecture (d_model, n_head, n_layers, etc.)
  train.yaml             # Hyperparameters (lr, epochs, batch_size), paths
  test.yaml              # Theta sweep grid settings
analysis_notebooks/      # Jupyter notebooks for visualization/analysis
docs/                    # Quarto documentation site
experiments/             # Output dir for models, loss histories, test CSVs
data/                    # Generated .pt data files
```

## Key Commands

```bash
# Full pipeline (generates experiment ID automatically)
python run_pipeline.py
python run_pipeline.py --experiment-id my_exp

# Individual steps
python scripts/data_gen.py
python scripts/train.py
python scripts/train.py --variable-seq-len --min-seq-len 5
python scripts/test.py --coef 0   # 0=mean, 1=variance, 2=zero

# Activate venv
source venv/bin/activate
```

## Architecture Notes

- **HermiteEmbedding**: Scalar inputs are embedded via probabilist Hermite polynomials H_0..H_{d_model-1}, forming an orthogonal basis of L2(R, N(0,1)). Normalized by sqrt(n!).
- **SimpleTransformer**: HermiteEmbedding -> linear projection -> positional encoding -> causal attention blocks -> feedforward -> output projection. Returns `(predictions, attention_maps)`.
- **AttentionBlock**: Standard causal (upper-triangular) masked multi-head attention with residual + LayerNorm. No feedforward inside the block.
- Model outputs shape `(batch, seq_len, d_output)` where `d_output` matches the number of MGF coefficients (currently 3).

## Data & Physics

- OU process: `dX = -theta(X - mu)dt + sqrt(2D)dW`
- Theta can be fixed or sampled from a lognormal distribution (controlled by `theta_sampling` in data.yaml)
- Marginal variance is held constant across trajectories; D is derived as `D = marginal_variance * theta`
- Targets: conditional mean `mu + (X_t - mu)*exp(-theta*dt)` and conditional variance `(D/theta)*(1 - exp(-2*theta*dt))`

## Config System

Configs are merged via `load_full_config()`: data.yaml + model.yaml + train.yaml (deep merge). Test config additionally merges test.yaml on top. Experiment IDs prefix all output filenames for isolation.

## Conventions

- All scripts use `sys.path.append` to find project root — run from the project root directory
- Data saved as `.pt` (torch) files; test results as `.csv`
- Model checkpoints include config, losses, and state dict
