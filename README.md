# Transformer Causal Dynamics

Exploring whether transformers can learn the dynamics of Ornstein-Uhlenbeck processes. The goal is to see if attention mechanisms can capture mean-reversion (θ) and volatility (σ) from time series data.

**OU Process:** `dX_t = θ(μ - X_t)dt + σdW_t`

---

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Dependencies: `torch`, `numpy`, `pyyaml`, `matplotlib`, `seaborn`, `jupyter`, `tqdm`

---

## Quick Start

1. **Generate data:**
   ```bash
   python scripts/data_gen.py
   ```
   Generates two datasets:
   - Next token prediction: `data/ou_next_token.pt`
   - Mean prediction: `data/ou_mean_pred.pt` (trajectories with random means)

2. **Train model:**
   ```bash
   python scripts/train.py
   ```
   Select training mode:
   - `1` for next token prediction
   - `2` for mean prediction
   
   Models saved to `experiments/`

3. **Analyze:**
   - `notebooks/01_data_viz.ipynb` - Sample data visualization
   - `notebooks/02_analysis.ipynb` - Attention maps and extrapolation
   - `notebooks/03_regression_plot.ipynb` - X_t vs X_{t+1} regression
   - `notebooks/04_mean_prediction.ipynb` - Mean prediction scatter plot

---

## Structure

- `configs/` - YAML configs for data generation, model architecture, and training
- `scripts/` - Data generation and training scripts
- `src/` - Model definitions and utilities
- `notebooks/` - Analysis and visualization
- `data/` - Generated datasets
- `experiments/` - Model checkpoints

---

## Configs

- `data.yaml` - Physics params (θ, σ, μ, dt), data generation, and multi-mean settings
- `model.yaml` - Transformer architecture (d_model, n_heads, n_layers)
- `train.yaml` - Training hyperparameters (lr, epochs, batch size)

Just edit the YAML files to change parameters. The scripts will pick them up automatically.
