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
   Reads `configs/data.yaml` and saves to `data/ou_dataset.pt`

2. **Train model:**
   ```bash
   python scripts/train.py
   ```
   Uses all configs (merged via `src.utils.load_full_config()`) and saves to `experiments/`

3. **Analyze:**
   Open `notebooks/02_analysis.ipynb` for visualizations and attention maps

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

- `data.yaml` - Physics params (θ, σ, μ, dt) and data generation settings
- `model.yaml` - Transformer architecture (d_model, n_heads, n_layers)
- `train.yaml` - Training hyperparameters (lr, epochs, batch size)

Just edit the YAML files to change parameters. The scripts will pick them up automatically.
