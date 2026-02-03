# Transformer Causal Dynamics

Training transformers to learn Ornstein-Uhlenbeck process dynamics by predicting the log moment-generating function (MGF).

**OU Process:** `dX_t = θ(μ - X_t)dt + σdW_t`

The transformer predicts `φ(s | X_L, θ) = s·E[X_{t+1}|X_t] + 0.5·s²·Var[X_{t+1}|X_t]` for multiple s values. Mean and variance are extracted via parabolic fitting.

---

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Quick Start

1. **Generate data:**
   ```bash
   python scripts/data_gen.py  # Select option 2 for MGF
   ```

2. **Train:**
   ```bash
   python scripts/train.py  # Select option 2 for MGF
   ```

3. **Analyze:**
   ```bash
   jupyter notebook notebooks/06_mgf_prediction.ipynb
   ```

### Other Options

- Option 1: Next token prediction
- Option 3: Mean prediction

### Notebooks

- `01_data_viz.ipynb` - Data visualization
- `02_analysis.ipynb` - Attention maps
- `03_regression_plot.ipynb` - Regression analysis
- `04_mean_prediction.ipynb` - Mean prediction
- `06_mgf_prediction.ipynb` - MGF prediction and moment extraction

---

## Structure

```
configs/          # YAML configuration files
scripts/          # Data generation and training
src/              # Model and dataset code
notebooks/        # Analysis notebooks
data/             # Generated datasets
experiments/      # Model checkpoints
```

## Configuration

Edit YAML files in `configs/`:
- `data.yaml` - Physics parameters, MGF settings
- `model.yaml` - Transformer architecture
- `train.yaml` - Training hyperparameters
