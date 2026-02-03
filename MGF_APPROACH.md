# MGF-Based Transformer Approach

## Overview

This approach trains a transformer to predict the **log conditional moment-generating function (MGF)** φ(s | X_L, θ) for Ornstein-Uhlenbeck processes with variable relaxation rates.

## Key Formula

The log conditional MGF is:

```
φ(s|X_L,θ) = s·e^(-θ dt)·X_L + 0.5·s²·(D/θ)·(1 - e^(-2θ dt))
```

where:
- **X_L**: Last observed state in the trajectory
- **θ**: Relaxation rate (sampled from lognormal distribution)
- **D**: Diffusion coefficient (D = σ²/2)
- **dt**: Time step
- **s**: Parameter of the MGF (evaluated on a grid)

## Architecture

- **Input**: OU trajectory of length 100: (X_0, X_1, ..., X_100)
- **Output**: φ(s) evaluated at 50 points in range [-2, 2]
- **Model**: 2-layer Transformer encoder (d_model=64, n_head=4)
- **Prediction**: Uses the hidden state at the **last token** X_L

## Configuration

### [configs/data.yaml](configs/data.yaml)
```yaml
theta_sampling:
  distribution: "lognormal"
  mean: 0.0        # log(θ) ~ N(0, 0.5)
  sigma: 0.5

generation:
  num_trajectories: 10000
  sequence_length: 100

mgf:
  s_min: -2.0
  s_max: 2.0
  num_points: 50   # Grid size for φ(s)
```

### [configs/model.yaml](configs/model.yaml)
```yaml
architecture:
  d_model: 64
  n_head: 4
  n_layers: 2
  d_output: 50     # Matches mgf.num_points
```

## Running Experiments

### 1. Generate Data
```bash
cd "/Users/jami/Documents/Research/URAD 26W/transformer-causal-dynamics"
python scripts/data_gen.py
# Select option 2 for MGF prediction
```

This creates `data/ou_mgf.pt` containing:
- 10,000 OU trajectories (length 100)
- MGF targets φ(s_k | X_L, θ) for each trajectory
- Gamma values sampled from lognormal(0, 0.5)

### 2. Train Model
```bash
python scripts/train.py
# Select option 2 for MGF prediction
```

Trains for 50 epochs, saves best model to `experiments/model_mgf.pth`

### 3. Test in Notebook

Open `notebooks/06_mgf_prediction.ipynb` and run all cells to:
- Load trained model
- Visualize predictions vs ground truth
- Analyze error distribution
- Test on trajectories with different θ values

## Key Files

- **[src/mgf_dataset.py](src/mgf_dataset.py)**: MGF dataset generation functions
- **[scripts/data_gen.py](scripts/data_gen.py)**: Unified data generation (option 2 for MGF)
- **[scripts/train.py](scripts/train.py)**: Unified training (option 2 for MGF)
- **[notebooks/06_mgf_prediction.ipynb](notebooks/06_mgf_prediction.ipynb)**: Analysis notebook

## What the Model Learns

The transformer learns to:
1. **Extract the final state X_L** from the trajectory
2. **Infer the relaxation rate θ** implicitly from the trajectory dynamics
3. **Predict φ(s)** using both X_L and the inferred θ

The model is NOT given θ explicitly - it must learn to extract it from the trajectory behavior!

## Why This Approach Works

Unlike next-token prediction:
- ✅ The MGF directly encodes both mean and variance information
- ✅ The relationship φ(s|X_L,θ) is deterministic (no noise amplification)
- ✅ The model learns a complete functional form, not just point estimates
- ✅ Numerically stable - no division by small numbers

## Gamma Statistics

With lognormal(0, 0.5):
- Mean θ ≈ 1.13
- Median θ ≈ 1.0
- Range typically [0.3, 3.0]

This gives a good distribution of relaxation rates to test generalization.
