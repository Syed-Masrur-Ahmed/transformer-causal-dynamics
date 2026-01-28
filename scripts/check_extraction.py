import torch
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import SimpleTransformer
from src.utils import load_full_config
from src.dataset import generate_ou_process

cfg = load_full_config()
device = torch.device('cpu')

# Load model
model = SimpleTransformer(**cfg['architecture'])
model.load_state_dict(torch.load('experiments/model_next_token.pth', map_location=device))
model.eval()

# Generate test trajectory with μ=0
test_traj = generate_ou_process(
    batch_size=1,
    time_steps=1000,
    theta=cfg['physics']['theta'],
    mu=0.0,  # True mean = 0
    sigma=cfg['physics']['sigma'],
    dt=cfg['physics']['dt']
)

print("=" * 70)
print("ALGEBRAIC EXTRACTION NUMERICAL STABILITY CHECK")
print("=" * 70)

theta = cfg['physics']['theta']
dt = cfg['physics']['dt']
alpha = np.exp(-theta * dt)
beta = 1 - alpha

print(f"θ = {theta}")
print(f"dt = {dt}")
print(f"α = exp(-θdt) = {alpha:.6f}")
print(f"β = 1 - α = {beta:.6f}  ← VERY SMALL! Dividing by this amplifies errors")
print(f"Amplification factor: 1/β = {1/beta:.1f}x")

with torch.no_grad():
    preds, _ = model(test_traj)

    pred_mean = preds[:, :-1, 0].numpy()
    X_t = test_traj[:, :-1, 0].numpy()
    X_next_true = test_traj[:, 1:, 0].numpy()

    # Check what model actually predicts
    print("\n" + "=" * 70)
    print("MODEL PREDICTIONS (first 5 timesteps)")
    print("=" * 70)
    print(f"{'X_t':>10} {'X_next (true)':>15} {'Pred':>15} {'Error':>10}")
    for i in range(5):
        print(f"{X_t[0,i]:10.4f} {X_next_true[0,i]:15.4f} {pred_mean[0,i]:15.4f} {pred_mean[0,i]-X_next_true[0,i]:10.4f}")

    # Next-step prediction quality
    mse = ((pred_mean - X_next_true) ** 2).mean()
    print(f"\nNext-step prediction MSE: {mse:.6f}")

    # Check dynamics
    changes = pred_mean - X_t
    slope, intercept = np.polyfit(X_t.flatten(), changes.flatten(), 1)
    expected_slope = -theta * dt

    print("\n" + "=" * 70)
    print("DYNAMICS CHECK")
    print("=" * 70)
    print(f"Expected slope: {expected_slope:.6f}")
    print(f"Learned slope:  {slope:.6f}")
    print(f"Error:          {abs(slope - expected_slope):.6f}")

    # Now try extraction
    print("\n" + "=" * 70)
    print("ALGEBRAIC EXTRACTION")
    print("=" * 70)

    numerator = pred_mean - X_t * alpha
    mu_estimates = numerator / beta

    print(f"\nNumerator range: [{numerator.min():.6f}, {numerator.max():.6f}]")
    print(f"After dividing by β={beta:.6f}:")
    print(f"μ estimates range: [{mu_estimates.min():.1f}, {mu_estimates.max():.1f}]  ← HUGE!")
    print(f"μ estimates mean: {mu_estimates.mean():.4f}")
    print(f"μ estimates std: {mu_estimates.std():.4f}")
    print(f"True μ: 0.0000")

    print("\n" + "=" * 70)
    print("WHAT'S HAPPENING")
    print("=" * 70)

    # Show specific example
    i = 100
    print(f"\nExample at timestep {i}:")
    print(f"  X_t = {X_t[0,i]:.6f}")
    print(f"  Predicted X_next = {pred_mean[0,i]:.6f}")
    print(f"  X_t * α = {X_t[0,i] * alpha:.6f}")
    print(f"  Numerator = pred - X_t*α = {numerator[0,i]:.6f}")
    print(f"  Divide by β = {beta:.6f}")
    print(f"  → μ estimate = {mu_estimates[0,i]:.2f}")

    # Show sensitivity
    print(f"\nSensitivity analysis:")
    print(f"  If prediction error is just 0.001:")
    error_amplified = 0.001 / beta
    print(f"  → μ error = {error_amplified:.2f}  (67x amplification!)")

    # Check if model is just predicting X_t (persistence)
    persistence_error = np.abs(pred_mean - X_t).mean()
    print(f"\nAverage prediction error from X_t: {persistence_error:.6f}")
    if persistence_error < 0.01:
        print("  → Model is basically predicting X_t (no change)")
        print("  → This makes extraction useless!")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

if abs(slope - expected_slope) / abs(expected_slope) > 0.2:
    print("✗ Model has NOT learned OU dynamics properly")
    print("  → Algebraic extraction cannot work")
    print("  → Need to train longer or fix model architecture")
else:
    print("✓ Model learned dynamics, but extraction is numerically unstable")
    print("  → Problem: β = 1 - exp(-θdt) = 0.0149 is too small")
    print("  → Small prediction errors get amplified 67x")
    print("  → Need a different approach")
