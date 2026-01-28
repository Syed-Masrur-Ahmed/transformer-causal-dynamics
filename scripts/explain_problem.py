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

print("=" * 70)
print("THE FUNDAMENTAL PROBLEM")
print("=" * 70)

# Generate trajectory with known mean
mu_true = 0.5
test_traj = generate_ou_process(
    batch_size=1,
    time_steps=1000,
    theta=cfg['physics']['theta'],
    mu=mu_true,
    sigma=cfg['physics']['sigma'],
    dt=cfg['physics']['dt']
)

empirical_mean = test_traj.mean().item()
print(f"\nTrue μ (what process reverts to): {mu_true:.4f}")
print(f"Empirical mean of trajectory: {empirical_mean:.4f}")

# Load model
model = SimpleTransformer(**cfg['architecture'])
model.load_state_dict(torch.load('experiments/model_next_token.pth', map_location=device))
model.eval()

with torch.no_grad():
    preds, _ = model(test_traj)

    pred_mean = preds[:, :, 0].numpy()
    X_t = test_traj[:, :, 0].numpy()

    avg_prediction = pred_mean.mean()

    print(f"\n" + "=" * 70)
    print("WHAT EACH QUANTITY MEANS")
    print("=" * 70)

    print(f"\n1. True μ = {mu_true:.4f}")
    print(f"   → The long-term mean the OU process reverts to")
    print(f"   → Parameter in: dX = θ(μ - X)dt + σdW")

    print(f"\n2. Empirical mean = {empirical_mean:.4f}")
    print(f"   → Average of the actual trajectory values")
    print(f"   → This is NOT the same as μ!")
    print(f"   → It's a noisy estimate from one trajectory")

    print(f"\n3. Model predictions = {avg_prediction:.4f}")
    print(f"   → Average of E[X_{{t+1}} | X_t] across timesteps")
    print(f"   → Model predicts NEXT STEP, not trajectory mean")
    print(f"   → This is also NOT the same as μ!")

    # Show what model is actually predicting
    print(f"\n" + "=" * 70)
    print("WHAT THE MODEL PREDICTS")
    print("=" * 70)

    print(f"\n{'X_t':>10} {'Model Pred':>12} {'Change':>10}")
    for i in [0, 100, 200, 300, 400]:
        change = pred_mean[0, i] - X_t[0, i]
        print(f"{X_t[0, i]:10.4f} {pred_mean[0, i]:12.4f} {change:10.4f}")

    # Check if predictions are just X_t (no learning)
    avg_change = (pred_mean[0] - X_t[0, :-1]).mean()
    print(f"\nAverage change predicted: {avg_change:.6f}")

    if abs(avg_change) < 0.001:
        print("  → Model predicts almost NO CHANGE (X_{{t+1}} ≈ X_t)")
        print("  → Model hasn't learned OU dynamics!")
        print("  → It's just copying the input")

print(f"\n" + "=" * 70)
print("WHY AVERAGING PREDICTIONS DOESN'T GIVE μ")
print("=" * 70)

print(f"\nFor OU process: E[X_{{t+1}} | X_t] = X_t·e^(-θdt) + μ(1-e^(-θdt))")
print(f"\nIf we average this across the trajectory:")
print(f"  mean(E[X_{{t+1}} | X_t]) = mean(X_t)·e^(-θdt) + μ(1-e^(-θdt))")
print(f"                          = {empirical_mean:.4f}·{np.exp(-0.015):.4f} + {mu_true:.4f}·{1-np.exp(-0.015):.4f}")
print(f"                          = {empirical_mean * np.exp(-0.015) + mu_true * (1-np.exp(-0.015)):.4f}")
print(f"\nThis is NOT equal to μ = {mu_true:.4f}")
print(f"It's a weighted mix of the trajectory mean and μ")

print(f"\n" + "=" * 70)
print("THE BOTTOM LINE")
print("=" * 70)

print(f"\n1. Next-token prediction learns: E[X_{{t+1}} | X_t]")
print(f"2. We want: μ (the trajectory parameter)")
print(f"3. These are fundamentally different things")
print(f"\n4. Algebraic extraction CAN extract μ, but:")
print(f"   - Requires model to learn dynamics perfectly")
print(f"   - Numerically unstable (divides by 0.015)")
print(f"\n5. Simple averaging CANNOT extract μ because:")
print(f"   - It mixes trajectory mean with μ")
print(f"   - No mathematical relationship to μ alone")
print(f"\n6. Your mentor's constraint (next-token only) makes this VERY hard")

print(f"\n" + "=" * 70)
print("WHAT YOU CAN ACTUALLY DO")
print("=" * 70)

print(f"\nOption A: Use the trajectory itself")
print(f"  Predicted μ = mean(trajectory) = {empirical_mean:.4f}")
print(f"  True μ = {mu_true:.4f}")
print(f"  Error = {abs(empirical_mean - mu_true):.4f}")
print(f"  → Don't use model predictions at all!")
print(f"  → Just average the input trajectory")

print(f"\nOption B: Train until algebraic extraction works")
print(f"  → Need model to learn dynamics nearly perfectly")
print(f"  → Requires 50-100+ epochs")
print(f"  → Still numerically sensitive")

print(f"\nOption C: Change OU parameters for stability")
print(f"  → Use larger θ or dt so β = 1-e^(-θdt) is bigger")
print(f"  → Makes algebraic extraction less sensitive")

print(f"\nOption D: Tell your mentor this approach is fundamentally limited")
print(f"  → Next-token training learns local dynamics")
print(f"  → Extracting global parameters is hard/unstable")
print(f"  → Need auxiliary loss or different approach")
