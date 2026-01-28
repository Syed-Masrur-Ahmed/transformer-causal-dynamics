import torch
import numpy as np

def extract_trajectory_statistics(model, trajectory, theta, dt, device='cpu'):
    """
    Extract mean and variance from trajectory using empirical averaging of predictions.

    Instead of algebraic extraction (unstable for small θdt), we:
    1. Generate a long sequence of predictions
    2. Average them to estimate the stationary mean
    """
    model.eval()
    with torch.no_grad():
        trajectory = trajectory.to(device)
        preds, _ = model(trajectory)

        pred_mean = preds[:, :, 0]
        pred_second = preds[:, :, 1]

        # Average predictions across the trajectory
        # For a stationary process, these converge to E[X] and E[X²]
        mu_extracted = pred_mean.mean(dim=1)
        second_moment_extracted = pred_second.mean(dim=1)

        # Variance: Var(X) = E[X²] - (E[X])²
        var_extracted = second_moment_extracted - mu_extracted ** 2

    return mu_extracted.cpu(), var_extracted.cpu()
