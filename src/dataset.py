import torch
import numpy as np

def generate_ou_process(batch_size, time_steps, theta, mu, sigma, dt):
    """
    Simulates Ornstein-Uhlenbeck processes using the exact solution.

    dX_t = γ(μ - X_t)dt + sqrt(2D) dW_t
    where γ = theta (mean reversion rate) and D is diffusion coefficient

    Stationary variance = D/γ
    With our parameterization: D = σ²/2, so stationary variance = σ²/(2θ)

    Exact solution:
    X_t = X_{t-1} * exp(-θdt) + μ(1 - exp(-θdt)) + sqrt(D/θ * (1 - exp(-2θdt))) * dW
    """
    X = torch.zeros(batch_size, time_steps, 1)
    X[:, 0, :] = mu

    # Pre-calculate constants for efficiency
    exp_theta_dt = np.exp(-theta * dt)
    # D = σ²/2, so sqrt_term = sqrt(D/θ * (1 - exp(-2θdt)))
    D = (sigma ** 2) / 2
    sqrt_term = np.sqrt(D / theta * (1 - np.exp(-2 * theta * dt)))

    for t in range(1, time_steps):
        x_prev = X[:, t-1, :]
        dW_exact = torch.randn_like(x_prev)
        X[:, t, :] = x_prev * exp_theta_dt + mu * (1 - exp_theta_dt) + sqrt_term * dW_exact

    return X

def create_windowed_dataset(data, input_len, output_len, stride):
    """
    Slices raw trajectories into (Input, Target) pairs for next-token prediction.

    Args:
        data: Tensor of shape (batch_size, time_steps, features)
        input_len: Length of input window
        output_len: Length of output window
        stride: Stride for sliding window

    Returns:
        inputs: Tensor of shape (num_samples, input_len, features)
        targets: Tensor of shape (num_samples, output_len, features)
    """
    inputs = []
    targets = []

    num_trajectories = data.shape[0]
    total_len = data.shape[1]

    for i in range(num_trajectories):
        traj = data[i]
        for start_idx in range(0, total_len - input_len - output_len + 1, stride):
            end_input = start_idx + input_len
            end_target = end_input + output_len

            inp = traj[start_idx : end_input]
            tar = traj[end_input : end_target]

            inputs.append(inp)
            targets.append(tar)

    return torch.stack(inputs), torch.stack(targets)
