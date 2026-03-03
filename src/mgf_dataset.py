import torch
import numpy as np

def simulate_ou_process(num_trajectories, sequence_length, theta_tensor, mu, D_tensor, dt):
    """
    Simulates OU processes with variable theta and D values.
    
    Args:
        num_trajectories: Number of trajectories to generate
        sequence_length: Length of each trajectory
        theta_tensor: Tensor of theta values, one per trajectory (num_trajectories,)
        mu: Long-term mean
        D_tensor: Tensor of diffusion coefficients, one per trajectory (num_trajectories,)
        dt: Time step
        
    Returns:
        trajectories: Tensor of shape (num_trajectories, sequence_length, 1)
    """
    trajectories = torch.zeros(num_trajectories, sequence_length, 1)
    trajectories[:, 0, :] = mu

    for t in range(1, sequence_length):
        x_prev = trajectories[:, t-1, :]
        exp_theta_dt = torch.exp(-theta_tensor.unsqueeze(1) * dt)
        sqrt_term = torch.sqrt(D_tensor.unsqueeze(1) / theta_tensor.unsqueeze(1) * (1 - torch.exp(-2 * theta_tensor.unsqueeze(1) * dt)))
        dW_exact = torch.randn_like(x_prev)
        trajectories[:, t, :] = x_prev * exp_theta_dt + mu * (1 - exp_theta_dt) + sqrt_term * dW_exact

    return trajectories

def create_mgf_dataset(cfg):
    """
    Creates a dataset for MGF prediction.

    Returns:
        trajectories: Tensor of shape (num_trajectories, sequence_length, 1)
        theta_tensor: Tensor of shape (num_trajectories,) - theta for each trajectory  
        targets: Tensor of shape (num_trajectories, sequence_length, order) - targets with [mean, variance, 0, 0, ...]
    """
    num_trajectories = cfg['structure']['num_trajectories']
    sequence_length = cfg['structure']['sequence_length']
    mu = cfg['physics']['mu']
    target_marginal_variance = cfg['physics']['marginal_variance']  # Target marginal variance
    dt = cfg['physics']['dt']
    order = cfg['target']['order']

    # Generate theta values
    theta_mean = cfg['theta_sampling']['mean']
    theta_sigma = cfg['theta_sampling']['sigma']
    theta_values = np.random.lognormal(mean=theta_mean, sigma=theta_sigma, size=num_trajectories)
    theta_tensor = torch.from_numpy(theta_values).float()
    
    # Compute D_tensor to maintain constant marginal variance: marginal_variance = D/θ
    # Therefore: D = marginal_variance * θ
    D_tensor = target_marginal_variance * theta_tensor

    # Generate OU process trajectories
    trajectories = simulate_ou_process(num_trajectories, sequence_length, theta_tensor, mu, D_tensor, dt)

    # Compute targets: shape (num_trajectories, sequence_length, order)
    targets = torch.zeros(num_trajectories, sequence_length, order)
    
    # Expand dimensions for broadcasting
    theta_expanded = theta_tensor.unsqueeze(1)  # (num_trajectories, 1)
    D_expanded = D_tensor.unsqueeze(1)  # (num_trajectories, 1)
    exp_theta_dt = torch.exp(-theta_expanded * dt)
    
    # First entry: conditional mean μ + (X_t - μ) * exp(-θ * dt) 
    targets[:, :, 0] = mu + (trajectories[:, :, 0] - mu) * exp_theta_dt
    
    # Second entry: conditional variance (D/θ) * (1 - exp(-2θ * dt))
    conditional_variance = (D_expanded / theta_expanded) * (1 - torch.exp(-2 * theta_expanded * dt))
    targets[:, :, 1] = conditional_variance.expand_as(targets[:, :, 1])
    
    return trajectories, theta_tensor, targets
