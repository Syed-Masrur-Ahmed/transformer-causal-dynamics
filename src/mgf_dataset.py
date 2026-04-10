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

def simulate_binary_ou_process(num_trajectories, sequence_length, theta_tensor, mu, D_tensor, dt, kappa):
    """
    Simulates OU processes where the mean switches between +mu and -mu
    with switching rate kappa (continuous-time Markov chain).

    At each time step, the probability of switching state is 1 - exp(-kappa * dt).

    Args:
        num_trajectories: Number of trajectories to generate
        sequence_length: Length of each trajectory
        theta_tensor: Tensor of theta values, one per trajectory (num_trajectories,)
        mu: Absolute value of the mean (switches between +mu and -mu)
        D_tensor: Tensor of diffusion coefficients, one per trajectory (num_trajectories,)
        dt: Time step
        kappa: Switching rate

    Returns:
        trajectories: Tensor of shape (num_trajectories, sequence_length, 1)
        states: Tensor of shape (num_trajectories, sequence_length) with values +1 or -1
    """
    trajectories = torch.zeros(num_trajectories, sequence_length, 1)
    states = torch.ones(num_trajectories, sequence_length)  # +1 or -1

    # Initial state: randomly +1 or -1
    states[:, 0] = 2 * torch.bernoulli(torch.full((num_trajectories,), 0.5)) - 1
    trajectories[:, 0, 0] = states[:, 0] * mu

    p_switch = 1 - np.exp(-kappa * dt)

    for t in range(1, sequence_length):
        # Determine switching
        switches = torch.bernoulli(torch.full((num_trajectories,), p_switch))
        states[:, t] = states[:, t-1] * (1 - 2 * switches)  # flip sign on switch

        current_mu = states[:, t] * mu
        x_prev = trajectories[:, t-1, 0]

        exp_theta_dt = torch.exp(-theta_tensor * dt)
        sqrt_term = torch.sqrt(D_tensor / theta_tensor * (1 - torch.exp(-2 * theta_tensor * dt)))
        dW = torch.randn(num_trajectories)

        trajectories[:, t, 0] = x_prev * exp_theta_dt + current_mu * (1 - exp_theta_dt) + sqrt_term * dW

    return trajectories, states


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


def create_binary_mgf_dataset(cfg):
    """
    Creates a dataset for MGF prediction with binary switching OU process.
    The mean switches between +mu and -mu with rate kappa.

    Returns:
        trajectories: Tensor of shape (num_trajectories, sequence_length, 1)
        theta_tensor: Tensor of shape (num_trajectories,) - theta for each trajectory
        targets: Tensor of shape (num_trajectories, sequence_length, order) - targets with [mean, variance, state, ...]
        states: Tensor of shape (num_trajectories, sequence_length) - state trajectory (+1 or -1)
    """
    num_trajectories = cfg['structure']['num_trajectories']
    sequence_length = cfg['structure']['sequence_length']
    mu = cfg['physics']['mu']
    target_marginal_variance = cfg['physics']['marginal_variance']
    dt = cfg['physics']['dt']
    kappa = cfg['physics']['kappa']
    order = cfg['target']['order']

    # Generate theta values
    theta_mean = cfg['theta_sampling']['mean']
    theta_sigma = cfg['theta_sampling']['sigma']
    theta_values = np.random.lognormal(mean=theta_mean, sigma=theta_sigma, size=num_trajectories)
    theta_tensor = torch.from_numpy(theta_values).float()

    # Compute D_tensor to maintain constant marginal variance
    D_tensor = target_marginal_variance * theta_tensor

    # Generate binary OU process trajectories and states
    trajectories, states = simulate_binary_ou_process(num_trajectories, sequence_length, theta_tensor, mu, D_tensor, dt, kappa)

    # Compute targets: shape (num_trajectories, sequence_length, order)
    targets = torch.zeros(num_trajectories, sequence_length, order)

    # Expand dimensions for broadcasting
    theta_expanded = theta_tensor.unsqueeze(1)  # (num_trajectories, 1)
    D_expanded = D_tensor.unsqueeze(1)  # (num_trajectories, 1)
    exp_theta_dt = torch.exp(-theta_expanded * dt)

    # Current mean at each timestep: states[:, :, None] * mu
    current_mu = states.unsqueeze(2) * mu  # (num_trajectories, sequence_length, 1)

    # First entry: conditional mean current_mu + (X_t - current_mu) * exp(-θ * dt)
    targets[:, :, 0] = current_mu[:, :, 0] + (trajectories[:, :, 0] - current_mu[:, :, 0]) * exp_theta_dt

    # Second entry: conditional variance (D/θ) * (1 - exp(-2θ * dt))
    conditional_variance = (D_expanded / theta_expanded) * (1 - torch.exp(-2 * theta_expanded * dt))
    targets[:, :, 1] = conditional_variance.expand_as(targets[:, :, 1])

    # Third entry: current state (+1 or -1)
    if order > 2:
        targets[:, :, 2] = states

    return trajectories, theta_tensor, targets, states
