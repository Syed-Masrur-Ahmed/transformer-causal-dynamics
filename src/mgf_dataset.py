import torch
import numpy as np

def generate_ou_process_variable_theta(batch_size, time_steps, theta_values, mu, D, dt):
    """
    Simulates OU processes with variable theta (relaxation rate).

    dX_t = θ(μ - X_t)dt + sqrt(2D) dW_t

    Args:
        batch_size: Number of trajectories
        time_steps: Length of each trajectory
        theta_values: Array of theta values, one per trajectory (batch_size,)
        mu: Long-term mean
        D: Diffusion coefficient
        dt: Time step

    Returns:
        X: Tensor of shape (batch_size, time_steps, 1)
        theta_values: Tensor of theta values (batch_size,)
    """
    X = torch.zeros(batch_size, time_steps, 1)
    X[:, 0, :] = mu

    if isinstance(theta_values, np.ndarray):
        theta_values = torch.from_numpy(theta_values).float()

    for t in range(1, time_steps):
        x_prev = X[:, t-1, :]
        exp_theta_dt = torch.exp(-theta_values.unsqueeze(1) * dt)
        sqrt_term = torch.sqrt(D / theta_values.unsqueeze(1) * (1 - torch.exp(-2 * theta_values.unsqueeze(1) * dt)))
        dW_exact = torch.randn_like(x_prev)
        X[:, t, :] = x_prev * exp_theta_dt + mu * (1 - exp_theta_dt) + sqrt_term * dW_exact

    return X, theta_values

def compute_log_mgf_targets(X_L, theta, D, dt, s_range):
    """
    Computes the log conditional moment-generating function φ(s | X_L, θ).

    φ(s|x,θ) = s·e^(-θ dt)·x + 0.5·s²·var_target·(1 - e^(-2θ dt))

    where var_target = D/θ (stationary variance)

    Args:
        X_L: Last state value, shape (batch_size, 1) or (batch_size,)
        theta: Relaxation rate per trajectory, shape (batch_size,)
        D: Diffusion coefficient (scalar)
        dt: Time step (scalar)
        s_range: Array of s values, shape (num_s,)

    Returns:
        phi: Tensor of shape (batch_size, num_s) with φ(s_k | X_L, θ) values
    """
    if X_L.dim() > 1:
        X_L = X_L.squeeze()

    batch_size = X_L.shape[0]
    num_s = len(s_range)

    s_grid = torch.tensor(s_range, dtype=torch.float32).unsqueeze(0)
    X_L_expanded = X_L.unsqueeze(1)
    theta_expanded = theta.unsqueeze(1)

    exp_theta_dt = torch.exp(-theta_expanded * dt)
    var_target = D / theta_expanded
    var_term = var_target * (1 - torch.exp(-2 * theta_expanded * dt))

    linear_term = s_grid * exp_theta_dt * X_L_expanded
    quadratic_term = 0.5 * (s_grid ** 2) * var_term
    phi = linear_term + quadratic_term

    return phi

def create_mgf_dataset(cfg):
    """
    Creates a dataset for MGF prediction.

    Returns:
        trajectories: Tensor of shape (num_trajectories, sequence_length, 1)
        targets: Tensor of shape (num_trajectories, num_s) - MGF values
        theta_values: Tensor of shape (num_trajectories,) - theta for each trajectory
        s_range: Array of s values
    """
    num_trajectories = cfg['generation']['num_trajectories']
    sequence_length = cfg['generation']['sequence_length']
    mu = cfg['physics']['mu']
    D = cfg['physics']['D']
    dt = cfg['physics']['dt']

    theta_mean = cfg['theta_sampling']['mean']
    theta_sigma = cfg['theta_sampling']['sigma']
    theta_values = np.random.lognormal(mean=theta_mean, sigma=theta_sigma, size=num_trajectories)

    print(f"Theta statistics: min={theta_values.min():.3f}, max={theta_values.max():.3f}, mean={theta_values.mean():.3f}")

    trajectories, theta_tensor = generate_ou_process_variable_theta(
        batch_size=num_trajectories,
        time_steps=sequence_length,
        theta_values=theta_values,
        mu=mu,
        D=D,
        dt=dt
    )

    s_min = cfg['mgf']['s_min']
    s_max = cfg['mgf']['s_max']
    num_points = cfg['mgf']['num_points']
    s_range = np.linspace(s_min, s_max, num_points)

    X_L = trajectories[:, -1, :]
    targets = compute_log_mgf_targets(X_L, theta_tensor, D, dt, s_range)

    print(f"Created dataset:")
    print(f"  Trajectories: {trajectories.shape}")
    print(f"  Targets: {targets.shape}")
    print(f"  Theta range: [{theta_tensor.min():.3f}, {theta_tensor.max():.3f}]")

    return trajectories, targets, theta_tensor, s_range
