import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split

def generate_ou_process(batch_size, time_steps, theta, mu, sigma, dt):
    """
    Simulates Ornstein-Uhlenbeck processes using the exact solution.
    """
    X = torch.zeros(batch_size, time_steps, 1)
    X[:, 0, :] = mu

    # Pre-calculate constants for efficiency
    exp_theta_dt = np.exp(-theta * dt)
    sqrt_term = sigma * np.sqrt((1 - np.exp(-2 * theta * dt)) / (2 * theta))

    for t in range(1, time_steps):
        x_prev = X[:, t-1, :]
        dW_exact = torch.randn_like(x_prev)
        X[:, t, :] = x_prev * exp_theta_dt + mu * (1 - exp_theta_dt) + sqrt_term * dW_exact

    return X

def create_windowed_dataset(data, input_len, output_len, stride):
    """
    Slices raw trajectories into (Input, Target) pairs.
    """
    inputs = []
    targets = []
    
    num_trajectories = data.shape[0]
    total_len = data.shape[1]
    
    for i in range(num_trajectories):
        traj = data[i]
        for start_idx in range(0, total_len - input_len - output_len + 1, stride): # Added +1 to be safe
            end_input = start_idx + input_len
            end_target = end_input + output_len
            
            inp = traj[start_idx : end_input]
            tar = traj[end_input : end_target]
            
            inputs.append(inp)
            targets.append(tar)
            
    return torch.stack(inputs), torch.stack(targets)

def prepare_dataloaders(X, Y, batch_size, train_split=0.8):
    """Wraps tensors into PyTorch DataLoaders."""
    dataset = TensorDataset(X, Y)
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    
    return train_loader, val_loader