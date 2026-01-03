import torch
import numpy as np
import os
import sys
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import load_full_config

def generate_ou_process(batch_size, time_steps, theta, mu, sigma, dt):
    """
    Simulates Ornstein-Uhlenbeck processes.
    dX_t = theta * (mu - X_t) * dt + sigma * dW_t
    """
    X = torch.zeros(batch_size, time_steps, 1)
    X[:, 0, :] = mu 
    
    for t in range(1, time_steps):
        x_prev = X[:, t-1, :]
        dW = torch.randn_like(x_prev) * np.sqrt(dt)
        drift = theta * (mu - x_prev) * dt
        X[:, t, :] = x_prev + drift + sigma * dW
        
    return X

def create_windowed_dataset(data, input_len, output_len, stride):
    """
    Slices raw trajectories into (Input, Target) pairs for the Transformer.
    """
    inputs = []
    targets = []
    
    # Iterate through each trajectory in the batch
    num_trajectories = data.shape[0]
    total_len = data.shape[1]
    
    for i in range(num_trajectories):
        traj = data[i] # (Time_Steps, 1)
        
        # Slide the window
        for start_idx in range(0, total_len - input_len - output_len, stride):
            end_input = start_idx + input_len
            end_target = end_input + output_len
            
            # Slice
            inp = traj[start_idx : end_input]
            tar = traj[end_input : end_target]
            
            inputs.append(inp)
            targets.append(tar)
            
    return torch.stack(inputs), torch.stack(targets)

def prepare_dataloaders(X, Y, batch_size, train_split):
    """
    Wraps tensors into PyTorch DataLoaders.
    """
    dataset = TensorDataset(X, Y)
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    
    return train_loader, val_loader

if __name__ == "__main__":
    print("Loading configuration...")
    cfg = load_full_config()
    
    # Unpack config sections for readability
    phys_cfg = cfg['physics']
    gen_cfg = cfg['generation']
    win_cfg = cfg['windowing']
    path_cfg = cfg['paths']

    print(f"Generating OU Process (Batch: {gen_cfg['batch_size']}, Steps: {gen_cfg['total_time_steps']})...")
    raw_data = generate_ou_process(
        batch_size=gen_cfg['batch_size'],
        time_steps=gen_cfg['total_time_steps'],
        theta=phys_cfg['theta'],
        mu=phys_cfg['mu'],
        sigma=phys_cfg['sigma'],
        dt=phys_cfg['dt']
    )
    
    print(f"Slicing into windows (Input: {win_cfg['input_len']} -> Output: {win_cfg['output_len']})...")
    X, Y = create_windowed_dataset(
        raw_data, 
        input_len=win_cfg['input_len'], 
        output_len=win_cfg['output_len'],
        stride=win_cfg['stride']
    )
    
    save_dir = os.path.dirname(path_cfg['save_path'])
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
    print(f"Saving dataset to {path_cfg['save_path']}: X shape {X.shape}, Y shape {Y.shape}")
    torch.save({'X': X, 'Y': Y}, path_cfg['save_path'])