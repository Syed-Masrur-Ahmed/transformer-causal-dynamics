import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

def generate_ou_process(batch_size, time_steps, theta=0.15, mu=0.0, sigma=0.2, dt=0.1):
    X = torch.zeros(batch_size, time_steps, 1)
    X[:, 0, :] = mu 
    
    for t in range(1, time_steps):
        x_prev = X[:, t-1, :]
        dW = torch.randn_like(x_prev) * np.sqrt(dt)
        drift = theta * (mu - x_prev) * dt
        X[:, t, :] = x_prev + drift + sigma * dW
        
    return X

def create_windowed_dataset(data, input_len=100, output_len=50, stride=1):
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

def prepare_dataloaders(X, Y, batch_size=64, train_split=0.8):
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
    # 1. Generate Raw Data
    print("Generating OU Process...")
    raw_data = generate_ou_process(batch_size=1000, time_steps=200)
    
    # 2. Chop into Windows (100 -> 50)
    print("Slicing into windows...")
    X, Y = create_windowed_dataset(raw_data, input_len=100, output_len=50)
    
    # 3. Save to disk (So you don't re-generate every time you train)
    print(f"Saving dataset: X shape {X.shape}, Y shape {Y.shape}")
    torch.save({'X': X, 'Y': Y}, 'data/ou_dataset.pt')