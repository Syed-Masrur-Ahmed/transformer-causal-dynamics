import torch
import numpy as np
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any
from torch.utils.data import TensorDataset, DataLoader, random_split


@dataclass
class StochasticSequence:
    data: torch.Tensor
    mean: float
    variance: float
    sequence_type: str
    theta: Optional[float] = None
    dt: Optional[float] = None
    timestamps: Optional[torch.Tensor] = None
    
    def save(self, path: str):
        save_dict = {
            'data': self.data,
            'mean': self.mean,
            'variance': self.variance,
            'sequence_type': self.sequence_type,
            'theta': self.theta,
            'dt': self.dt,
            'timestamps': self.timestamps
        }
        torch.save(save_dict, path)
    
    @classmethod
    def load(cls, path: str) -> 'StochasticSequence':
        d = torch.load(path)
        return cls(**d)
    
    @property
    def stationary_variance(self) -> float:
        if self.sequence_type == "ou_process" and self.theta:
            return self.variance / (2 * self.theta)
        return self.variance


@dataclass
class SequenceDataset:
    sequences: torch.Tensor
    means: torch.Tensor
    sequence_type: str
    physics_params: Dict[str, Any] = field(default_factory=dict)
    timestamps: Optional[torch.Tensor] = None
    thetas: Optional[torch.Tensor] = None
    
    def __len__(self) -> int:
        return self.sequences.shape[0]
    
    def __getitem__(self, idx: int) -> StochasticSequence:
        theta_val = float(self.thetas[idx]) if self.thetas is not None else self.physics_params.get('theta')
        return StochasticSequence(
            data=self.sequences[idx],
            mean=float(self.means[idx]),
            variance=self.physics_params.get('sigma', 0.0) ** 2,
            sequence_type=self.sequence_type,
            theta=theta_val,
            dt=self.physics_params.get('dt'),
            timestamps=self.timestamps
        )
    
    def save(self, path: str):
        save_dict = {
            'sequences': self.sequences,
            'means': self.means,
            'sequence_type': self.sequence_type,
            'physics_params': self.physics_params,
            'timestamps': self.timestamps,
            'thetas': self.thetas
        }
        torch.save(save_dict, path)
    
    @classmethod
    def load(cls, path: str) -> 'SequenceDataset':
        d = torch.load(path)
        return cls(**d)

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