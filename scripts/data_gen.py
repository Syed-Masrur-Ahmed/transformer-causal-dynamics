import sys
import os
import torch
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import load_full_config
from src.dataset import generate_ou_process, create_windowed_dataset, SequenceDataset

if __name__ == "__main__":
    print("Loading configuration...")
    cfg = load_full_config()
    
    print("Generating OU Process...")
    raw_data = generate_ou_process(
        batch_size=cfg['generation']['batch_size'],
        time_steps=cfg['generation']['total_time_steps'],
        **cfg['physics']
    )
    
    print("Slicing into windows...")
    X, Y = create_windowed_dataset(
        raw_data, 
        input_len=cfg['windowing']['input_len'], 
        output_len=cfg['windowing']['output_len'],
        stride=cfg['windowing']['stride']
    )
    
    next_token_path = cfg['paths']['next_token_data_path']
    os.makedirs(os.path.dirname(next_token_path), exist_ok=True)
    print(f"Saving {len(X)} samples to {next_token_path}...")
    torch.save({'X': X, 'Y': Y}, next_token_path)
    
    print("Generating mean prediction OU trajectories...")
    multi_cfg = cfg['multi_mean']
    num_trajectories = multi_cfg['num_trajectories']
    mean_min = multi_cfg['mean_min']
    mean_max = multi_cfg['mean_max']
    random_means = np.random.uniform(mean_min, mean_max, num_trajectories)
    
    all_trajectories = []
    all_means = []
    
    for mu in random_means:
        traj = generate_ou_process(
            batch_size=1,
            time_steps=cfg['generation']['total_time_steps'],
            theta=cfg['physics']['theta'],
            mu=float(mu),
            sigma=cfg['physics']['sigma'],
            dt=cfg['physics']['dt']
        )
        all_trajectories.append(traj)
        all_means.append(mu)
    
    all_trajectories = torch.cat(all_trajectories, dim=0)
    all_means = torch.tensor(all_means, dtype=torch.float32)
    timestamps = torch.arange(cfg['generation']['total_time_steps']) * cfg['physics']['dt']
    
    dataset = SequenceDataset(
        sequences=all_trajectories,
        means=all_means,
        sequence_type="ou_process",
        physics_params={
            'theta': cfg['physics']['theta'],
            'sigma': cfg['physics']['sigma'],
            'dt': cfg['physics']['dt']
        },
        timestamps=timestamps
    )
    
    mean_pred_path = cfg['paths']['mean_pred_data_path']
    os.makedirs(os.path.dirname(mean_pred_path), exist_ok=True)
    print(f"Saving {num_trajectories} mean prediction trajectories to {mean_pred_path}...")
    dataset.save(mean_pred_path)
    
    print("Generating multi-theta OU trajectories...")
    theta_cfg = cfg['multi_theta']
    num_trajectories = theta_cfg['num_trajectories']
    gamma = theta_cfg['gamma']
    mean_min = theta_cfg['mean_min']
    mean_max = theta_cfg['mean_max']
    
    random_thetas = np.random.exponential(scale=1.0/gamma, size=num_trajectories)
    random_means = np.random.uniform(mean_min, mean_max, num_trajectories)
    
    all_trajectories = []
    all_means = []
    all_thetas = []
    
    for theta, mu in zip(random_thetas, random_means):
        traj = generate_ou_process(
            batch_size=1,
            time_steps=cfg['generation']['total_time_steps'],
            theta=float(theta),
            mu=float(mu),
            sigma=cfg['physics']['sigma'],
            dt=cfg['physics']['dt']
        )
        all_trajectories.append(traj)
        all_means.append(mu)
        all_thetas.append(theta)
    
    all_trajectories = torch.cat(all_trajectories, dim=0)
    all_means = torch.tensor(all_means, dtype=torch.float32)
    all_thetas = torch.tensor(all_thetas, dtype=torch.float32)
    timestamps = torch.arange(cfg['generation']['total_time_steps']) * cfg['physics']['dt']
    
    theta_dataset = SequenceDataset(
        sequences=all_trajectories,
        means=all_means,
        sequence_type="ou_process",
        physics_params={
            'sigma': cfg['physics']['sigma'],
            'dt': cfg['physics']['dt'],
            'gamma': gamma
        },
        timestamps=timestamps,
        thetas=all_thetas
    )
    
    multi_theta_path = cfg['paths']['multi_theta_data_path']
    os.makedirs(os.path.dirname(multi_theta_path), exist_ok=True)
    print(f"Saving {num_trajectories} multi-theta trajectories to {multi_theta_path}...")
    theta_dataset.save(multi_theta_path)