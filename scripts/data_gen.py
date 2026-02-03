import sys
import os
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import load_full_config
from src.dataset import generate_ou_process, create_windowed_dataset
from src.mgf_dataset import create_mgf_dataset

def generate_next_token_data(cfg):
    """Generate data for next-token prediction."""
    print("\nGenerating OU process...")
    raw_data = generate_ou_process(
        batch_size=cfg['generation']['batch_size'],
        time_steps=cfg['generation']['total_time_steps'],
        **cfg['physics']
    )

    print("Creating windowed dataset...")
    X, Y = create_windowed_dataset(
        raw_data,
        input_len=cfg['windowing']['input_len'],
        output_len=cfg['windowing']['output_len'],
        stride=cfg['windowing']['stride']
    )

    next_token_path = cfg['paths']['next_token_data_path']
    os.makedirs(os.path.dirname(next_token_path), exist_ok=True)
    print(f"Saving {len(X):,} samples to {next_token_path}...")
    torch.save({'X': X, 'Y': Y}, next_token_path)
    print("Done!")

def generate_mgf_data(cfg):
    """Generate data for MGF prediction."""
    print("\nGenerating MGF dataset...")
    trajectories, targets, theta_values, s_range = create_mgf_dataset(cfg)

    data_path = cfg['paths']['mgf_data_path']
    os.makedirs(os.path.dirname(data_path), exist_ok=True)

    print(f"\nSaving to {data_path}...")
    torch.save({
        'trajectories': trajectories,
        'targets': targets,
        'theta_values': theta_values,
        's_range': s_range,
        'config': cfg
    }, data_path)
    print("Done!")

if __name__ == "__main__":
    print("Select data generation mode:")
    print("  1. Next-token prediction (windowed OU trajectories)")
    print("  2. MGF prediction (log conditional moment-generating function)")

    choice = input("Enter choice (1 or 2): ").strip()

    print("Loading configuration...")
    cfg = load_full_config()

    if choice == '1':
        generate_next_token_data(cfg)
    elif choice == '2':
        generate_mgf_data(cfg)
    else:
        print("Invalid choice. Please enter 1 or 2.")
