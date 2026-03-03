import sys
import os
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import apply_experiment_id_to_paths, load_full_config
from src.mgf_dataset import create_mgf_dataset


def generate_mgf_data(cfg):
    """Generate data for MGF prediction."""
    apply_experiment_id_to_paths(cfg)
    print("\nGenerating MGF dataset...")
    trajectories, theta_tensor, targets = create_mgf_dataset(cfg)

    data_path = cfg['paths']['mgf_data_path']
    os.makedirs(os.path.dirname(data_path), exist_ok=True)

    print(f"\nSaving to {data_path}...")
    torch.save({
        'trajectories': trajectories,
        'targets': targets,
        'theta_values': theta_tensor,
        'config': cfg
    }, data_path)
    print("Done!")

if __name__ == "__main__":
    print("Loading configuration...")
    cfg = load_full_config()
    apply_experiment_id_to_paths(cfg)

    generate_mgf_data(cfg)
