import sys
import os
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import apply_experiment_id_to_paths, load_full_config
from src.mgf_dataset import create_mgf_dataset, create_binary_observed_dataset


def generate_mgf_data(cfg):
    """
    Generate data for MGF prediction.

    Args:
        cfg: Configuration dictionary
    """
    apply_experiment_id_to_paths(cfg)

    mode = cfg.get('mode', 'standard_ou')

    if mode == 'binary_ou_observed':
        print("\nGenerating binary OU dataset (observed mu)...")
        trajectories, theta_tensor, targets, states = create_binary_observed_dataset(cfg)
    else:
        print("\nGenerating normal OU dataset...")
        trajectories, theta_tensor, targets = create_mgf_dataset(cfg)
        states = None

    data_path = cfg['paths']['mgf_data_path']
    os.makedirs(os.path.dirname(data_path), exist_ok=True)

    print(f"\nSaving to {data_path}...")
    save_dict = {
        'trajectories': trajectories,
        'targets': targets,
        'theta_values': theta_tensor,
        'config': cfg
    }
    if states is not None:
        save_dict['states'] = states

    torch.save(save_dict, data_path)
    print("Done!")

if __name__ == "__main__":
    print("Loading configuration...")
    cfg = load_full_config()
    apply_experiment_id_to_paths(cfg)
    print(f"Mode: {cfg.get('mode', 'standard_ou')}")
    generate_mgf_data(cfg)
