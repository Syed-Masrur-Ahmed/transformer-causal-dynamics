import sys
import os
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import apply_experiment_id_to_paths, load_full_config
from src.mgf_dataset import create_mgf_dataset, create_binary_mgf_dataset


def generate_mgf_data(cfg, use_binary=False):
    """
    Generate data for MGF prediction.

    Args:
        cfg: Configuration dictionary
        use_binary: If True, generate binary switching OU process. If False, generate normal OU process.
    """
    apply_experiment_id_to_paths(cfg)

    if use_binary:
        print("\nGenerating binary switching OU dataset...")
        trajectories, theta_tensor, targets, states = create_binary_mgf_dataset(cfg)
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

    # Ask user which type of data to generate
    print("\nChoose dataset type:")
    print("  1. Normal OU process")
    print("  2. Binary switching OU process")
    choice = input("Enter choice (1 or 2) [default: 1]: ").strip()

    use_binary = choice == "2"
    generate_mgf_data(cfg, use_binary=use_binary)
