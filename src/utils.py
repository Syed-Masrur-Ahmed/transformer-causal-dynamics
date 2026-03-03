import yaml
import os
from datetime import datetime
import uuid

def deep_update(base_dict, new_dict):
    """
    Recursively updates base_dict with new_dict.
    If a key exists in both and both are dictionaries, it merges them.
    Otherwise, it overwrites.
    """
    for key, value in new_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            deep_update(base_dict[key], value)
        else:
            base_dict[key] = value

def load_full_config(config_dir_name="configs"):
    """
    Merges data.yaml, model.yaml, and train.yaml into a single dictionary.
    Robustly finds the directory regardless of where the script is run.
    """
    config = {}
    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file_path))
    config_path = os.path.join(project_root, config_dir_name)
    
    files = ['data.yaml', 'model.yaml', 'train.yaml']
    
    for filename in files:
        file_full_path = os.path.join(config_path, filename)
        
        if os.path.exists(file_full_path):
            with open(file_full_path, 'r') as f:
                new_data = yaml.safe_load(f)
                if new_data:
                    deep_update(config, new_data) # <--- USE DEEP MERGE HERE
        else:
            print(f"Warning: Config file {filename} not found at {file_full_path}")
            
    return config


def generate_experiment_id(prefix="exp"):
    """Create a compact unique experiment identifier."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    return f"{prefix}_{timestamp}_{suffix}"


def _prefix_filename(filename, experiment_id):
    if not experiment_id:
        return filename
    prefixed = f"{experiment_id}_"
    if filename.startswith(prefixed):
        return filename
    return f"{prefixed}{filename}"


def apply_experiment_id_to_paths(cfg):
    """
    Prefix output artifacts with a shared experiment identifier.
    Reads `paths.experiment_id` first, then `EXPERIMENT_ID` env var.
    """
    path_cfg = cfg.setdefault("paths", {})
    experiment_id = path_cfg.get("experiment_id") or os.environ.get("EXPERIMENT_ID")

    if not experiment_id:
        return None

    path_cfg["experiment_id"] = experiment_id

    mgf_data_path = path_cfg.get("mgf_data_path", "data/ou_mgf.pt")
    data_dir, data_file = os.path.split(mgf_data_path)
    path_cfg["mgf_data_path"] = os.path.join(data_dir, _prefix_filename(data_file, experiment_id))

    mgf_model_name = path_cfg.get("mgf_model_name", "model_mgf.pth")
    path_cfg["mgf_model_name"] = _prefix_filename(mgf_model_name, experiment_id)

    loss_history_name = path_cfg.get("loss_history_name", "loss_history.pt")
    path_cfg["loss_history_name"] = _prefix_filename(loss_history_name, experiment_id)

    test_results_name = path_cfg.get("predictive_test_results_name", "predictive_test_results.csv")
    path_cfg["predictive_test_results_name"] = _prefix_filename(test_results_name, experiment_id)

    return experiment_id
