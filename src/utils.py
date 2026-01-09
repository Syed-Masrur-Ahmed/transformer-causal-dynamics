import yaml
import os

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