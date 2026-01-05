import yaml
import os

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
                config.update(yaml.safe_load(f))
        else:
            raise FileNotFoundError(f"Config file not found at: {file_full_path}")
            
    return config