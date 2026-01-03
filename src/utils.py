import yaml

def load_full_config():
    """Merges all configs into one dictionary"""
    config = {}
    for filename in ['data.yaml', 'model.yaml', 'train.yaml']:
        with open(f"../configs/{filename}", 'r') as f:
            config.update(yaml.safe_load(f))
    return config