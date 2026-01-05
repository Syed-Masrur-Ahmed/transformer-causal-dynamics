import sys
import os
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import load_full_config
from src.dataset import generate_ou_process, create_windowed_dataset

if __name__ == "__main__":
    print("Loading configuration...")
    cfg = load_full_config()
    
    # 1. Generate
    print("Generating OU Process...")
    raw_data = generate_ou_process(
        batch_size=cfg['generation']['batch_size'],
        time_steps=cfg['generation']['total_time_steps'],
        **cfg['physics'] # Unpacks theta, mu, sigma, dt
    )
    
    # 2. Slice
    print("Slicing into windows...")
    X, Y = create_windowed_dataset(
        raw_data, 
        input_len=cfg['windowing']['input_len'], 
        output_len=cfg['windowing']['output_len'],
        stride=cfg['windowing']['stride']
    )
    
    # 3. Save
    save_path = cfg['paths']['save_path']
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"Saving {len(X)} samples to {save_path}...")
    torch.save({'X': X, 'Y': Y}, save_path)