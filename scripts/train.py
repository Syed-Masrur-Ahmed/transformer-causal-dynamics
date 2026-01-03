import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import SimpleTransformer
from src.utils import load_full_config

def train():
    cfg = load_full_config()
    
    train_cfg = cfg['hyperparameters']
    path_cfg = cfg['paths'] 
    data_cfg = cfg['paths']
    
    data_path = data_cfg.get('save_path', 'data/ou_dataset.pt')
    print(f"Loading data from {data_path}...")
    checkpoint = torch.load(data_path)
    full_X = checkpoint['X'] # (N, 100, 1)
    
    # Input: Steps 0 to 98
    X_train = full_X[:, :-1, :] 
    # Target: Steps 1 to 99 (Next step prediction)
    Y_train = full_X[:, 1:, :]
    
    dataset = TensorDataset(X_train, Y_train)
    loader = DataLoader(dataset, batch_size=train_cfg['batch_size'], shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() and cfg['system']['device'] == "cuda" else "cpu")
    print(f"Using device: {device}")
    model = SimpleTransformer(**cfg['architecture'])
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=train_cfg['learning_rate'])
    criterion = nn.MSELoss()
    
    print("Starting training...")
    for epoch in range(train_cfg['epochs']):
        total_loss = 0
        for batch_X, batch_Y in loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            preds, _ = model(batch_X)
            
            loss = criterion(preds, batch_Y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}: Loss = {total_loss / len(loader):.6f}")
        
    os.makedirs(path_cfg['save_dir'], exist_ok=True)
    save_path = os.path.join(path_cfg['save_dir'], path_cfg['model_save_name'])
    
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train()