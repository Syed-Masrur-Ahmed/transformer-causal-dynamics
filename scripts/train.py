import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import SimpleTransformer
from src.utils import load_full_config

def train_next_token(cfg):
    train_cfg = cfg['hyperparameters']
    path_cfg = cfg['paths']
    
    next_token_data_path = path_cfg.get('next_token_data_path', 'data/ou_next_token.pt')
    print(f"Loading data from {next_token_data_path}...")
    checkpoint = torch.load(next_token_data_path)
    full_X = checkpoint['X']
    
    X_train = full_X[:, :-1, :]
    Y_train = full_X[:, 1:, :]
    
    dataset = TensorDataset(X_train, Y_train)
    loader = DataLoader(dataset, batch_size=train_cfg['batch_size'], shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() and cfg['system']['device'] == "cuda" else "cpu")
    print(f"Using device: {device}")
    model = SimpleTransformer(**cfg['architecture'])
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=train_cfg['learning_rate'])
    criterion = nn.MSELoss()
    
    print("Starting training (next token prediction)...")
    for epoch in range(train_cfg['epochs']):
        total_loss = 0
        for batch_X, batch_Y in loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            optimizer.zero_grad()
            preds, _ = model(batch_X)
            loss = criterion(preds, batch_Y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss / len(loader):.6f}")
        
    os.makedirs(path_cfg['save_dir'], exist_ok=True)
    next_token_model_path = os.path.join(path_cfg['save_dir'], path_cfg['next_token_model_name'])
    torch.save(model.state_dict(), next_token_model_path)
    print(f"Model saved to {next_token_model_path}")

def train_mean_prediction(cfg):
    train_cfg = cfg['hyperparameters']
    path_cfg = cfg['paths']
    
    mean_pred_data_path = path_cfg.get('mean_pred_data_path', 'data/ou_mean_pred.pt')
    print(f"Loading data from {mean_pred_data_path}...")
    checkpoint = torch.load(mean_pred_data_path)
    trajectories = checkpoint['trajectories']
    means = checkpoint['means'].float().unsqueeze(1)
    
    dataset = TensorDataset(trajectories, means)
    loader = DataLoader(dataset, batch_size=train_cfg['batch_size'], shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() and cfg['system']['device'] == "cuda" else "cpu")
    print(f"Using device: {device}")
    model = SimpleTransformer(**cfg['architecture'])
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=train_cfg['learning_rate'])
    criterion = nn.MSELoss()
    
    print("Starting training (mean prediction)...")
    for epoch in range(train_cfg['epochs']):
        total_loss = 0
        for batch_X, batch_Y in loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            optimizer.zero_grad()
            preds, _ = model(batch_X)
            pred_mean = preds[:, -1, :]
            loss = criterion(pred_mean, batch_Y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss / len(loader):.6f}")
        
    os.makedirs(path_cfg['save_dir'], exist_ok=True)
    mean_pred_model_path = os.path.join(path_cfg['save_dir'], path_cfg.get('mean_pred_model_name', 'model_mean_pred.pth'))
    torch.save(model.state_dict(), mean_pred_model_path)
    print(f"Model saved to {mean_pred_model_path}")

if __name__ == "__main__":
    print("Select training mode:")
    print("  1. Next token prediction")
    print("  2. Mean prediction")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    cfg = load_full_config()
    
    if choice == '1':
        train_next_token(cfg)
    elif choice == '2':
        train_mean_prediction(cfg)
    else:
        print("Invalid choice. Please enter 1 or 2.")