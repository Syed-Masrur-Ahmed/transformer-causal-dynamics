import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import SimpleTransformer
from src.utils import load_full_config
from src.dataset import SequenceDataset

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
    seq_dataset = SequenceDataset.load(mean_pred_data_path)
    trajectories = seq_dataset.sequences
    means = seq_dataset.means.float().unsqueeze(1)
    
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

def train_multi_theta_mean(cfg):
    train_cfg = cfg['hyperparameters']
    path_cfg = cfg['paths']
    
    multi_theta_data_path = path_cfg.get('multi_theta_data_path', 'data/ou_multi_theta.pt')
    print(f"Loading data from {multi_theta_data_path}...")
    seq_dataset = SequenceDataset.load(multi_theta_data_path)
    trajectories = seq_dataset.sequences
    means = seq_dataset.means.float().unsqueeze(1)
    
    dataset = TensorDataset(trajectories, means)
    loader = DataLoader(dataset, batch_size=train_cfg['batch_size'], shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() and cfg['system']['device'] == "cuda" else "cpu")
    print(f"Using device: {device}")
    model = SimpleTransformer(**cfg['architecture'])
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=train_cfg['learning_rate'])
    criterion = nn.MSELoss()
    
    print("Starting training (multi-theta mean prediction)...")
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
    multi_theta_model_path = os.path.join(path_cfg['save_dir'], path_cfg.get('multi_theta_model_name', 'model_multi_theta.pth'))
    torch.save(model.state_dict(), multi_theta_model_path)
    print(f"Model saved to {multi_theta_model_path}")

if __name__ == "__main__":
    print("Select training mode:")
    print("  1. Next token prediction")
    print("  2. Mean prediction (fixed theta)")
    print("  3. Mean prediction (multi-theta)")
    
    choice = input("Enter choice (1, 2, or 3): ").strip()
    
    cfg = load_full_config()
    
    if choice == '1':
        train_next_token(cfg)
    elif choice == '2':
        train_mean_prediction(cfg)
    elif choice == '3':
        train_multi_theta_mean(cfg)
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")