import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import SimpleTransformer
from src.utils import load_full_config

def train_next_token(cfg):
    """Train for next-token prediction."""
    train_cfg = cfg['hyperparameters']
    path_cfg = cfg['paths']

    next_token_data_path = path_cfg.get('next_token_data_path', 'data/ou_next_token.pt')
    print(f"Loading data from {next_token_data_path}...")
    checkpoint = torch.load(next_token_data_path, weights_only=False)
    full_X = checkpoint['X']

    X_train = full_X[:, :-1, :]
    Y = full_X[:, 1:, :]

    dataset = TensorDataset(X_train, Y)
    loader = DataLoader(dataset, batch_size=train_cfg['batch_size'], shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() and cfg['system']['device'] == "cuda" else "cpu")
    print(f"Using device: {device}")
    model = SimpleTransformer(**cfg['architecture'])
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=train_cfg['learning_rate'])
    criterion = nn.MSELoss()

    print("\nStarting training (next-token prediction)...")
    for epoch in range(train_cfg['epochs']):
        total_loss = 0
        for batch_X, batch_Y in loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            optimizer.zero_grad()
            preds, _ = model(batch_X)
            pred_mean = preds[:, :, 0:1]
            pred_second_moment = preds[:, :, 1:2]
            loss_1 = criterion(pred_mean, batch_Y)
            loss_2 = criterion(pred_second_moment, batch_Y ** 2)
            loss = loss_1 + 0.1 * loss_2
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss / len(loader):.6f}")

    os.makedirs(path_cfg['save_dir'], exist_ok=True)
    model_path = os.path.join(path_cfg['save_dir'], path_cfg['next_token_model_name'])
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

def train_mgf_prediction(cfg):
    """Train for MGF prediction."""
    train_cfg = cfg['hyperparameters']
    path_cfg = cfg['paths']

    mgf_data_path = path_cfg.get('mgf_data_path', 'data/ou_mgf.pt')
    print(f"Loading data from {mgf_data_path}...")
    data = torch.load(mgf_data_path, weights_only=False)

    trajectories = data['trajectories']  # (N, seq_len, 1)
    targets = data['targets']  # (N, num_s)

    print(f"Data shapes: trajectories={trajectories.shape}, targets={targets.shape}")

    # Create dataset
    dataset = TensorDataset(trajectories, targets)

    # Train/val split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=train_cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=train_cfg['batch_size'])

    device = torch.device("cuda" if torch.cuda.is_available() and cfg['system']['device'] == "cuda" else "cpu")
    print(f"Using device: {device}")

    model = SimpleTransformer(**cfg['architecture'])
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=train_cfg['learning_rate'])
    criterion = nn.MSELoss()

    print("\nStarting training (MGF prediction)...")
    print("Epoch | Train Loss | Val Loss")
    print("-" * 40)

    best_val_loss = float('inf')

    for epoch in range(train_cfg['epochs']):
        # Training
        model.train()
        total_train_loss = 0
        for batch_X, batch_Y in train_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            optimizer.zero_grad()

            # Model predicts φ(s) from last token
            preds, _ = model(batch_X)  # (batch, seq_len, d_output)
            final_pred = preds[:, -1, :]  # (batch, d_output) - use last timestep

            loss = criterion(final_pred, batch_Y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_X, batch_Y in val_loader:
                batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
                preds, _ = model(batch_X)
                final_pred = preds[:, -1, :]
                loss = criterion(final_pred, batch_Y)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(f"{epoch+1:5d} | {avg_train_loss:10.6f} | {avg_val_loss:9.6f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(path_cfg['save_dir'], exist_ok=True)
            model_path = os.path.join(path_cfg['save_dir'], 'model_mgf.pth')
            torch.save(model.state_dict(), model_path)

    print(f"\nBest validation loss: {best_val_loss:.6f}")
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    print("Select training mode:")
    print("  1. Next-token prediction")
    print("  2. MGF prediction")

    choice = input("Enter choice (1 or 2): ").strip()

    cfg = load_full_config()

    if choice == '1':
        train_next_token(cfg)
    elif choice == '2':
        train_mgf_prediction(cfg)
    else:
        print("Invalid choice. Please enter 1 or 2.")
