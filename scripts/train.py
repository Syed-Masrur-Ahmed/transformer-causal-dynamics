import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import SimpleTransformer
from src.utils import apply_experiment_id_to_paths, load_full_config


def train_mgf_prediction(cfg):
    """Train for MGF prediction."""
    apply_experiment_id_to_paths(cfg)
    train_cfg = cfg['hyperparameters']
    path_cfg = cfg['paths']

    mgf_data_path = path_cfg.get('mgf_data_path', 'data/ou_mgf.pt')
    print(f"Loading data from {mgf_data_path}...")
    data = torch.load(mgf_data_path, weights_only=False)

    trajectories = data['trajectories']  # (N, seq_len, 1)
    targets = data['targets']  # (N, seq_len, order)

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

    print("\nStarting training (Target prediction)...")
    print("Epoch | Train Loss | Val Loss")
    print("-" * 40)

    # Initialize loss tracking
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    os.makedirs(path_cfg['save_dir'], exist_ok=True)
    model_name = path_cfg.get('mgf_model_name', 'model_mgf.pth')
    model_path = os.path.join(path_cfg['save_dir'], model_name)

    for epoch in range(train_cfg['epochs']):
        # Training
        model.train()
        total_train_loss = 0
        for batch_X, batch_Y in train_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            optimizer.zero_grad()

            # Model predicts targets for all timesteps
            preds, _ = model(batch_X)  # (batch, seq_len, d_output)
            
            # Ensure predictions match target dimensions
            if preds.shape[-1] != batch_Y.shape[-1]:
                # If output dimension doesn't match, we may need to adjust the model architecture
                print(f"Warning: Model output dim {preds.shape[-1]} != target dim {batch_Y.shape[-1]}")
            
            loss = criterion(preds, batch_Y)
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
                loss = criterion(preds, batch_Y)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        # Store losses for tracking
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"{epoch+1:5d} | {avg_train_loss:10.6f} | {avg_val_loss:9.6f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_val_loss': best_val_loss,
                'epoch': epoch + 1,
                'config': cfg
            }, model_path)

    # Save final loss history separately for easy access
    loss_history_name = path_cfg.get('loss_history_name', 'loss_history.pt')
    loss_history_path = os.path.join(path_cfg['save_dir'], loss_history_name)
    torch.save({
        'train_losses': train_losses,
        'val_losses': val_losses,
        'epochs': list(range(1, len(train_losses) + 1)),
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'best_val_loss': best_val_loss
    }, loss_history_path)

    print(f"\nBest validation loss: {best_val_loss:.6f}")
    print(f"Model saved to {model_path}")
    print(f"Loss history saved to {loss_history_path}")

if __name__ == "__main__":
    cfg = load_full_config()
    apply_experiment_id_to_paths(cfg)
    train_mgf_prediction(cfg)
