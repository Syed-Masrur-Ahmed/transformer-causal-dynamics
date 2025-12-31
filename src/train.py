import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from model import SimpleTransformer

def train():
    print("Loading data...")
    checkpoint = torch.load('data/ou_dataset.pt')
    full_X = checkpoint['X'] # (N, 100, 1)
    
    # Input: Steps 0 to 98
    X_train = full_X[:, :-1, :] 
    # Target: Steps 1 to 99 (Next step prediction)
    Y_train = full_X[:, 1:, :]
    
    dataset = TensorDataset(X_train, Y_train)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleTransformer(d_input=1, d_model=64, n_head=1, n_layers=1)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    print("Starting training...")
    for epoch in range(10):
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
        
    torch.save(model.state_dict(), 'experiments/model_v1.pth')
    print("Model saved to experiments/model_v1.pth")

if __name__ == "__main__":
    train()