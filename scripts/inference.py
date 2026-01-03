import torch

def autoregressive_predict(model, history, future_steps=50, device='cpu'):
    model.eval() 
    current_seq = history.clone().to(device) # Shape: (1, 100, 1)
    predictions = []
    
    with torch.no_grad(): 
        for _ in range(future_steps):
            output, maps = model(current_seq)
            next_val = output[:, -1, :]
            
            predictions.append(next_val.item())
            next_val_reshaped = next_val.unsqueeze(1) # (1, 1, 1)
            current_seq = torch.cat([current_seq[:, 1:, :], next_val_reshaped], dim=1)
            
    return predictions
