import torch
import torch.nn as nn
import math

class SimpleTransformer(nn.Module):
    def __init__(self, d_input, d_model, n_head, n_layers, output_len, max_len=5000):
        super().__init__()
        self.output_len = output_len
        self.input_projection = nn.Linear(d_input, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            AttentionBlock(d_model, n_head) for _ in range(n_layers)
        ])
        self.output_projection = nn.Linear(d_model, 1)

    def forward(self, src):
        # src: (Batch, 100, 1)
        x = self.input_projection(src) # -> (Batch, 100, 64)
        x = self.pos_encoder(x)
        
        # Apply Transformer Blocks
        attention_maps = []
        for layer in self.layers:
            x, attn = layer(x)
            attention_maps.append(attn)
            
        predictions = self.output_projection(x) # -> (Batch, 100, 1)
        
        return predictions, attention_maps

class AttentionBlock(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Causal Mask
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
        mask = mask.to(x.device)
        
        # Attention
        attn_out, attn_weights = self.attn(x, x, x, attn_mask=mask)
        
        # Residual + Norm
        x = self.norm1(x + attn_out)
        
        return x, attn_weights

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :]