import torch
import torch.nn as nn
import math

class SimpleTransformer(nn.Module):
    def __init__(self, d_input, d_model, n_head, n_layers, max_len=5000, d_output=2, dim_feedforward=None):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model
            
        self.input_projection = nn.Linear(d_input, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            AttentionBlock(d_model, n_head, dim_feedforward) for _ in range(n_layers)
        ])
        self.output_projection = nn.Linear(d_model, d_output)

    def forward(self, src):
        x = self.input_projection(src)
        x = self.pos_encoder(x)
        attention_maps = []
        for layer in self.layers:
            x, attn = layer(x)
            attention_maps.append(attn)
        predictions = self.output_projection(x)
        return predictions, attention_maps

class AttentionBlock(nn.Module):
    def __init__(self, d_model, n_head, dim_feedforward):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Causal Mask
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
        mask = mask.to(x.device)
        
        # Attention
        attn_out, attn_weights = self.attn(x, x, x, attn_mask=mask)
        
        # Residual + Norm
        x = self.norm1(x + attn_out)
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x, attn_weights

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Handle both odd and even d_model
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # For odd d_model, pe[:, 1::2] has one fewer column than pe[:, 0::2]
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
            
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :]