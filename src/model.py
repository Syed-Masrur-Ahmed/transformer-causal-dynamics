import torch
import torch.nn as nn
import math


# we embed the scaler input by evaluating the first L Hermite polynomials on it, where L is a hyperparameter.
# this is equivalent to using a truncated orthogonal basis of L2(R, N(0,1)) as input features, which is a natural choice for scalar Gaussian inputs.
class HermiteEmbedding(nn.Module):
    def __init__(self, L, normalize=True):
        """
        L: maximum degree (will return features H_0,...,H_L)
        """
        super().__init__()
        self.L = L
        self.normalize = normalize

        if normalize:
            # Precompute sqrt(n!) for n = 0,...,L as a buffer
            n = torch.arange(L + 1, dtype=torch.float32)
            log_fact = torch.lgamma(n + 1)         # log(n!)
            norm = torch.exp(0.5 * log_fact)       # sqrt(n!)
            self.register_buffer("norm", norm)     # shape: (L+1,)
        else:
            self.norm = None

    def forward(self, x):
        """
        x: Tensor of shape (batch, seq_len, 1) or (batch, seq_len)
           entries should be scalar Gaussian inputs.
        Returns:
            feats: (batch, seq_len, L+1) with Hermite features
        """
        if x.dim() == 3 and x.size(-1) == 1:
            x = x.squeeze(-1)   # (batch, seq_len)
        elif x.dim() != 2:
            raise ValueError("x should have shape (batch, seq_len) or (batch, seq_len, 1)")

        batch_size, seq_len = x.shape

        # H_0 and H_1
        H_list = []
        H0 = torch.ones_like(x)
        H_list.append(H0)

        if self.L >= 1:
            H1 = x
            H_list.append(H1)
            Hm2, Hm1 = H0, H1

            # Recurrence for H_n up to n = L
            for n in range(1, self.L):
                Hn1 = x * Hm1 - n * Hm2   # H_{n+1}(x) = x H_n(x) - n H_{n-1}(x)
                H_list.append(Hn1)
                Hm2, Hm1 = Hm1, Hn1

        # Stack along feature dimension
        feats = torch.stack(H_list, dim=-1)   # (batch, seq_len, L+1)

        if self.normalize:
            feats = feats / self.norm.view(1, 1, -1)  # broadcast sqrt(n!) over batch, seq_len

        return feats
    

class SimpleTransformer(nn.Module):
    def __init__(self, d_input, d_model, n_head, n_layers,
                 max_len=5000, d_output=2, dim_feedforward=None,
                 hermite_degree=None):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model


        self.hermite = HermiteEmbedding(L= d_model-1, normalize=True)

        self.input_projection = nn.Linear( d_model, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        self.layers = nn.ModuleList([
            AttentionBlock(d_model, n_head, dim_feedforward)
            for _ in range(n_layers)
        ])

        # Feedforward network after attention blocks
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm_final = nn.LayerNorm(d_model)

        self.output_projection = nn.Linear(d_model, d_output)

    def forward(self, src):
        """
        src:
            if self.hermite is None:
                shape (batch, seq_len, d_input)
            else:
                shape (batch, seq_len, 1) with scalar Gaussian inputs
        """
        if self.hermite is not None:
            src = self.hermite(src)   # (batch, seq_len, L+1)

        x = self.input_projection(src)
        x = self.pos_encoder(x)

        attention_maps = []
        for layer in self.layers:
            x, attn = layer(x)
            attention_maps.append(attn)

        ff_out = self.feedforward(x)
        x = self.norm_final(x + ff_out)

        predictions = self.output_projection(x)
        return predictions, attention_maps

class AttentionBlock(nn.Module):
    def __init__(self, d_model, n_head, dim_feedforward):
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