"""
Exercise 33 - Transformer block (pre-LayerNorm, residuals, MLP)  [attention]

class FeedForward(nn.Module): Linear(d, 4d) -> GELU -> Linear(4d, d) -> Dropout
class Block(nn.Module) (GPT-style, pre-norm):
     x = x + self.attn(self.ln1(x), causal=True)
     x = x + self.ffn(self.ln2(x))
Use your MultiHeadAttention from ex 31 (rewrite it here, don't import!) or nn.MultiheadAttention.
Test: Block(64, 4) on (2, 10, 64) keeps the shape; count its parameters.
Interview: pre-norm vs post-norm? (pre-norm = LayerNorm before the sublayer, more stable training)
Self-check expects these names: Block(d_model, n_heads)
"""

import torch
import torch.nn.functional as F
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x, causal=True):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q, k, v = (t.view(B, T, self.n_heads, -1).transpose(1, 2) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        return self.proj(out.transpose(1, 2).contiguous().view(B, T, C))


class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


block = Block(64, 4)
out = block(torch.randn(2, 10, 64))
n_params = sum(p.numel() for p in block.parameters())
print(out.shape, n_params)


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    import math
    import torch
    import torch.nn.functional as F
    from torch import nn
    blk = Block(64, 4)
    assert blk(torch.randn(2, 10, 64)).shape == (2, 10, 64)
    d = 64
    assert sum(p.numel() for p in blk.parameters()) == 2 * (2 * d) + (3 * d * d + 3 * d) + (d * d + d) + (4 * d * d + 4 * d) + (4 * d * d + d)
    print("All checks passed!")
