"""
Exercise 31 - Multi-head self-attention module  [attention]

class MultiHeadAttention(nn.Module):
  __init__(d_model, n_heads, dropout=0.0): assert d_model % n_heads == 0
     self.qkv = nn.Linear(d_model, 3 * d_model)   (one projection for q, k, v)
     self.proj = nn.Linear(d_model, d_model)
  forward(x, causal=False): x (B, T, C)
     1. q, k, v = self.qkv(x).split(C, dim=-1)
     2. reshape each to (B, n_heads, T, head_dim): .view(B, T, H, hd).transpose(1, 2)
     3. attention per head (your function from ex 30, or F.scaled_dot_product_attention)
     4. merge heads back: .transpose(1, 2).contiguous().view(B, T, C)
     5. self.proj(...)
Test the output shape, and that with causal=True changing a future token does
not change earlier outputs.
Self-check expects these names: MultiHeadAttention
"""

import math

import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal=False):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        if causal:
            mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = self.dropout(torch.softmax(scores, dim=-1))
        out = weights @ v                                    # (B, H, T, hd)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


torch.manual_seed(0)
mha = MultiHeadAttention(d_model=32, n_heads=4)
x = torch.randn(2, 6, 32)
y = mha(x, causal=True)
x2 = x.clone()
x2[:, -1] += 10.0          # change the last (future) token
y2 = mha(x2, causal=True)
print(y.shape, torch.allclose(y[:, :-1], y2[:, :-1], atol=1e-6))


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    import math
    import torch
    import torch.nn.functional as F
    from torch import nn
    torch.manual_seed(0)
    m = MultiHeadAttention(d_model=32, n_heads=4)
    xx = torch.randn(2, 6, 32)
    y1 = m(xx, causal=True)
    x2 = xx.clone()
    x2[:, -1] += 10.0
    y2_ = m(x2, causal=True)
    assert y1.shape == (2, 6, 32)
    assert torch.allclose(y1[:, :-1], y2_[:, :-1], atol=1e-5)
    assert not torch.allclose(y1[:, -1], y2_[:, -1])
    print("All checks passed!")
