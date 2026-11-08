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

How to practice:
  1. Write everything in blank.py from memory (imports included).
  2. Stuck? Open solution.py, read ONE line, close it, keep going. Note where you got stuck.
  3. Run `python blank.py`. The self-check at the bottom prints "All checks passed!" when you're right.
  4. Redo it tomorrow and the day after, until it comes out without peeking.
"""

# ============ YOUR CODE BELOW ============





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
