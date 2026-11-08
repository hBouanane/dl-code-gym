"""
Exercise 32 - Sinusoidal & learned positional encodings  [attention]

1. sinusoidal_pe(max_len, d_model) -> (max_len, d_model):
     pe[pos, 2i]   = sin(pos / 10000^(2i/d_model))
     pe[pos, 2i+1] = cos(pos / 10000^(2i/d_model))
   vectorized: position = arange(max_len)[:, None]; div = exp(arange(0, d, 2) * (-log(10000)/d))
2. class PositionalEncoding(nn.Module) that stores pe with register_buffer and in
   forward(x) (B, T, C) returns x + pe[:T]
3. Learned version: class LearnedPositionalEmbedding with nn.Embedding(max_len, d_model);
   forward(x) adds self.emb(torch.arange(T, device=x.device))
Self-check expects these names: sinusoidal_pe, PositionalEncoding, LearnedPositionalEmbedding

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
    pe_ = sinusoidal_pe(50, 16)
    assert pe_.shape == (50, 16)
    assert torch.allclose(pe_[0, 0::2], torch.zeros(8)) and torch.allclose(pe_[0, 1::2], torch.ones(8))
    assert abs(pe_[3, 2].item() - math.sin(3 / 10000 ** (2 / 16))) < 1e-6
    xx = torch.zeros(2, 10, 16)
    assert torch.allclose(PositionalEncoding(16)(xx)[0], pe_[:10], atol=1e-6)
    assert len(list(PositionalEncoding(16).parameters())) == 0
    assert LearnedPositionalEmbedding(16)(xx).shape == (2, 10, 16)
    print("All checks passed!")
