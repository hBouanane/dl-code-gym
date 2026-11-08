"""
Exercise 30 - Scaled dot-product attention (+ causal mask)  [attention]

THE most asked interview snippet.
attention(q, k, v, mask=None) with q, k, v of shape (..., seq, d):
   scores  = q @ k.transpose(-2, -1) / sqrt(d)
   if mask is not None: scores = scores.masked_fill(mask == 0, float("-inf"))   (mask: 1 = keep)
   weights = softmax(scores, dim=-1)
   return weights @ v, weights
Then:
  causal_mask(T) -> lower-triangular (T, T) boolean
  Verify against F.scaled_dot_product_attention(q, k, v) and (..., is_causal=True).
Interview question: why divide by sqrt(d)? (dot products grow with d -> softmax saturates -> tiny grads)
Self-check expects these names: attention, causal_mask

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
    qq, kk, vv = torch.randn(2, 4, 5, 8), torch.randn(2, 4, 5, 8), torch.randn(2, 4, 5, 8)
    o, wt = attention(qq, kk, vv)
    oc, wc = attention(qq, kk, vv, causal_mask(5))
    assert torch.allclose(o, F.scaled_dot_product_attention(qq, kk, vv), atol=1e-5)
    assert torch.allclose(oc, F.scaled_dot_product_attention(qq, kk, vv, is_causal=True), atol=1e-5)
    assert torch.allclose(wt.sum(-1), torch.ones(2, 4, 5))
    assert wc[0, 0, 0, 1:].abs().sum() == 0
    print("All checks passed!")
