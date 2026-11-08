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
"""

import math

import torch
import torch.nn.functional as F


def attention(q, k, v, mask=None):
    d = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ v, weights


def causal_mask(T):
    return torch.tril(torch.ones(T, T, dtype=torch.bool))


torch.manual_seed(0)
q, k, v = torch.randn(2, 4, 5, 8), torch.randn(2, 4, 5, 8), torch.randn(2, 4, 5, 8)
out, w = attention(q, k, v)
out_c, w_c = attention(q, k, v, causal_mask(5))
print(out.shape, w_c[0, 0])


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
