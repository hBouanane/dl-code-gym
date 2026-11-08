"""
Exercise 22 - nn.Linear from scratch with nn.Parameter  [from-scratch]

Write class MyLinear(nn.Module):
  __init__(in_features, out_features, bias=True):
     self.weight = nn.Parameter of shape (out_features, in_features)
     self.bias   = nn.Parameter of shape (out_features,) or None
     init: uniform(-k, k) with k = 1/sqrt(in_features)  (same as PyTorch)
  forward(x): x @ W.T + b
Check that parameters() finds them (2 params), and that copying the weights of a real
nn.Linear into MyLinear gives identical outputs. Then train it on y = 3x + 2 (reuse exercise 08).
Bonus: register a non-trainable tensor with self.register_buffer("scale", torch.ones(1)).
Self-check expects these names: MyLinear

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
    m, r = MyLinear(4, 3), nn.Linear(4, 3)
    assert len(list(m.parameters())) == 2
    with torch.no_grad():
        m.weight.copy_(r.weight)
        m.bias.copy_(r.bias)
    xx = torch.randn(5, 4)
    assert torch.allclose(m(xx), r(xx), atol=1e-6)
    assert MyLinear(3, 2, bias=False)(torch.randn(1, 3)).shape == (1, 2)
    print("All checks passed!")
