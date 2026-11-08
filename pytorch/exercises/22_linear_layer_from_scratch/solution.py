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
"""

import math

import torch
from torch import nn


class MyLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        k = 1 / math.sqrt(in_features)
        self.weight = nn.Parameter(torch.empty(out_features, in_features).uniform_(-k, k))
        self.bias = nn.Parameter(torch.empty(out_features).uniform_(-k, k)) if bias else None
        self.register_buffer("scale", torch.ones(1))

    def forward(self, x):
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out * self.scale


torch.manual_seed(0)
mine = MyLinear(4, 3)
ref = nn.Linear(4, 3)
with torch.no_grad():
    mine.weight.copy_(ref.weight)
    mine.bias.copy_(ref.bias)
x = torch.randn(5, 4)
print(torch.allclose(mine(x), ref(x)))

X = torch.rand(100, 1)
y = 3 * X + 2
lin = MyLinear(1, 1)
opt = torch.optim.SGD(lin.parameters(), lr=0.1)
for _ in range(2000):
    loss = ((lin(X) - y) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
print(lin.weight.item(), lin.bias.item())


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
