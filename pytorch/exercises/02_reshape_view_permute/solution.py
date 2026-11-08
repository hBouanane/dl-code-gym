"""
Exercise 02 - Reshape, view, permute, cat, stack  [basics]

Start from x = torch.arange(24).
  a -> reshape x to (2, 3, 4)
  b -> view a as (6, 4)
  c -> permute a to shape (4, 2, 3)
  d -> flatten c into 1D (hint: c is not contiguous, .view will fail -> why?)
  e -> add a leading dim to a: shape (1, 2, 3, 4)
  f -> remove that dim again: shape (2, 3, 4)
  g -> flatten a from dim 1: shape (2, 12)
  h -> swap the last two dims of a: shape (2, 4, 3)
  i -> concatenate a with itself along dim 0: shape (4, 3, 4)
  j -> stack a with itself along a new dim 0: shape (2, 2, 3, 4)
  k -> reshape a to (4, -1) and let PyTorch infer the second size
"""

import torch

x = torch.arange(24)
a = x.reshape(2, 3, 4)
b = a.view(6, 4)
c = a.permute(2, 0, 1)
d = c.contiguous().view(-1)   # or c.reshape(-1) / c.flatten()
e = a.unsqueeze(0)
f = e.squeeze(0)
g = a.flatten(start_dim=1)
h = a.transpose(1, 2)
i = torch.cat([a, a], dim=0)
j = torch.stack([a, a], dim=0)
k = a.reshape(4, -1)

for name in "abcdefghijk":
    print(name, tuple(eval(name).shape))


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    import math
    import torch
    import torch.nn.functional as F
    from torch import nn
    assert a.shape == (2, 3, 4) and b.shape == (6, 4)
    assert c.shape == (4, 2, 3) and not c.is_contiguous()
    assert d.shape == (24,) and d[1].item() == 4
    assert e.shape == (1, 2, 3, 4) and f.shape == (2, 3, 4)
    assert g.shape == (2, 12) and h.shape == (2, 4, 3)
    assert i.shape == (4, 3, 4) and j.shape == (2, 2, 3, 4)
    assert k.shape == (4, 6)
    print("All checks passed!")
