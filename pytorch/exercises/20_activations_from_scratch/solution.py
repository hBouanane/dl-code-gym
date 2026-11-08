"""
Exercise 20 - Activations from scratch (interview)  [from-scratch]

Implement using only basic tensor ops (no torch.relu / F.softmax etc.):
  relu(x), sigmoid(x), tanh(x), gelu(x)  (tanh approximation:
      0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x**3))) ),
  softmax(x, dim=-1)      -> NUMERICALLY STABLE: subtract the max first
  log_softmax(x, dim=-1)  -> x - max - log(sum(exp(x - max)))
Test: softmax(torch.tensor([1000., 1001., 1002.])) must not return nan.
Interview question: why subtract the max? (exp(1000) overflows; softmax is shift-invariant)
Self-check expects these names: relu, sigmoid, tanh, gelu, softmax, log_softmax
"""

import math

import torch
import torch.nn.functional as F


def relu(x):
    return torch.clamp(x, min=0)          # or x * (x > 0)


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def tanh(x):
    return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


def softmax(x, dim=-1):
    x = x - x.max(dim=dim, keepdim=True).values
    e = torch.exp(x)
    return e / e.sum(dim=dim, keepdim=True)


def log_softmax(x, dim=-1):
    x = x - x.max(dim=dim, keepdim=True).values
    return x - torch.log(torch.exp(x).sum(dim=dim, keepdim=True))


print(softmax(torch.tensor([1000., 1001., 1002.])))


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    import math
    import torch
    import torch.nn.functional as F
    from torch import nn
    x = torch.randn(4, 5)
    assert torch.allclose(relu(x), F.relu(x))
    assert torch.allclose(sigmoid(x), torch.sigmoid(x), atol=1e-6)
    assert torch.allclose(tanh(x), torch.tanh(x), atol=1e-6)
    assert torch.allclose(gelu(x), F.gelu(x, approximate="tanh"), atol=1e-5)
    assert torch.allclose(softmax(x), F.softmax(x, dim=-1), atol=1e-6)
    assert torch.allclose(log_softmax(x), F.log_softmax(x, dim=-1), atol=1e-5)
    assert not torch.isnan(softmax(torch.tensor([1000., 1001., 1002.]))).any()
    print("All checks passed!")
