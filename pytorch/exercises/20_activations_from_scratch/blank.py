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
    x = torch.randn(4, 5)
    assert torch.allclose(relu(x), F.relu(x))
    assert torch.allclose(sigmoid(x), torch.sigmoid(x), atol=1e-6)
    assert torch.allclose(tanh(x), torch.tanh(x), atol=1e-6)
    assert torch.allclose(gelu(x), F.gelu(x, approximate="tanh"), atol=1e-5)
    assert torch.allclose(softmax(x), F.softmax(x, dim=-1), atol=1e-6)
    assert torch.allclose(log_softmax(x), F.log_softmax(x, dim=-1), atol=1e-5)
    assert not torch.isnan(softmax(torch.tensor([1000., 1001., 1002.]))).any()
    print("All checks passed!")
