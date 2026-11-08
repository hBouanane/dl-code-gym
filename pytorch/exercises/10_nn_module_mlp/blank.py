"""
Exercise 10 - Writing an nn.Module (MLP)  [models]

1. Write class MLP(nn.Module) with __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
     Linear(in, hidden) -> ReLU -> Dropout -> Linear(hidden, hidden) -> ReLU -> Linear(hidden, out)
   and forward(self, x).
2. model = MLP(784, 128, 10). Pass a dummy batch torch.randn(32, 784) -> out shape (32, 10)
3. n_params -> total number of parameters (sum of p.numel())
   n_trainable -> only those with requires_grad
4. Write the same network with nn.Sequential -> seq_model
5. Print the model and loop over model.named_parameters() printing name + shape.

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
    assert out.shape == (32, 10)
    assert n_params == 784 * 128 + 128 + 128 * 128 + 128 + 128 * 10 + 10
    assert n_trainable == n_params
    assert seq_model(torch.randn(4, 784)).shape == (4, 10)
    print("All checks passed!")
