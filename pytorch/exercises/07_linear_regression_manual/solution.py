"""
Exercise 07 - Linear regression with raw tensors (no nn, no optim)  [training]

THE first exercise. No nn.Module, no optimizer.
1. torch.manual_seed(42). Generate 100 points X in [0, 1) shape (100, 1)
   and y = 3*X + 2 + noise (noise = 0.1 * randn)
2. Create w and b (shape (1,)) with requires_grad=True
3. Train for 1000 steps with lr = 0.1:
     forward  -> y_pred = X * w + b
     loss     -> mean squared error (write it yourself)
     backward -> loss.backward()
     update   -> inside torch.no_grad(): w -= lr * w.grad ; same for b
     zero     -> w.grad.zero_() ; b.grad.zero_()
4. Print w and b every 100 steps. They should end close to 3 and 2.
"""

import torch

torch.manual_seed(42)
X = torch.rand(100, 1)
y = 3 * X + 2 + 0.1 * torch.randn(100, 1)

w = torch.randn(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
lr = 0.1

for step in range(1000):
    y_pred = X * w + b
    loss = ((y_pred - y) ** 2).mean()
    loss.backward()
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
    w.grad.zero_()
    b.grad.zero_()
    if step % 100 == 0:
        print(f"step {step:4d} loss {loss.item():.4f} w {w.item():.3f} b {b.item():.3f}")

print(f"final w={w.item():.3f} b={b.item():.3f}")


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    import math
    import torch
    import torch.nn.functional as F
    from torch import nn
    assert abs(w.item() - 3) < 0.2, w
    assert abs(b.item() - 2) < 0.2, b
    print("All checks passed!")
