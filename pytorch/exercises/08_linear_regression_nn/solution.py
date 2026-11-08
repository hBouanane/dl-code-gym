"""
Exercise 08 - The standard loop: nn.Linear + optim + loss  [training]

Same problem as 07 (y = 3x + 2 + noise), now the "real" way.
  model     -> nn.Linear(1, 1)
  loss_fn   -> nn.MSELoss()
  optimizer -> torch.optim.SGD(model.parameters(), lr=0.1)
Loop for 1000 epochs, the 5 lines you must know by heart:
  1. y_pred = model(X)
  2. loss = loss_fn(y_pred, y)
  3. optimizer.zero_grad()
  4. loss.backward()
  5. optimizer.step()
At the end print model.weight and model.bias (close to 3 and 2).
Bonus: put model.train() before the loop, then predict on X_new = [[0.], [1.]]
with model.eval() + torch.inference_mode().
"""

import torch
from torch import nn

torch.manual_seed(42)
X = torch.rand(100, 1)
y = 3 * X + 2 + 0.1 * torch.randn(100, 1)

model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

model.train()
for epoch in range(1000):
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 200 == 0:
        print(f"epoch {epoch} loss {loss.item():.4f}")

print("weight", model.weight.item(), "bias", model.bias.item())

model.eval()
with torch.inference_mode():
    X_new = torch.tensor([[0.], [1.]])
    print(model(X_new))


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    import math
    import torch
    import torch.nn.functional as F
    from torch import nn
    assert abs(model.weight.item() - 3) < 0.2
    assert abs(model.bias.item() - 2) < 0.2
    print("All checks passed!")
