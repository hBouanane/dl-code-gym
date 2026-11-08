"""
Exercise 09 - Binary classification with BCEWithLogitsLoss  [training]

1. torch.manual_seed(0). Make 2 blobs of 200 points each in 2D:
   class 0 centered at (-2, -2), class 1 at (2, 2)  (randn + center)
   X shape (400, 2), y shape (400, 1) float with 0/1.
2. model = nn.Linear(2, 1)  (outputs LOGITS, no sigmoid inside!)
3. loss_fn = nn.BCEWithLogitsLoss() ; optimizer = Adam lr=0.05
4. Train 200 epochs.
5. Predictions: probs = torch.sigmoid(logits); preds = (probs > 0.5).float()
   accuracy -> fraction of correct predictions (should be > 0.95)
Question to answer for yourself: why BCEWithLogitsLoss and not sigmoid + BCELoss?
"""

import torch
from torch import nn

torch.manual_seed(0)
n = 200
X0 = torch.randn(n, 2) + torch.tensor([-2., -2.])
X1 = torch.randn(n, 2) + torch.tensor([2., 2.])
X = torch.cat([X0, X1])
y = torch.cat([torch.zeros(n, 1), torch.ones(n, 1)])

model = nn.Linear(2, 1)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

for epoch in range(200):
    logits = model(X)
    loss = loss_fn(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

with torch.inference_mode():
    probs = torch.sigmoid(model(X))
    preds = (probs > 0.5).float()
    accuracy = (preds == y).float().mean().item()
print(f"loss {loss.item():.4f} accuracy {accuracy:.3f}")
# Answer: BCEWithLogitsLoss fuses sigmoid + log in a numerically stable way (log-sum-exp trick).


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    import math
    import torch
    import torch.nn.functional as F
    from torch import nn
    assert accuracy > 0.95
    print("All checks passed!")
