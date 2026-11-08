"""
Exercise 17 - LR scheduler, gradient clipping, early stopping  [training]

Upgrade the training loop with the 3 tricks used everywhere:
  scheduler -> torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
               call scheduler.step() once per EPOCH (after optimizer.step()s)
  clipping  -> torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
               between loss.backward() and optimizer.step()
  early stop-> track best_val_loss; patience = 3; if no improvement for 3 epochs, break.
               Keep best_state = copy.deepcopy(model.state_dict()) and reload it at the end.
Data: torch.manual_seed(0); X = randn(500, 10); y = X @ randn(10, 1) + 0.1*randn(500, 1)
Split first 400 train / last 100 val (no DataLoader needed, full batch is fine).
Model: Linear(10, 32) -> ReLU -> Linear(32, 1), SGD lr=0.05, up to 200 epochs.
Record the learning rate each epoch in `lrs` (scheduler.get_last_lr()[0]).
Self-check expects these names: model, loss_fn, X_va, y_va, best_state, best_val_loss, lrs
"""

import copy

import torch
from torch import nn

torch.manual_seed(0)
X = torch.randn(500, 10)
y = X @ torch.randn(10, 1) + 0.1 * torch.randn(500, 1)
X_tr, y_tr, X_va, y_va = X[:400], y[:400], X[400:], y[400:]

model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))
loss_fn = nn.MSELoss()
epochs = 200
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

best_val_loss, best_state, patience, bad_epochs = float("inf"), None, 3, 0
lrs = []
for epoch in range(epochs):
    model.train()
    loss = loss_fn(model(X_tr), y_tr)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    lrs.append(scheduler.get_last_lr()[0])

    model.eval()
    with torch.inference_mode():
        val_loss = loss_fn(model(X_va), y_va).item()

    if val_loss < best_val_loss:
        best_val_loss, bad_epochs = val_loss, 0
        best_state = copy.deepcopy(model.state_dict())
    else:
        bad_epochs += 1
        if bad_epochs >= patience:
            print(f"early stopping at epoch {epoch}")
            break

model.load_state_dict(best_state)
print(f"best val loss {best_val_loss:.4f}, epochs run {len(lrs)}")


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    import math
    import torch
    import torch.nn.functional as F
    from torch import nn
    assert best_state is not None
    assert lrs[-1] < 0.05
    with torch.inference_mode():
        assert abs(loss_fn(model(X_va), y_va).item() - best_val_loss) < 1e-6
    print("All checks passed!")
