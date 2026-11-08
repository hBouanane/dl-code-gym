"""
Exercise 12 - train_one_epoch() + evaluate() — the reusable skeleton  [training]

THE skeleton you will reuse in every project and interview. Write:
  train_one_epoch(model, loader, loss_fn, optimizer, device) -> (avg_loss, accuracy)
     model.train(); for each batch: to(device), forward, loss, zero_grad, backward, step;
     accumulate loss * batch_size and number of correct predictions.
  evaluate(model, loader, loss_fn, device) -> (avg_loss, accuracy)
     model.eval(); torch.inference_mode(); no backward, no step.
Then:
  torch.manual_seed(0). Data: 3 classes, 300 points each in 2D, centers (0,4), (-4,-2), (4,-2)
  Split 80/20, DataLoaders (batch 32), model = small MLP 2 -> 32 -> 3,
  CrossEntropyLoss, Adam lr=1e-2. Train 10 epochs and print train/val loss & acc each epoch.
  Keep a `history` dict with lists "train_loss", "val_loss", "val_acc".
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += X.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.inference_mode():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            total_loss += loss_fn(logits, y).item() * X.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += X.size(0)
    return total_loss / total, correct / total


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
centers = torch.tensor([[0., 4.], [-4., -2.], [4., -2.]])
X = torch.cat([torch.randn(300, 2) + c for c in centers])
y = torch.arange(3).repeat_interleave(300)

ds = TensorDataset(X, y)
train_ds, val_ds = random_split(ds, [720, 180])
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

model = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 3)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

history = {"train_loss": [], "val_loss": [], "val_acc": []}
for epoch in range(10):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
    va_loss, va_acc = evaluate(model, val_loader, loss_fn, device)
    history["train_loss"].append(tr_loss)
    history["val_loss"].append(va_loss)
    history["val_acc"].append(va_acc)
    print(f"epoch {epoch+1:2d} | train {tr_loss:.3f}/{tr_acc:.3f} | val {va_loss:.3f}/{va_acc:.3f}")


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    import math
    import torch
    import torch.nn.functional as F
    from torch import nn
    assert history["val_acc"][-1] > 0.95
    assert history["train_loss"][-1] < history["train_loss"][0]
    print("All checks passed!")
