"""
Exercise 11 - Custom Dataset + DataLoader + random_split  [data]

1. Write class ToyDataset(Dataset) that takes X (N, D) and y (N,) tensors in __init__
   and implements __len__ and __getitem__ (returns (X[i], y[i])).
   Optional: accept a `transform` callable applied to X[i].
2. torch.manual_seed(0). X = randn(1000, 20), y = randint(0, 3, (1000,))
   dataset = ToyDataset(X, y)
3. Split into train_ds (80%) and val_ds (20%) with random_split
4. train_loader -> DataLoader(batch_size=32, shuffle=True)
   val_loader   -> DataLoader(batch_size=64, shuffle=False)
5. Grab one batch: xb, yb = next(iter(train_loader)) and print the shapes.
6. n_batches -> len(train_loader) (think: why 25?)
Bonus: also build the same thing with TensorDataset in one line.
"""

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split


class ToyDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.transform:
            x = self.transform(x)
        return x, self.y[idx]


torch.manual_seed(0)
X = torch.randn(1000, 20)
y = torch.randint(0, 3, (1000,))
dataset = ToyDataset(X, y)

train_size = int(0.8 * len(dataset))
train_ds, val_ds = random_split(dataset, [train_size, len(dataset) - train_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

xb, yb = next(iter(train_loader))
n_batches = len(train_loader)
print(xb.shape, yb.shape, n_batches)

tds = TensorDataset(X, y)


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    import math
    import torch
    import torch.nn.functional as F
    from torch import nn
    assert len(dataset) == 1000 and len(train_ds) == 800 and len(val_ds) == 200
    assert xb.shape == (32, 20) and yb.shape == (32,)
    assert n_batches == 25 and len(val_loader) == 4
    assert len(tds) == 1000
    print("All checks passed!")
