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
    assert len(dataset) == 1000 and len(train_ds) == 800 and len(val_ds) == 200
    assert xb.shape == (32, 20) and yb.shape == (32,)
    assert n_batches == 25 and len(val_loader) == 4
    assert len(tds) == 1000
    print("All checks passed!")
