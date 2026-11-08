"""
Exercise 14 - A small CNN (conv -> pool -> fc)  [real-data]

1. Write class SmallCNN(nn.Module) for 1x28x28 images, 10 classes:
     block1: Conv2d(1, 32, 3, padding=1) -> ReLU -> MaxPool2d(2)    -> 32x14x14
     block2: Conv2d(32, 64, 3, padding=1) -> ReLU -> MaxPool2d(2)   -> 64x7x7
     classifier: Flatten -> Linear(64*7*7, 128) -> ReLU -> Dropout(0.25) -> Linear(128, 10)
   Shape trick: compute the flatten size by passing a dummy torch.zeros(1, 1, 28, 28).
2. Train it on FashionMNIST (datasets.FashionMNIST, ~30 MB) for 1 epoch, report test accuracy (> 0.85).
Formula to remember: out = (in + 2*padding - kernel) // stride + 1
Bonus: adapt it to CIFAR-10 (3x32x32) — what changes?
Self-check expects these names: SmallCNN, test_acc

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
    assert SmallCNN()(torch.zeros(2, 1, 28, 28)).shape == (2, 10)
    assert test_acc > 0.85
    print("All checks passed!")
