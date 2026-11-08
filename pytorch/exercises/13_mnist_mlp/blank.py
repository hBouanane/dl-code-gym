"""
Exercise 13 - Full pipeline: MLP on MNIST (torchvision)  [real-data]

Full pipeline on real data (downloads ~10 MB into ./data).
1. transform = transforms.ToTensor() (optionally Normalize((0.1307,), (0.3081,)))
2. train_data / test_data = datasets.MNIST(root="data", train=True/False, download=True, transform=...)
3. DataLoaders batch 64 (shuffle train only)
4. Model: nn.Flatten() -> Linear(784, 256) -> ReLU -> Linear(256, 10)
5. CrossEntropyLoss + Adam(lr=1e-3), train 2 epochs with your train/eval functions (rewrite them!)
6. Print test accuracy (should be > 0.95)
7. Predict a single image: test_data[0] -> add batch dim -> argmax -> compare to label
Self-check expects these names: test_acc

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
    assert test_acc > 0.95
    print("All checks passed!")
