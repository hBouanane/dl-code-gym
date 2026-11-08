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
    assert history["val_acc"][-1] > 0.95
    assert history["train_loss"][-1] < history["train_loss"][0]
    print("All checks passed!")
