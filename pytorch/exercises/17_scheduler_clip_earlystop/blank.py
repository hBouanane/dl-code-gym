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
    assert best_state is not None
    assert lrs[-1] < 0.05
    with torch.inference_mode():
        assert abs(loss_fn(model(X_va), y_va).item() - best_val_loss) < 1e-6
    print("All checks passed!")
