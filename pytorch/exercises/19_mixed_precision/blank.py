"""
Exercise 19 - Mixed precision (autocast + GradScaler)  [training]

Write a training step that uses automatic mixed precision:
  device = "cuda" if available else "cpu"
  amp_dtype = torch.float16 on cuda, torch.bfloat16 on cpu
  scaler = torch.amp.GradScaler(device, enabled=(device == "cuda"))
  step:
    with torch.autocast(device_type=device, dtype=amp_dtype):
        logits = model(X); loss = loss_fn(logits, y)
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
Train a Linear(20, 2) on random data with labels y = (X[:, 0] > 0) for 100 steps,
accuracy should get > 0.9. Explain to yourself: why does fp16 need a GradScaler but bf16 doesn't?
Self-check expects these names: acc

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
    assert acc > 0.9
    print("All checks passed!")
