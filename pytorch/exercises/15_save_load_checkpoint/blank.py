"""
Exercise 15 - Save / load models and resume training  [training]

1. model = nn.Linear(4, 2); optimizer = Adam
2. Save only the weights: torch.save(model.state_dict(), "model.pt")
3. Load them into a NEW model: model2 = nn.Linear(4, 2); model2.load_state_dict(torch.load(...))
   -> check both give the same output
4. Full checkpoint for resuming: save a dict with keys
     "epoch", "model_state", "optimizer_state", "loss"  -> "checkpoint.pt"
5. Write resume(path, model, optimizer) that loads the checkpoint into model and
   optimizer and returns the next epoch to train (epoch + 1).
   Use map_location="cpu" when loading (why?).
6. Clean up the files at the end (os.remove).
Self-check expects these names: same (bool), start_epoch

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
    assert same
    assert start_epoch == 6
    print("All checks passed!")
