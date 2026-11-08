"""
Exercise 09 - Binary classification with BCEWithLogitsLoss  [training]

1. torch.manual_seed(0). Make 2 blobs of 200 points each in 2D:
   class 0 centered at (-2, -2), class 1 at (2, 2)  (randn + center)
   X shape (400, 2), y shape (400, 1) float with 0/1.
2. model = nn.Linear(2, 1)  (outputs LOGITS, no sigmoid inside!)
3. loss_fn = nn.BCEWithLogitsLoss() ; optimizer = Adam lr=0.05
4. Train 200 epochs.
5. Predictions: probs = torch.sigmoid(logits); preds = (probs > 0.5).float()
   accuracy -> fraction of correct predictions (should be > 0.95)
Question to answer for yourself: why BCEWithLogitsLoss and not sigmoid + BCELoss?

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
    assert accuracy > 0.95
    print("All checks passed!")
