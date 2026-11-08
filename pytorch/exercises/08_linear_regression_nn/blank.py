"""
Exercise 08 - The standard loop: nn.Linear + optim + loss  [training]

Same problem as 07 (y = 3x + 2 + noise), now the "real" way.
  model     -> nn.Linear(1, 1)
  loss_fn   -> nn.MSELoss()
  optimizer -> torch.optim.SGD(model.parameters(), lr=0.1)
Loop for 1000 epochs, the 5 lines you must know by heart:
  1. y_pred = model(X)
  2. loss = loss_fn(y_pred, y)
  3. optimizer.zero_grad()
  4. loss.backward()
  5. optimizer.step()
At the end print model.weight and model.bias (close to 3 and 2).
Bonus: put model.train() before the loop, then predict on X_new = [[0.], [1.]]
with model.eval() + torch.inference_mode().

How to practice:
  1. Write everything in blank.py from memory (imports included).
  2. Stuck? Open solution.py, read ONE line, close it, keep going. Note where you got stuck.
  3. Run `python blank.py`. The self-check at the bottom prints "All checks passed!" when you're right.
  4. Redo it tomorrow and the day after, until it comes out without peeking.
"""

# ============ YOUR CODE BELOW ============





# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    import torch
    from torch import nn

    assert abs(model.weight.item() - 3) < 0.2
    assert abs(model.bias.item() - 2) < 0.2
    print("All checks passed!")
