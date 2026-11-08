"""
Exercise 07 - Linear regression with raw tensors (no nn, no optim)  [training]

THE first exercise. No nn.Module, no optimizer.
1. torch.manual_seed(42). Generate 100 points X in [0, 1) shape (100, 1)
   and y = 3*X + 2 + noise (noise = 0.1 * randn)
2. Create w and b (shape (1,)) with requires_grad=True
3. Train for 1000 steps with lr = 0.1:
     forward  -> y_pred = X * w + b
     loss     -> mean squared error (write it yourself)
     backward -> loss.backward()
     update   -> inside torch.no_grad(): w -= lr * w.grad ; same for b
     zero     -> w.grad.zero_() ; b.grad.zero_()
4. Print w and b every 100 steps. They should end close to 3 and 2.

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

    assert abs(w.item() - 3) < 0.2, w
    assert abs(b.item() - 2) < 0.2, b
    print("All checks passed!")
