"""
Exercise 01 - Tensor creation  [basics]

Create these tensors (use exactly these variable names):
  zeros     -> 3x4 tensor of zeros
  ones64    -> 2x2 tensor of ones with dtype float64
  ar        -> integers 0..9 (torch.arange)
  lin       -> 5 evenly spaced values from 0 to 1 (torch.linspace)
  rnd       -> 2x3 standard-normal tensor, after torch.manual_seed(0)
  from_list -> tensor from the Python list [[1, 2], [3, 4]]
  eye       -> 3x3 identity matrix
  zl        -> zeros with the same shape and dtype as rnd (torch.zeros_like)
  full7     -> 2x2 tensor filled with 7.0
Then print the shape and dtype of each one.
Before running: predict each shape and dtype in your head.

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

    assert zeros.shape == (3, 4) and zeros.sum() == 0
    assert ones64.dtype == torch.float64
    assert ar.tolist() == list(range(10)) and ar.dtype == torch.int64
    assert torch.allclose(lin, torch.tensor([0, 0.25, 0.5, 0.75, 1.0]))
    assert rnd.shape == (2, 3)
    assert from_list.dtype == torch.int64
    assert torch.equal(eye, torch.eye(3))
    assert zl.shape == rnd.shape and zl.sum() == 0
    assert full7.sum() == 28
    print("All checks passed!")
