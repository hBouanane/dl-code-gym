"""
Exercise 05 - Device-agnostic code & NumPy interop  [basics]

The 3 lines you will write at the top of every project:
  device -> "cuda" if available, else "mps" if available, else "cpu"
Then:
  t       -> tensor [1., 2., 3.] created directly on `device`
  model   -> nn.Linear(3, 1) moved to `device`
  out     -> model(t) ; get it back as a Python float with .item()
  arr     -> t converted to a NumPy array (needs .detach().cpu() first if on GPU)
  from_np -> torch.from_numpy(np.array([1, 2, 3])) ; then modify the numpy array
             and observe that from_np changed too (they share memory!)
  as_f32  -> from_np converted to float32

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
    assert t.device.type == device
    assert next(model.parameters()).device.type == device
    assert isinstance(out, float)
    assert isinstance(arr, np.ndarray)
    assert from_np[0].item() == 100
    assert as_f32.dtype == torch.float32
    print("All checks passed!")
