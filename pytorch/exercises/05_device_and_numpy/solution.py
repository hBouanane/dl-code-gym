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
"""

import numpy as np
import torch
from torch import nn

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print("Using device:", device)

t = torch.tensor([1., 2., 3.], device=device)
model = nn.Linear(3, 1).to(device)
out = model(t).item()
arr = t.detach().cpu().numpy()

np_arr = np.array([1, 2, 3])
from_np = torch.from_numpy(np_arr)
np_arr[0] = 100
print("shared memory:", from_np)
as_f32 = from_np.to(torch.float32)   # or .float()
print(out, arr, as_f32.dtype)


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
