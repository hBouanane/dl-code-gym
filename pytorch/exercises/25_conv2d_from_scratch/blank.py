"""
Exercise 25 - Conv2d from scratch (loops, then unfold)  [from-scratch]

Implement 2D convolution (stride 1, no padding first, then add padding):
  conv2d_loops(x, w, b)   x: (N, C_in, H, W), w: (C_out, C_in, kH, kW), b: (C_out,)
     -> nested loops over output positions, sum of (patch * kernel) per out channel
  conv2d_unfold(x, w, b, padding=0)
     -> F.unfold(x, kernel_size, padding=padding) gives (N, C_in*kH*kW, L)
        out = w.view(C_out, -1) @ cols + b[:, None]  -> reshape to (N, C_out, H_out, W_out)
Output size: H_out = H + 2*padding - kH + 1
Compare with F.conv2d.
Self-check expects these names: conv2d_loops, conv2d_unfold

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
    xx, ww, bb = torch.randn(2, 3, 6, 6), torch.randn(4, 3, 3, 3), torch.randn(4)
    assert torch.allclose(conv2d_loops(xx, ww, bb), F.conv2d(xx, ww, bb), atol=1e-4)
    assert torch.allclose(conv2d_unfold(xx, ww, bb), F.conv2d(xx, ww, bb), atol=1e-4)
    assert torch.allclose(conv2d_unfold(xx, ww, bb, padding=1), F.conv2d(xx, ww, bb, padding=1), atol=1e-4)
    print("All checks passed!")
