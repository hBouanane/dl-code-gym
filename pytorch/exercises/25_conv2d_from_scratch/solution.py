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
"""

import torch
import torch.nn.functional as F


def conv2d_loops(x, w, b):
    N, C_in, H, W = x.shape
    C_out, _, kH, kW = w.shape
    H_out, W_out = H - kH + 1, W - kW + 1
    out = torch.zeros(N, C_out, H_out, W_out)
    for n in range(N):
        for c in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    patch = x[n, :, i:i + kH, j:j + kW]
                    out[n, c, i, j] = (patch * w[c]).sum() + b[c]
    return out


def conv2d_unfold(x, w, b, padding=0):
    N, C_in, H, W = x.shape
    C_out, _, kH, kW = w.shape
    H_out, W_out = H + 2 * padding - kH + 1, W + 2 * padding - kW + 1
    cols = F.unfold(x, kernel_size=(kH, kW), padding=padding)   # (N, C_in*kH*kW, L)
    out = w.view(C_out, -1) @ cols + b[:, None]                  # (N, C_out, L)
    return out.view(N, C_out, H_out, W_out)


torch.manual_seed(0)
x = torch.randn(2, 3, 6, 6)
w = torch.randn(4, 3, 3, 3)
b = torch.randn(4)
print(conv2d_loops(x, w, b).shape, conv2d_unfold(x, w, b, padding=1).shape)


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
