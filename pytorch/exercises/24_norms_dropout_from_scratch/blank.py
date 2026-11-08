"""
Exercise 24 - Dropout, LayerNorm, BatchNorm1d from scratch  [from-scratch]

1. MyDropout(p): in training mode, zero each element with prob p and scale the rest by 1/(1-p)
   ("inverted dropout"); in eval mode return x unchanged (use self.training).
2. MyLayerNorm(dim, eps=1e-5): normalize over the LAST dim (mean/var per sample),
   learnable gamma (ones) and beta (zeros). Use the biased variance (unbiased=False).
3. MyBatchNorm1d(num_features, eps=1e-5, momentum=0.1): input (N, C)
   training: normalize with batch mean/var over dim 0, update running stats:
       running = (1 - momentum) * running + momentum * batch_stat
       (PyTorch uses the UNBIASED batch var for running_var!)
   eval: normalize with running_mean / running_var.
   Keep running stats with register_buffer.
Interview question: LayerNorm vs BatchNorm — what does each normalize over, and why do
Transformers use LayerNorm?
Self-check expects these names: MyDropout, MyLayerNorm, MyBatchNorm1d

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
    d = MyDropout(0.5)
    d.train()
    out = d(torch.ones(10000))
    assert abs(out.mean().item() - 1.0) < 0.05 and set(out.unique().tolist()) <= {0.0, 2.0}
    d.eval()
    assert torch.equal(d(torch.ones(5)), torch.ones(5))
    xx = torch.randn(16, 8) * 3 + 1
    assert torch.allclose(MyLayerNorm(8)(xx), nn.LayerNorm(8)(xx), atol=1e-5)
    mine, ref = MyBatchNorm1d(8), nn.BatchNorm1d(8)
    for _ in range(3):
        xb = torch.randn(16, 8) * 2 + 3
        assert torch.allclose(mine(xb), ref(xb), atol=1e-5)
    assert torch.allclose(mine.running_var, ref.running_var, atol=1e-5)
    mine.eval(); ref.eval()
    assert torch.allclose(mine(xx), ref(xx), atol=1e-5)
    print("All checks passed!")
