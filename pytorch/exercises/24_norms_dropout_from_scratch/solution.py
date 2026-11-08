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
"""

import torch
from torch import nn


class MyDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        mask = (torch.rand_like(x) > self.p).float()
        return x * mask / (1 - self.p)


class MyLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta


class MyBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps, self.momentum = eps, momentum
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean)
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * x.var(dim=0, unbiased=True))
        else:
            mean, var = self.running_mean, self.running_var
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta


torch.manual_seed(0)
x = torch.randn(16, 8) * 3 + 1
print(MyLayerNorm(8)(x)[0], nn.LayerNorm(8)(x)[0])
# LayerNorm: over features of each sample (batch-size independent, works for variable-length seqs).
# BatchNorm: over the batch for each feature (depends on batch statistics).


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
