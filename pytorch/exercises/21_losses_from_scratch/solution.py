"""
Exercise 21 - Loss functions from scratch: MSE, BCE, CrossEntropy  [from-scratch]

Implement (mean reduction):
  mse(pred, target)
  bce_with_logits(logits, target)       -> stable form: max(z,0) - z*t + log(1 + exp(-|z|))
  cross_entropy(logits, targets)        -> logits (N, C), targets (N,) integer class ids
                                           = mean of -log_softmax(logits)[i, targets[i]]
                                           use torch.logsumexp; pick with indexing or gather
  cross_entropy_ls(logits, targets, eps=0.1)  -> with label smoothing:
       (1-eps) * NLL + eps * mean over classes of (-log_softmax)
Compare each with the built-in F.mse_loss / F.binary_cross_entropy_with_logits / F.cross_entropy.
Self-check expects these names: mse, bce_with_logits, cross_entropy, cross_entropy_ls
"""

import torch
import torch.nn.functional as F


def mse(pred, target):
    return ((pred - target) ** 2).mean()


def bce_with_logits(z, t):
    return (z.clamp(min=0) - z * t + torch.log1p(torch.exp(-z.abs()))).mean()


def cross_entropy(logits, targets):
    log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return -log_probs[torch.arange(len(targets)), targets].mean()


def cross_entropy_ls(logits, targets, eps=0.1):
    log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    smooth = -log_probs.mean(dim=1)
    return ((1 - eps) * nll + eps * smooth).mean()


torch.manual_seed(0)
logits = torch.randn(8, 5)
targets = torch.randint(0, 5, (8,))
print(cross_entropy(logits, targets), F.cross_entropy(logits, targets))


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    import math
    import torch
    import torch.nn.functional as F
    from torch import nn
    torch.manual_seed(1)
    p, t = torch.randn(10), torch.randn(10)
    assert torch.allclose(mse(p, t), F.mse_loss(p, t))
    z, tb = torch.randn(10) * 5, torch.randint(0, 2, (10,)).float()
    assert torch.allclose(bce_with_logits(z, tb), F.binary_cross_entropy_with_logits(z, tb), atol=1e-6)
    lg, tg = torch.randn(8, 5), torch.randint(0, 5, (8,))
    assert torch.allclose(cross_entropy(lg, tg), F.cross_entropy(lg, tg), atol=1e-6)
    assert torch.allclose(cross_entropy_ls(lg, tg), F.cross_entropy(lg, tg, label_smoothing=0.1), atol=1e-6)
    print("All checks passed!")
