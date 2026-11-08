"""
Exercise 23 - SGD with momentum and Adam from scratch  [from-scratch]

Write two optimizers that subclass torch.optim.Optimizer (or a plain class with
params, zero_grad(), step()):
  MySGD(params, lr, momentum=0.0):   v = momentum * v + grad ; p -= lr * v
  MyAdam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
     m = b1*m + (1-b1)*g ; v = b2*v + (1-b2)*g^2
     m_hat = m / (1 - b1^t) ; v_hat = v / (1 - b2^t)
     p -= lr * m_hat / (sqrt(v_hat) + eps)
Use @torch.no_grad() on step(). Verify: starting from the same weights, 10 steps with your
optimizer and the torch one give the same parameters.
Self-check expects these names: MySGD, MyAdam

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
    def _run(make_opt, steps=10):
        torch.manual_seed(0)
        mdl = nn.Linear(5, 1)
        X_, y_ = torch.randn(32, 5), torch.randn(32, 1)
        opt = make_opt(mdl.parameters())
        for _ in range(steps):
            l = ((mdl(X_) - y_) ** 2).mean()
            opt.zero_grad()
            l.backward()
            opt.step()
        return mdl.weight.detach()


    assert torch.allclose(_run(lambda p: MySGD(p, lr=0.1, momentum=0.9)), _run(lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9)), atol=1e-6)
    assert torch.allclose(_run(lambda p: MyAdam(p, lr=0.01)), _run(lambda p: torch.optim.Adam(p, lr=0.01)), atol=1e-6)
    print("All checks passed!")
