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
"""

import torch
from torch import nn


class MySGD:
    def __init__(self, params, lr, momentum=0.0):
        self.params = list(params)
        self.lr, self.momentum = lr, momentum
        self.velocity = [torch.zeros_like(p) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self):
        for p, v in zip(self.params, self.velocity):
            if p.grad is None:
                continue
            v.mul_(self.momentum).add_(p.grad)
            p -= self.lr * v


class MyAdam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr, (self.b1, self.b2), self.eps = lr, betas, eps
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self):
        self.t += 1
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad
            m.mul_(self.b1).add_((1 - self.b1) * g)
            v.mul_(self.b2).add_((1 - self.b2) * g * g)
            m_hat = m / (1 - self.b1 ** self.t)
            v_hat = v / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (v_hat.sqrt() + self.eps)


def run(opt_factory, steps=10):
    torch.manual_seed(0)
    model = nn.Linear(5, 1)
    X, y = torch.randn(32, 5), torch.randn(32, 1)
    opt = opt_factory(model.parameters())
    for _ in range(steps):
        loss = ((model(X) - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.weight.detach()


w_my_sgd = run(lambda p: MySGD(p, lr=0.1, momentum=0.9))
w_sgd = run(lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9))
w_my_adam = run(lambda p: MyAdam(p, lr=0.01))
w_adam = run(lambda p: torch.optim.Adam(p, lr=0.01))
print(w_my_sgd, w_sgd, w_my_adam, w_adam, sep="\n")


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
