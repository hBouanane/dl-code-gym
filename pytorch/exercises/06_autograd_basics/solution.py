"""
Exercise 06 - Autograd basics  [basics]

1. x = tensor(2.0, requires_grad=True). y = x**2 + 3*x. Call backward.
   grad_x1 -> x.grad (should be 2*2 + 3 = 7)
2. Call y2 = x**2 + 3*x again and backward again WITHOUT zeroing.
   grad_x2 -> x.grad now (gradients ACCUMULATE -> 14). Then zero it with x.grad.zero_()
3. a = tensor([1., 2., 3.], requires_grad=True); loss = (a ** 2).sum(); backward.
   grad_a -> a.grad (should be 2*a)
4. Inside torch.no_grad(): z = a * 2   -> z.requires_grad should be False
5. d = a.detach() -> d.requires_grad is False
6. w = torch.randn(3, requires_grad=True); compute out = (w * a).sum() and get the
   gradient with torch.autograd.grad(out, w) instead of .backward()
   grad_w -> the result (should equal a)
"""

import torch

x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x
y.backward()
grad_x1 = x.grad.item()

y2 = x ** 2 + 3 * x
y2.backward()
grad_x2 = x.grad.item()
x.grad.zero_()

a = torch.tensor([1., 2., 3.], requires_grad=True)
loss = (a ** 2).sum()
loss.backward()
grad_a = a.grad.clone()

with torch.no_grad():
    z = a * 2

d = a.detach()

w = torch.randn(3, requires_grad=True)
out = (w * a).sum()
(grad_w,) = torch.autograd.grad(out, w)

print(grad_x1, grad_x2, grad_a, z.requires_grad, d.requires_grad, grad_w)


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    import math
    import torch
    import torch.nn.functional as F
    from torch import nn
    assert grad_x1 == 7.0 and grad_x2 == 14.0 and x.grad.item() == 0.0
    assert torch.allclose(grad_a, torch.tensor([2., 4., 6.]))
    assert z.requires_grad is False and d.requires_grad is False
    assert torch.allclose(grad_w, a.detach())
    print("All checks passed!")
