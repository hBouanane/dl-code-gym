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
    assert grad_x1 == 7.0 and grad_x2 == 14.0 and x.grad.item() == 0.0
    assert torch.allclose(grad_a, torch.tensor([2., 4., 6.]))
    assert z.requires_grad is False and d.requires_grad is False
    assert torch.allclose(grad_w, a.detach())
    print("All checks passed!")
