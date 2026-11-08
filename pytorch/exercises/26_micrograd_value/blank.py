"""
Exercise 26 - A tiny autograd engine (scalar Value class)  [from-scratch]

Rebuild the core of Karpathy's micrograd. Class Value:
  __init__(self, data, _children=(), _op="")  stores data, grad=0.0, _backward=lambda: None, _prev=set(children)
  __add__, __mul__, __pow__ (int/float exponent), tanh(), relu()
  __radd__, __rmul__, __neg__, __sub__, __truediv__ (via pow -1)
  backward(): build a topological order of the graph, set self.grad = 1, then call
              _backward of each node in REVERSE topological order.
Each op defines its local derivative in a closure, and ACCUMULATES (+=) into child.grad.
Test: a = Value(2.0); b = Value(-3.0); c = a*b + a**2 ; d = c.tanh()... compare with torch.
Self-check expects these names: Value

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
    va, vb = Value(2.0), Value(-3.0)
    vc = va * vb + va ** 2 - vb / va
    vd = (vc * 0.5).tanh() + vc.relu()
    vd.backward()
    ta = torch.tensor(2.0, requires_grad=True, dtype=torch.float64)
    tb = torch.tensor(-3.0, requires_grad=True, dtype=torch.float64)
    tc = ta * tb + ta ** 2 - tb / ta
    td = torch.tanh(tc * 0.5) + torch.relu(tc)
    td.backward()
    assert abs(vd.data - td.item()) < 1e-6
    assert abs(va.grad - ta.grad.item()) < 1e-6 and abs(vb.grad - tb.grad.item()) < 1e-6
    print("All checks passed!")
