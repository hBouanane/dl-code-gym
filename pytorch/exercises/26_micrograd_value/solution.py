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
"""

import math

import torch


class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, k):
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __sub__(self, other): return self + (-other)
    def __truediv__(self, other): return self * other ** -1
    def __repr__(self): return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def backward(self):
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()


a, b = Value(2.0), Value(-3.0)
c = a * b + a ** 2 - b / a
d = (c * 0.5).tanh() + c.relu()
d.backward()

ta = torch.tensor(2.0, requires_grad=True, dtype=torch.float64)
tb = torch.tensor(-3.0, requires_grad=True, dtype=torch.float64)
tc = ta * tb + ta ** 2 - tb / ta
td = torch.tanh(tc * 0.5) + torch.relu(tc)
td.backward()
print(a, b, ta.grad.item(), tb.grad.item())


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
