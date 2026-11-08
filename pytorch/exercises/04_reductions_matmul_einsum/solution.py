"""
Exercise 04 - Reductions, matmul, bmm, einsum  [basics]

torch.manual_seed(0); A = randn(2, 3); B = randn(3, 4); X = randn(8, 2, 3); Y = randn(8, 3, 5)
  col_sum   -> sum of A over dim 0 (shape (3,))
  row_mean  -> mean of A over dim 1 with keepdim (shape (2, 1))
  AB        -> A @ B (shape (2, 4))
  XY        -> batched matmul of X and Y (shape (8, 2, 5)) with torch.bmm
  AB_ein    -> A @ B but written with torch.einsum
  XY_ein    -> batched matmul with torch.einsum
  dot       -> dot product of [1, 2, 3] and [4, 5, 6]
  outer     -> outer product of [1, 2] and [3, 4, 5] (shape (2, 3))
  fro       -> Frobenius norm of A (torch.linalg.norm)
  sm        -> softmax of A over the last dim (rows sum to 1)
  mx_val, mx_idx -> max of A over dim 1 (values and indices)
"""

import torch

torch.manual_seed(0)
A = torch.randn(2, 3)
B = torch.randn(3, 4)
X = torch.randn(8, 2, 3)
Y = torch.randn(8, 3, 5)

col_sum = A.sum(dim=0)
row_mean = A.mean(dim=1, keepdim=True)
AB = A @ B
XY = torch.bmm(X, Y)
AB_ein = torch.einsum("ij,jk->ik", A, B)
XY_ein = torch.einsum("bij,bjk->bik", X, Y)
dot = torch.dot(torch.tensor([1., 2., 3.]), torch.tensor([4., 5., 6.]))
outer = torch.outer(torch.tensor([1., 2.]), torch.tensor([3., 4., 5.]))
fro = torch.linalg.norm(A)
sm = torch.softmax(A, dim=-1)
mx_val, mx_idx = A.max(dim=1)

print(AB.shape, XY.shape, dot, outer, fro, sm.sum(-1), mx_val, mx_idx, sep="\n")


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    import math
    import torch
    import torch.nn.functional as F
    from torch import nn
    assert col_sum.shape == (3,) and row_mean.shape == (2, 1)
    assert AB.shape == (2, 4) and torch.allclose(AB, AB_ein, atol=1e-6)
    assert XY.shape == (8, 2, 5) and torch.allclose(XY, XY_ein, atol=1e-5)
    assert dot.item() == 32
    assert outer.tolist() == [[3, 4, 5], [6, 8, 10]]
    assert torch.isclose(fro, (A ** 2).sum().sqrt())
    assert torch.allclose(sm.sum(-1), torch.ones(2))
    assert torch.equal(mx_idx, A.argmax(dim=1))
    print("All checks passed!")
