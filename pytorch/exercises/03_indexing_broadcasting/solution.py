"""
Exercise 03 - Indexing, masks, broadcasting  [basics]

x = torch.arange(1, 13).reshape(3, 4)   (values 1..12)
  first_row  -> first row of x
  last_col   -> last column of x
  sub        -> rows 0-1 and columns 1-2 (shape (2, 2))
  big        -> 1D tensor of all elements > 6 (boolean mask)
  rows02     -> rows 0 and 2 (fancy indexing with a list)
  plus_row   -> x + [10, 20, 30, 40]  (broadcast a row vector)
  row_norm   -> each row divided by its row sum (use keepdim=True), float
  clipped    -> torch.where: keep values > 6, replace others by 0
  argmax_row -> index of the max of each row
  picked     -> torch.gather: from each row pick column idx = [[0], [3], [1]]
Also: set every even number in a COPY of x (y = x.clone()) to -1.
"""

import torch

x = torch.arange(1, 13).reshape(3, 4)
first_row = x[0]
last_col = x[:, -1]
sub = x[0:2, 1:3]
big = x[x > 6]
rows02 = x[[0, 2]]
plus_row = x + torch.tensor([10, 20, 30, 40])
row_norm = x.float() / x.sum(dim=1, keepdim=True)
clipped = torch.where(x > 6, x, torch.zeros_like(x))
argmax_row = x.argmax(dim=1)
idx = torch.tensor([[0], [3], [1]])
picked = torch.gather(x, dim=1, index=idx)

y = x.clone()
y[y % 2 == 0] = -1

print(x, first_row, last_col, sub, big, rows02, plus_row, row_norm,
      clipped, argmax_row, picked, y, sep="\n")


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    import math
    import torch
    import torch.nn.functional as F
    from torch import nn
    assert first_row.tolist() == [1, 2, 3, 4]
    assert last_col.tolist() == [4, 8, 12]
    assert sub.tolist() == [[2, 3], [6, 7]]
    assert big.tolist() == [7, 8, 9, 10, 11, 12]
    assert rows02.tolist() == [[1, 2, 3, 4], [9, 10, 11, 12]]
    assert plus_row[0].tolist() == [11, 22, 33, 44]
    assert torch.allclose(row_norm.sum(dim=1), torch.ones(3))
    assert clipped.sum() == 57
    assert argmax_row.tolist() == [3, 3, 3]
    assert picked.flatten().tolist() == [1, 8, 10]
    assert (y == -1).sum() == 6 and x[0, 1] == 2
    print("All checks passed!")
