"""
Part 1: Visualizing the "Right-to-Left" Rule
Imagine you have two shapes, and you want to stack them on top of each other, aligning them strictly to the right side (like lining up numbers by their decimal points).

If one shape has fewer dimensions than the other, PyTorch automatically fills the empty spots on the left with a 1.

Example: Matrix [3, 4] and Vector [4]
Line them up on the right:

Plaintext
   [3, 4]  <-- Matrix (2 dimensions)
      [4]  <-- Vector (1 dimension)
Pad the shorter one with a 1 on the left:

Plaintext
   [3, 4]  <-- Matrix
   [1, 4]  <-- Padded Vector
Compare column by column (from right to left):

Rightmost column: 4 and 4. They match! (They are equal).

Leftmost column: 3 and 1. One of them is a 1, which means it’s allowed to "stretch" to match the other (1 stretches to 3).

Result: Success! The final shape is [3, 4].
"""

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

How to practice:
  1. Write everything in blank.py from memory (imports included).
  2. Stuck? Open solution.py, read ONE line, close it, keep going. Note where you got stuck.
  3. Run `python blank.py`. The self-check at the bottom prints "All checks passed!" when you're right.
  4. Redo it tomorrow and the day after, until it comes out without peeking.
"""

# ============ YOUR CODE BELOW ============





# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    import torch

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
