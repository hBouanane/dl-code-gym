"""
Exercise 29 - K-means and KNN with pure tensor ops (vectorization)  [from-scratch]

Practice vectorized thinking (no Python loops over points!).
1. pairwise_dist(A, B) -> (n, m) euclidean distances (torch.cdist is allowed to CHECK, not to use;
   hint: ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b, or broadcasting A[:, None] - B[None])
2. kmeans(X, k, iters=20, seed=0) -> (centroids, labels): init with k random points,
   assign = argmin distance, update = mean of assigned points.
3. knn_predict(X_train, y_train, X_test, k=5) -> majority vote of the k nearest labels
   (torch.topk with largest=False, then torch.mode).
Self-check expects these names: pairwise_dist, kmeans, knn_predict

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
    A, B = torch.randn(5, 3), torch.randn(7, 3)
    assert torch.allclose(pairwise_dist(A, B), torch.cdist(A, B), atol=1e-5)
    torch.manual_seed(1)
    cs = torch.tensor([[0., 0.], [8., 8.], [0., 8.]])
    XX = torch.cat([torch.randn(50, 2) + c for c in cs])
    yy = torch.arange(3).repeat_interleave(50)
    cents, labs = kmeans(XX, 3)
    assert torch.cdist(cs, cents).min(dim=1).values.max() < 1.0
    assert len(labs) == 150
    assert (knn_predict(XX, yy, XX, k=5) == yy).float().mean() > 0.95
    print("All checks passed!")
