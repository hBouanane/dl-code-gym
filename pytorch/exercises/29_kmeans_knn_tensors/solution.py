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
"""

import torch


def pairwise_dist(A, B):
    return ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1).sqrt()


def kmeans(X, k, iters=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    centroids = X[torch.randperm(len(X), generator=g)[:k]].clone()
    for _ in range(iters):
        labels = pairwise_dist(X, centroids).argmin(dim=1)
        for j in range(k):
            if (labels == j).any():
                centroids[j] = X[labels == j].mean(dim=0)
    return centroids, labels


def knn_predict(X_train, y_train, X_test, k=5):
    d = pairwise_dist(X_test, X_train)
    idx = d.topk(k, dim=1, largest=False).indices
    return y_train[idx].mode(dim=1).values


torch.manual_seed(0)
centers = torch.tensor([[0., 0.], [6., 6.], [0., 6.]])
X = torch.cat([torch.randn(100, 2) + c for c in centers])
y = torch.arange(3).repeat_interleave(100)
centroids, labels = kmeans(X, 3)
preds = knn_predict(X, y, X + 0.1 * torch.randn_like(X))
print(centroids, (preds == y).float().mean())


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
