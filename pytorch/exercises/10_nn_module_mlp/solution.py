"""
Exercise 10 - Writing an nn.Module (MLP)  [models]

1. Write class MLP(nn.Module) with __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
     Linear(in, hidden) -> ReLU -> Dropout -> Linear(hidden, hidden) -> ReLU -> Linear(hidden, out)
   and forward(self, x).
2. model = MLP(784, 128, 10). Pass a dummy batch torch.randn(32, 784) -> out shape (32, 10)
3. n_params -> total number of parameters (sum of p.numel())
   n_trainable -> only those with requires_grad
4. Write the same network with nn.Sequential -> seq_model
5. Print the model and loop over model.named_parameters() printing name + shape.
"""

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


model = MLP(784, 128, 10)
out = model(torch.randn(32, 784))
n_params = sum(p.numel() for p in model.parameters())
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

seq_model = nn.Sequential(
    nn.Linear(784, 128), nn.ReLU(), nn.Dropout(0.1),
    nn.Linear(128, 128), nn.ReLU(),
    nn.Linear(128, 10),
)

print(model)
for name, p in model.named_parameters():
    print(name, tuple(p.shape))
print("params:", n_params, "trainable:", n_trainable)


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    import math
    import torch
    import torch.nn.functional as F
    from torch import nn
    assert out.shape == (32, 10)
    assert n_params == 784 * 128 + 128 + 128 * 128 + 128 + 128 * 10 + 10
    assert n_trainable == n_params
    assert seq_model(torch.randn(4, 784)).shape == (4, 10)
    print("All checks passed!")
