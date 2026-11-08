"""
Exercise 14 - A small CNN (conv -> pool -> fc)  [real-data]

1. Write class SmallCNN(nn.Module) for 1x28x28 images, 10 classes:
     block1: Conv2d(1, 32, 3, padding=1) -> ReLU -> MaxPool2d(2)    -> 32x14x14
     block2: Conv2d(32, 64, 3, padding=1) -> ReLU -> MaxPool2d(2)   -> 64x7x7
     classifier: Flatten -> Linear(64*7*7, 128) -> ReLU -> Dropout(0.25) -> Linear(128, 10)
   Shape trick: compute the flatten size by passing a dummy torch.zeros(1, 1, 28, 28).
2. Train it on FashionMNIST (datasets.FashionMNIST, ~30 MB) for 1 epoch, report test accuracy (> 0.85).
Formula to remember: out = (in + 2*padding - kernel) // stride + 1
Bonus: adapt it to CIFAR-10 (3x32x32) — what changes?
Self-check expects these names: SmallCNN, test_acc
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class SmallCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.25), nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


device = "cuda" if torch.cuda.is_available() else "cpu"
print("feature shape:", SmallCNN().features(torch.zeros(1, 1, 28, 28)).shape)

tf = transforms.ToTensor()
train_data = datasets.FashionMNIST("data", train=True, download=True, transform=tf)
test_data = datasets.FashionMNIST("data", train=False, download=True, transform=tf)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=256)

model = SmallCNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model.train()
for X, y in train_loader:
    X, y = X.to(device), y.to(device)
    loss = loss_fn(model(X), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model.eval()
correct = 0
with torch.inference_mode():
    for X, y in test_loader:
        correct += (model(X.to(device)).argmax(1).cpu() == y).sum().item()
test_acc = correct / len(test_data)
print("test acc", test_acc)
# CIFAR-10: in_channels=3, and the flatten size becomes 64*8*8 (32 -> 16 -> 8).


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    import math
    import torch
    import torch.nn.functional as F
    from torch import nn
    assert SmallCNN()(torch.zeros(2, 1, 28, 28)).shape == (2, 10)
    assert test_acc > 0.85
    print("All checks passed!")
