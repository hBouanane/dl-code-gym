"""
Exercise 13 - Full pipeline: MLP on MNIST (torchvision)  [real-data]

Full pipeline on real data (downloads ~10 MB into ./data).
1. transform = transforms.ToTensor() (optionally Normalize((0.1307,), (0.3081,)))
2. train_data / test_data = datasets.MNIST(root="data", train=True/False, download=True, transform=...)
3. DataLoaders batch 64 (shuffle train only)
4. Model: nn.Flatten() -> Linear(784, 256) -> ReLU -> Linear(256, 10)
5. CrossEntropyLoss + Adam(lr=1e-3), train 2 epochs with your train/eval functions (rewrite them!)
6. Print test accuracy (should be > 0.95)
7. Predict a single image: test_data[0] -> add batch dim -> argmax -> compare to label
Self-check expects these names: test_acc
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=256)

model = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def train_one_epoch():
    model.train()
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        loss = loss_fn(model(X), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate():
    model.eval()
    correct = 0
    with torch.inference_mode():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            correct += (model(X).argmax(1) == y).sum().item()
    return correct / len(test_data)


for epoch in range(2):
    train_one_epoch()
    test_acc = evaluate()
    print(f"epoch {epoch+1} test acc {test_acc:.4f}")

img, label = test_data[0]
with torch.inference_mode():
    pred = model(img.unsqueeze(0).to(device)).argmax(1).item()
print("pred", pred, "label", label)


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    import math
    import torch
    import torch.nn.functional as F
    from torch import nn
    assert test_acc > 0.95
    print("All checks passed!")
