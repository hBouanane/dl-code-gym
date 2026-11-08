"""
Exercise 15 - Save / load models and resume training  [training]

1. model = nn.Linear(4, 2); optimizer = Adam
2. Save only the weights: torch.save(model.state_dict(), "model.pt")
3. Load them into a NEW model: model2 = nn.Linear(4, 2); model2.load_state_dict(torch.load(...))
   -> check both give the same output
4. Full checkpoint for resuming: save a dict with keys
     "epoch", "model_state", "optimizer_state", "loss"  -> "checkpoint.pt"
5. Write resume(path, model, optimizer) that loads the checkpoint into model and
   optimizer and returns the next epoch to train (epoch + 1).
   Use map_location="cpu" when loading (why?).
6. Clean up the files at the end (os.remove).
Self-check expects these names: same (bool), start_epoch
"""

import os

import torch
from torch import nn

torch.manual_seed(0)
model = nn.Linear(4, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

torch.save(model.state_dict(), "model.pt")
model2 = nn.Linear(4, 2)
model2.load_state_dict(torch.load("model.pt", map_location="cpu"))
x = torch.randn(3, 4)
same = torch.allclose(model(x), model2(x))

torch.save({
    "epoch": 5,
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "loss": 0.123,
}, "checkpoint.pt")


def resume(path, model, optimizer):
    ckpt = torch.load(path, map_location="cpu")   # works even if saved on GPU
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt["epoch"] + 1


model3 = nn.Linear(4, 2)
opt3 = torch.optim.Adam(model3.parameters(), lr=1e-3)
start_epoch = resume("checkpoint.pt", model3, opt3)
print("same output:", same, "resume at epoch", start_epoch)

os.remove("model.pt")
os.remove("checkpoint.pt")


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    import math
    import torch
    import torch.nn.functional as F
    from torch import nn
    assert same
    assert start_epoch == 6
    print("All checks passed!")
