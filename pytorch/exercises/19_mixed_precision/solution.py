"""
Exercise 19 - Mixed precision (autocast + GradScaler)  [training]

Write a training step that uses automatic mixed precision:
  device = "cuda" if available else "cpu"
  amp_dtype = torch.float16 on cuda, torch.bfloat16 on cpu
  scaler = torch.amp.GradScaler(device, enabled=(device == "cuda"))
  step:
    with torch.autocast(device_type=device, dtype=amp_dtype):
        logits = model(X); loss = loss_fn(logits, y)
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
Train a Linear(20, 2) on random data with labels y = (X[:, 0] > 0) for 100 steps,
accuracy should get > 0.9. Explain to yourself: why does fp16 need a GradScaler but bf16 doesn't?
Self-check expects these names: acc
"""

import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
amp_dtype = torch.float16 if device == "cuda" else torch.bfloat16

torch.manual_seed(0)
X = torch.randn(512, 20, device=device)
y = (X[:, 0] > 0).long()
model = nn.Linear(20, 2).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
scaler = torch.amp.GradScaler(device, enabled=(device == "cuda"))

for step in range(100):
    with torch.autocast(device_type=device, dtype=amp_dtype):
        logits = model(X)
        loss = loss_fn(logits, y)
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

with torch.inference_mode():
    acc = (model(X).argmax(1) == y).float().mean().item()
print("loss", loss.item(), "acc", acc)
# fp16 has a small exponent range: tiny gradients underflow to 0, so we scale the loss up.
# bf16 has the same exponent range as fp32, so no scaling is needed.


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    import math
    import torch
    import torch.nn.functional as F
    from torch import nn
    assert acc > 0.9
    print("All checks passed!")
