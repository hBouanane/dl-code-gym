"""
Exercise 16 - Transfer learning: freeze a pretrained ResNet, new head  [models]

(downloads ResNet18 weights, ~45 MB)
1. Load torchvision.models.resnet18 with weights=ResNet18_Weights.DEFAULT
2. Freeze ALL parameters (requires_grad = False)
3. Replace model.fc by nn.Linear(model.fc.in_features, 5)  (new layer is trainable by default)
4. n_trainable -> number of trainable params (should be 512*5 + 5)
5. optimizer must only receive trainable params:
     torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
6. Get the preprocessing transforms from the weights: weights.transforms()
7. Forward a dummy batch torch.randn(2, 3, 224, 224) -> shape (2, 5)
Bonus: unfreeze only layer4 (fine-tuning the last block).
Self-check expects these names: out (output of the dummy batch), n_trainable
"""

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

for p in model.parameters():
    p.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, 5)
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
preprocess = weights.transforms()

model.eval()
with torch.inference_mode():
    out = model(torch.randn(2, 3, 224, 224))
print(out.shape, n_trainable, preprocess)

for p in model.layer4.parameters():   # bonus
    p.requires_grad = True


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    import math
    import torch
    import torch.nn.functional as F
    from torch import nn
    assert out.shape == (2, 5)
    assert n_trainable == 512 * 5 + 5
    print("All checks passed!")
