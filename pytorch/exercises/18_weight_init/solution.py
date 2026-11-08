"""
Exercise 18 - Weight initialization with model.apply  [models]

1. Write init_weights(m): if m is nn.Linear -> nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
   and nn.init.zeros_(m.bias); if m is nn.Conv2d -> xavier_uniform_ on weight, zeros on bias.
2. model = nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU(), nn.Flatten(), nn.Linear(8*30*30, 10))
   model.apply(init_weights)
3. Verify all biases are 0 and the Linear weight std is close to sqrt(2 / fan_in).
4. Freeze the conv layer only (model[0]) and count trainable params.
Self-check expects these names: model, n_trainable
"""

import math

import torch
from torch import nn


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


torch.manual_seed(0)
model = nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU(), nn.Flatten(), nn.Linear(8 * 30 * 30, 10))
model.apply(init_weights)

fan_in = 8 * 30 * 30
lin_std = model[3].weight.std().item()
print("linear std", lin_std, "expected", math.sqrt(2 / fan_in))

for p in model[0].parameters():
    p.requires_grad = False
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("trainable", n_trainable)


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    import math
    import torch
    import torch.nn.functional as F
    from torch import nn
    assert model[0].bias.abs().sum() == 0 and model[3].bias.abs().sum() == 0
    assert abs(model[3].weight.std().item() - math.sqrt(2 / (8 * 30 * 30))) < 0.002
    assert n_trainable == 8 * 30 * 30 * 10 + 10
    print("All checks passed!")
