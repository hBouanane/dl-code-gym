"""
Exercise 18 - Weight initialization with model.apply  [models]

1. Write init_weights(m): if m is nn.Linear -> nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
   and nn.init.zeros_(m.bias); if m is nn.Conv2d -> xavier_uniform_ on weight, zeros on bias.
2. model = nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU(), nn.Flatten(), nn.Linear(8*30*30, 10))
   model.apply(init_weights)
3. Verify all biases are 0 and the Linear weight std is close to sqrt(2 / fan_in).
4. Freeze the conv layer only (model[0]) and count trainable params.
Self-check expects these names: model, n_trainable

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
    assert model[0].bias.abs().sum() == 0 and model[3].bias.abs().sum() == 0
    assert abs(model[3].weight.std().item() - math.sqrt(2 / (8 * 30 * 30))) < 0.002
    assert n_trainable == 8 * 30 * 30 * 10 + 10
    print("All checks passed!")
