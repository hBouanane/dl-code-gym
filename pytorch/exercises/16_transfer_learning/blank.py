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
    assert out.shape == (2, 5)
    assert n_trainable == 512 * 5 + 5
    print("All checks passed!")
