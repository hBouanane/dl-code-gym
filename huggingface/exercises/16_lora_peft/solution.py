"""
Exercise 16 - Parameter-efficient fine-tuning with LoRA (peft)  [training]

(needs: pip install peft)
  1. base = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny", num_labels=2)
  2. config = LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16, lora_dropout=0.1,
                         target_modules=["query", "value"])
  3. model = get_peft_model(base, config); model.print_trainable_parameters()
     -> only a few % of params are trainable
  4. Run one training step on a dummy batch to check gradients flow.
  5. BONUS (understand it!): write class LoRALinear(nn.Module) wrapping a frozen nn.Linear:
        out = base(x) + (x @ A.T @ B.T) * (alpha / r),  A: (r, in) random small, B: (out, r) zeros
     At init its output equals the base layer (because B = 0).
Self-check expects these names: LoRALinear
"""

import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from peft import LoraConfig, TaskType, get_peft_model
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False
    print("peft not installed: pip install peft  (skipping the peft part)")

ckpt = "prajjwal1/bert-tiny"
if HAS_PEFT:
    tok = AutoTokenizer.from_pretrained(ckpt)
    base = AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=2)
    config = LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16, lora_dropout=0.1,
                        target_modules=["query", "value"])
    model = get_peft_model(base, config)
    model.print_trainable_parameters()
    batch = tok(["good movie", "bad movie"], padding=True, return_tensors="pt")
    loss = model(**batch, labels=torch.tensor([1, 0])).loss
    loss.backward()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    opt.step()


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scaling = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scaling


lin = nn.Linear(64, 32)
lora = LoRALinear(lin, r=4)
x = torch.randn(5, 64)
n_trainable = sum(p.numel() for p in lora.parameters() if p.requires_grad)
print(torch.allclose(lora(x), lin(x)), n_trainable)


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    import numpy as np
    import torch
    from torch import nn
    lin_ = nn.Linear(64, 32)
    lora_ = LoRALinear(lin_, r=4)
    xx = torch.randn(5, 64)
    assert torch.allclose(lora_(xx), lin_(xx))
    assert sum(p.numel() for p in lora_.parameters() if p.requires_grad) == 4 * 64 + 32 * 4
    print("All checks passed!")
