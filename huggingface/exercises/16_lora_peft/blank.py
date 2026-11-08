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

How to practice:
  1. Write everything in blank.py from memory (imports included).
  2. Stuck? Open solution.py, read ONE line, close it, keep going. Note where you got stuck.
  3. Run `python blank.py`. The self-check at the bottom prints "All checks passed!" when you're right.
  4. Redo it tomorrow and the day after, until it comes out without peeking.
"""

# ============ YOUR CODE BELOW ============





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
