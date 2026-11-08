"""
Exercise 11 - AutoModel backbone + your own nn.Module head  [models]

Build class TransformerClassifier(nn.Module):
  __init__(checkpoint, num_labels, dropout=0.1):
     self.backbone = AutoModel.from_pretrained(checkpoint)
     hidden = self.backbone.config.hidden_size
     self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, num_labels))
  forward(input_ids, attention_mask, token_type_ids=None, labels=None):
     (BERT tokenizers also return token_type_ids, so accept it and pass it on)
     h = self.backbone(input_ids=..., attention_mask=..., token_type_ids=...).last_hidden_state   (B, T, H)
     pooled = h[:, 0]                 # [CLS] token
     logits = self.head(pooled)
     loss = F.cross_entropy(logits, labels) if labels is not None else None
     return {"loss": loss, "logits": logits}
Test with "prajjwal1/bert-tiny", num_labels=3, a batch of 2 texts.
Bonus: option freeze_backbone=True that sets requires_grad=False on the backbone.
Self-check expects these names: model, out, n_trainable

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
    assert out["logits"].shape == (2, 3)
    assert n_trainable == model.backbone.config.hidden_size * 3 + 3
    assert model.head[1].weight.grad is not None
    print("All checks passed!")
