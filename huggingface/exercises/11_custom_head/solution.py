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
"""

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer


class TransformerClassifier(nn.Module):
    def __init__(self, checkpoint, num_labels, dropout=0.1, freeze_backbone=False):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(checkpoint)
        hidden = self.backbone.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, num_labels))
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                          token_type_ids=token_type_ids).last_hidden_state
        logits = self.head(h[:, 0])
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}


checkpoint = "prajjwal1/bert-tiny"
tok = AutoTokenizer.from_pretrained(checkpoint)
model = TransformerClassifier(checkpoint, num_labels=3, freeze_backbone=True)
batch = tok(["first text", "a second, longer text"], padding=True, return_tensors="pt")
out = model(**batch, labels=torch.tensor([0, 2]))
out["loss"].backward()
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(out["logits"].shape, out["loss"].item(), n_trainable)


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    import numpy as np
    import torch
    from torch import nn
    assert out["logits"].shape == (2, 3)
    assert n_trainable == model.backbone.config.hidden_size * 3 + 3
    assert model.head[1].weight.grad is not None
    print("All checks passed!")
