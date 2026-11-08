"""
Exercise 03 - Batch tokenization: padding, truncation, tensors  [tokenizers]

tok = AutoTokenizer.from_pretrained("bert-base-uncased")
texts = ["short one", "this sentence is quite a bit longer than the first one", "medium length text"]
  batch    -> tok(texts, padding=True, truncation=True, max_length=16, return_tensors="pt")
              shapes of input_ids and attention_mask?  (3, longest-in-batch)
  fixed    -> same with padding="max_length", max_length=16 -> (3, 16)
  trunc    -> tok(texts[1], truncation=True, max_length=6) -> only 6 ids incl. [CLS]/[SEP]
  lengths  -> number of real tokens per row = batch["attention_mask"].sum(1)
Then decode row 0 WITH special tokens and see the [PAD]s.
Interview: why is attention_mask needed? (so the model ignores pad positions)
"""

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("bert-base-uncased")
texts = ["short one", "this sentence is quite a bit longer than the first one", "medium length text"]

batch = tok(texts, padding=True, truncation=True, max_length=16, return_tensors="pt")
fixed = tok(texts, padding="max_length", truncation=True, max_length=16, return_tensors="pt")
trunc = tok(texts[1], truncation=True, max_length=6)
lengths = batch["attention_mask"].sum(1)

print(batch["input_ids"].shape, fixed["input_ids"].shape, len(trunc["input_ids"]))
print(lengths)
print(tok.decode(batch["input_ids"][0]))


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    import numpy as np
    import torch
    from torch import nn
    assert batch["input_ids"].shape == batch["attention_mask"].shape
    assert batch["input_ids"].shape[1] == lengths.max().item()
    assert fixed["input_ids"].shape == (3, 16)
    assert len(trunc["input_ids"]) == 6
    assert (batch["input_ids"][0][lengths[0]:] == tok.pad_token_id).all()
    print("All checks passed!")
