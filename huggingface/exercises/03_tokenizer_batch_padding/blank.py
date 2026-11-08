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
    assert batch["input_ids"].shape == batch["attention_mask"].shape
    assert batch["input_ids"].shape[1] == lengths.max().item()
    assert fixed["input_ids"].shape == (3, 16)
    assert len(trunc["input_ids"]) == 6
    assert (batch["input_ids"][0][lengths[0]:] == tok.pad_token_id).all()
    print("All checks passed!")
