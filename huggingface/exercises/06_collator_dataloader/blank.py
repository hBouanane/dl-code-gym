"""
Exercise 06 - Dynamic padding: DataCollatorWithPadding + DataLoader  [data]

Connect HF datasets to the PyTorch DataLoader you already know:
  1. Load rotten_tomatoes, take 512 training examples, tokenize with truncation (NO padding here)
  2. Remove the "text" column (remove_columns) and rename "label" -> "labels"
     (HF models compute the loss automatically when they receive `labels`)
  3. set_format("torch")
  4. collator = DataCollatorWithPadding(tokenizer=tok)
  5. loader = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=collator)
  6. Look at 3 batches: their seq lengths differ (dynamic padding = pad to longest IN THE BATCH)
Tokenizer: "prajjwal1/bert-tiny"
Self-check expects these names: batch (last batch seen), seq_lens (list of seq lengths of the batches)

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
    assert set(batch.keys()) >= {"input_ids", "attention_mask", "labels"}
    assert batch["input_ids"].shape[0] == 16 and batch["labels"].shape == (16,)
    assert len(set(seq_lens)) > 1, "dynamic padding should give different lengths"
    print("All checks passed!")
