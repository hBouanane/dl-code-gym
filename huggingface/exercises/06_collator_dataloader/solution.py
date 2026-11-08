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
"""

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

tok = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
ds = load_dataset("rotten_tomatoes", split="train").shuffle(seed=0).select(range(512))
ds = ds.map(lambda b: tok(b["text"], truncation=True), batched=True)
ds = ds.remove_columns(["text"]).rename_column("label", "labels")
ds.set_format("torch")

collator = DataCollatorWithPadding(tokenizer=tok)
loader = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=collator)

seq_lens = []
for i, batch in enumerate(loader):
    print({k: tuple(v.shape) for k, v in batch.items()})
    seq_lens.append(batch["input_ids"].shape[1])
    if i == 2:
        break


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    import numpy as np
    import torch
    from torch import nn
    assert set(batch.keys()) >= {"input_ids", "attention_mask", "labels"}
    assert batch["input_ids"].shape[0] == 16 and batch["labels"].shape == (16,)
    assert len(set(seq_lens)) > 1, "dynamic padding should give different lengths"
    print("All checks passed!")
