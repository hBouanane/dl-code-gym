"""
Exercise 05 - datasets: load, inspect, map, filter, split  [data]

from datasets import load_dataset
  ds = load_dataset("rotten_tomatoes")    (small sentiment dataset: train/validation/test)
  Print ds, ds["train"].features, ds["train"][0]
  small     -> ds["train"].shuffle(seed=42).select(range(1000))
  with_len  -> small.map(lambda ex: {"n_words": len(ex["text"].split())})
  short     -> with_len.filter(lambda ex: ex["n_words"] < 20)
  tokenized -> small.map(tokenize_fn, batched=True) where tokenize_fn(batch) calls
               tok(batch["text"], truncation=True) with tok = bert-base-uncased tokenizer
  splits    -> small.train_test_split(test_size=0.2, seed=42)
  Finally: tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
  and pandas: small.to_pandas().head()
Why batched=True? (the function gets 1000 examples at once -> much faster with fast tokenizers)
"""

from datasets import load_dataset
from transformers import AutoTokenizer

ds = load_dataset("rotten_tomatoes")
print(ds)
print(ds["train"].features)
print(ds["train"][0])

small = ds["train"].shuffle(seed=42).select(range(1000))
with_len = small.map(lambda ex: {"n_words": len(ex["text"].split())})
short = with_len.filter(lambda ex: ex["n_words"] < 20)

tok = AutoTokenizer.from_pretrained("bert-base-uncased")


def tokenize_fn(batch):
    return tok(batch["text"], truncation=True)


tokenized = small.map(tokenize_fn, batched=True)
splits = small.train_test_split(test_size=0.2, seed=42)
tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
print(small.to_pandas().head())
print(len(short), splits, tokenized[0]["input_ids"][:10])


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    import numpy as np
    import torch
    from torch import nn
    assert len(small) == 1000
    assert all(n < 20 for n in short["n_words"])
    assert "input_ids" in tokenized.column_names
    assert len(splits["train"]) == 800 and len(splits["test"]) == 200
    assert set(ds["train"].unique("label")) == {0, 1}
    print("All checks passed!")
