"""
Exercise 15 - save_pretrained / from_pretrained / push_to_hub  [deployment]

  1. Load "prajjwal1/bert-tiny" as AutoModelForSequenceClassification(num_labels=2) and its tokenizer,
     with id2label={0: "NEG", 1: "POS"} and label2id={"NEG": 0, "POS": 1}
  2. model.save_pretrained("my_model"); tok.save_pretrained("my_model")
     -> list the files created (config.json, model.safetensors, tokenizer files...)
  3. Reload both from "my_model" and check the outputs are identical and that
     reloaded.config.id2label[1] == "POS"
  4. Pushing to the Hub (do it for real once you have an account; needs `huggingface-cli login`):
       model.push_to_hub("your-username/my-first-model")
       tok.push_to_hub("your-username/my-first-model")
     (keep these lines commented out in the file)
  (The my_model/ folder is git-ignored. On Windows you cannot delete it while a loaded model still has it open.)
Self-check expects these names: files, same, reloaded
"""

import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ckpt = "prajjwal1/bert-tiny"
tok = AutoTokenizer.from_pretrained(ckpt)
model = AutoModelForSequenceClassification.from_pretrained(
    ckpt, num_labels=2, id2label={0: "NEG", 1: "POS"}, label2id={"NEG": 0, "POS": 1}
).eval()

for p in model.parameters():          # bert-tiny has non-contiguous weights,
    p.data = p.data.contiguous()      # and safetensors refuses to save those
model.save_pretrained("my_model")
tok.save_pretrained("my_model")
files = sorted(os.listdir("my_model"))
print(files)

reloaded = AutoModelForSequenceClassification.from_pretrained("my_model").eval()
tok2 = AutoTokenizer.from_pretrained("my_model")
batch = tok2(["hello world"], return_tensors="pt")
with torch.no_grad():
    same = torch.allclose(model(**batch).logits, reloaded(**batch).logits)
print("same outputs:", same, reloaded.config.id2label)

# model.push_to_hub("your-username/my-first-model")
# tok.push_to_hub("your-username/my-first-model")


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    import numpy as np
    import torch
    from torch import nn
    assert "config.json" in files
    assert same and reloaded.config.id2label[1] == "POS"
    print("All checks passed!")
