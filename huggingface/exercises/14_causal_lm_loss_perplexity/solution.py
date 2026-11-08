"""
Exercise 14 - Causal LM loss by hand + perplexity  [generation]

model = AutoModelForCausalLM.from_pretrained("distilgpt2")
text = "The capital of France is Paris. It is known for the Eiffel Tower."
  1. hf_loss -> model(**enc, labels=enc["input_ids"]).loss   (HF shifts the labels internally)
  2. manual  -> shift yourself:
        shift_logits = logits[:, :-1, :]   ; shift_labels = input_ids[:, 1:]
        F.cross_entropy(shift_logits.reshape(-1, V), shift_labels.reshape(-1))
     must equal hf_loss
  3. perplexity -> exp(loss)
  4. Compare the perplexity of a normal sentence vs a shuffled-words version (should be higher).
  5. With padding: labels for pad positions must be -100. Build labels = input_ids.clone();
     labels[attention_mask == 0] = -100
Self-check expects these names: hf_loss, manual, ppl_normal, ppl_shuffled, labels
"""

import random

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "distilgpt2"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name).eval()

text = "The capital of France is Paris. It is known for the Eiffel Tower."
enc = tok(text, return_tensors="pt")
with torch.no_grad():
    out = model(**enc, labels=enc["input_ids"])
hf_loss = out.loss
logits = out.logits
V = logits.size(-1)
manual = F.cross_entropy(logits[:, :-1, :].reshape(-1, V), enc["input_ids"][:, 1:].reshape(-1))


def perplexity(t):
    e = tok(t, return_tensors="pt")
    with torch.no_grad():
        return torch.exp(model(**e, labels=e["input_ids"]).loss).item()


words = text.split()
random.seed(0)
random.shuffle(words)
ppl_normal, ppl_shuffled = perplexity(text), perplexity(" ".join(words))
print(hf_loss.item(), manual.item(), ppl_normal, ppl_shuffled)

tok.pad_token = tok.eos_token
batch = tok(["short text", "a much longer piece of text here"], padding=True, return_tensors="pt")
labels = batch["input_ids"].clone()
labels[batch["attention_mask"] == 0] = -100
with torch.no_grad():
    batch_loss = model(**batch, labels=labels).loss
print("padded batch loss", batch_loss.item())


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    import numpy as np
    import torch
    from torch import nn
    assert torch.allclose(hf_loss, manual, atol=1e-4)
    assert ppl_shuffled > ppl_normal
    assert (labels == -100).sum() > 0
    print("All checks passed!")
