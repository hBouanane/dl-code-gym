"""
Exercise 04 - Tokenizer + model by hand: logits -> probabilities  [models]

What pipeline("sentiment-analysis") does, by hand:
  name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
  tok = AutoTokenizer.from_pretrained(name); model = AutoModelForSequenceClassification.from_pretrained(name)
  inputs = tok(texts, padding=True, truncation=True, return_tensors="pt")
  model.eval(); with torch.no_grad(): outputs = model(**inputs)
  logits -> outputs.logits (shape (batch, 2))
  probs  -> softmax(logits, dim=-1)
  labels -> [model.config.id2label[i] for i in probs.argmax(-1).tolist()]
Also load the bare backbone AutoModel.from_pretrained(name) and look at
last_hidden_state shape (batch, seq, 768). What's the difference between the two classes?
Self-check expects these names: logits, labels, hidden (last_hidden_state)
"""

import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name)

texts = ["I love this course!", "This is the worst movie ever."]
inputs = tok(texts, padding=True, truncation=True, return_tensors="pt")
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
logits = outputs.logits
probs = torch.softmax(logits, dim=-1)
labels = [model.config.id2label[i] for i in probs.argmax(-1).tolist()]
print(logits, probs, labels, sep="\n")

backbone = AutoModel.from_pretrained(name)
with torch.no_grad():
    hidden = backbone(**inputs).last_hidden_state
print(hidden.shape)   # AutoModel = no head (raw hidden states); ...ForSequenceClassification adds a classifier head


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    import numpy as np
    import torch
    from torch import nn
    assert logits.shape == (2, 2)
    assert labels == ["POSITIVE", "NEGATIVE"]
    assert hidden.shape[0] == 2 and hidden.shape[2] == 768
    print("All checks passed!")
