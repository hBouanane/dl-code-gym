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
    assert logits.shape == (2, 2)
    assert labels == ["POSITIVE", "NEGATIVE"]
    assert hidden.shape[0] == 2 and hidden.shape[2] == 768
    print("All checks passed!")
