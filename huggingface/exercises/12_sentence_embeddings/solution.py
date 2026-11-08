"""
Exercise 12 - Sentence embeddings: mean pooling + cosine similarity  [models]

name = "sentence-transformers/all-MiniLM-L6-v2" (load with AutoTokenizer / AutoModel)
  1. mean_pool(last_hidden_state, attention_mask): masked average over tokens
       mask = attention_mask.unsqueeze(-1).float()
       (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
  2. embed(texts) -> normalized embeddings (F.normalize(..., p=2, dim=1)), shape (N, 384)
  3. sim = emb @ emb.T  (cosine similarity since normalized)
  4. Semantic search: for a query, return the most similar sentence from the corpus.
corpus = ["A man is eating food.", "A cat plays with a ball.", "The stock market fell today.",
          "Someone is having a meal.", "Shares dropped sharply."]
Self-check expects these names: emb, sim, corpus, search(query, corpus, corpus_emb) -> (sentence, score)
"""

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

name = "sentence-transformers/all-MiniLM-L6-v2"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModel.from_pretrained(name).eval()


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def embed(texts):
    batch = tok(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        h = model(**batch).last_hidden_state
    return F.normalize(mean_pool(h, batch["attention_mask"]), p=2, dim=1)


corpus = ["A man is eating food.", "A cat plays with a ball.", "The stock market fell today.",
          "Someone is having a meal.", "Shares dropped sharply."]
emb = embed(corpus)
sim = emb @ emb.T
print(sim.round(decimals=2))


def search(query, corpus, corpus_emb):
    q = embed([query])
    scores = (q @ corpus_emb.T)[0]
    best = scores.argmax().item()
    return corpus[best], scores[best].item()


print(search("financial markets crashed", corpus, emb))


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    import numpy as np
    import torch
    from torch import nn
    assert emb.shape == (5, 384)
    assert torch.allclose(emb.norm(dim=1), torch.ones(5), atol=1e-5)
    assert sim[0, 3] > sim[0, 2]
    assert search("financial markets crashed", corpus, emb)[0] in {corpus[2], corpus[4]}
    print("All checks passed!")
