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
    assert emb.shape == (5, 384)
    assert torch.allclose(emb.norm(dim=1), torch.ones(5), atol=1e-5)
    assert sim[0, 3] > sim[0, 2]
    assert search("financial markets crashed", corpus, emb)[0] in {corpus[2], corpus[4]}
    print("All checks passed!")
