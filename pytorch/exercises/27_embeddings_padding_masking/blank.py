"""
Exercise 27 - Embeddings, padding variable-length sequences, masks  [sequences]

1. seqs = [tensor([5, 2, 8]), tensor([3, 1]), tensor([7, 4, 9, 6])]  (token ids, PAD id = 0)
   padded -> torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0)  -> (3, 4)
   lengths -> tensor([3, 2, 4])
2. mask -> boolean (3, 4), True where token is real. Build it from lengths with
   torch.arange(max_len)[None, :] < lengths[:, None]
3. emb = nn.Embedding(num_embeddings=10, embedding_dim=6, padding_idx=0); x = emb(padded) -> (3, 4, 6)
4. masked mean pooling: sum(x * mask[..., None], dim=1) / lengths[:, None]  -> (3, 6)
   (this is EXACTLY how sentence embeddings are built from BERT outputs later)
5. Build a causal mask for length 4: torch.tril(torch.ones(4, 4)).bool()
Self-check expects these names: padded, lengths, mask, emb, x, pooled, causal

How to practice:
  1. Write everything in blank.py from memory (imports included).
  2. Stuck? Open solution.py, read ONE line, close it, keep going. Note where you got stuck.
  3. Run `python blank.py`. The self-check at the bottom prints "All checks passed!" when you're right.
  4. Redo it tomorrow and the day after, until it comes out without peeking.
"""

# ============ YOUR CODE BELOW ============





# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    import math
    import torch
    import torch.nn.functional as F
    from torch import nn
    assert padded.tolist() == [[5, 2, 8, 0], [3, 1, 0, 0], [7, 4, 9, 6]]
    assert torch.equal(mask, padded != 0)
    assert x.shape == (3, 4, 6) and torch.all(emb.weight[0] == 0)
    assert torch.allclose(pooled[1], x[1, :2].mean(0))
    assert causal.sum() == 10 and not causal[0, 1]
    print("All checks passed!")
