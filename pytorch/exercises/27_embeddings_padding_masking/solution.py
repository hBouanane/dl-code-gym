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
"""

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

seqs = [torch.tensor([5, 2, 8]), torch.tensor([3, 1]), torch.tensor([7, 4, 9, 6])]
padded = pad_sequence(seqs, batch_first=True, padding_value=0)
lengths = torch.tensor([len(s) for s in seqs])
max_len = padded.size(1)
mask = torch.arange(max_len)[None, :] < lengths[:, None]

torch.manual_seed(0)
emb = nn.Embedding(10, 6, padding_idx=0)
x = emb(padded)
pooled = (x * mask[..., None]).sum(dim=1) / lengths[:, None]
causal = torch.tril(torch.ones(4, 4)).bool()
print(padded, mask, pooled.shape, causal, sep="\n")


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
