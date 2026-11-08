"""
Exercise 28 - RNN cell from scratch + nn.LSTM classifier  [sequences]

1. Write my_rnn(x, W_xh, W_hh, b_h) for x: (batch, seq, in): loop over time,
   h_t = tanh(x_t @ W_xh.T + h_{t-1} @ W_hh.T + b_h), h_0 = zeros. Return all h's (batch, seq, hidden)
   Compare with nn.RNN(batch_first=True, bias=True) by copying its weights
   (weight_ih_l0, weight_hh_l0, bias_ih_l0 + bias_hh_l0).
2. Write class LSTMClassifier(nn.Module): Embedding(vocab, 32) -> nn.LSTM(32, 64, batch_first=True)
   -> Linear(64, n_classes) applied to the LAST hidden state h_n[-1].
   Input (batch, seq) of token ids -> output (batch, n_classes).
   Remember: lstm returns output, (h_n, c_n); h_n shape = (num_layers, batch, hidden).
Self-check expects these names: my_rnn, LSTMClassifier(vocab_size, n_classes)
"""

import torch
from torch import nn


def my_rnn(x, W_xh, W_hh, b_h):
    batch, seq, _ = x.shape
    h = torch.zeros(batch, W_hh.size(0))
    outs = []
    for t in range(seq):
        h = torch.tanh(x[:, t] @ W_xh.T + h @ W_hh.T + b_h)
        outs.append(h)
    return torch.stack(outs, dim=1)


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, n_classes, emb_dim=32, hidden=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, n_classes)

    def forward(self, ids):
        out, (h_n, c_n) = self.lstm(self.emb(ids))
        return self.fc(h_n[-1])


torch.manual_seed(0)
rnn = nn.RNN(4, 8, batch_first=True)
x = torch.randn(2, 5, 4)
ref, _ = rnn(x)
mine = my_rnn(x, rnn.weight_ih_l0, rnn.weight_hh_l0, rnn.bias_ih_l0 + rnn.bias_hh_l0)

clf = LSTMClassifier(vocab_size=100, n_classes=3)
logits = clf(torch.randint(0, 100, (4, 12)))
print(mine.shape, logits.shape)


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    import math
    import torch
    import torch.nn.functional as F
    from torch import nn
    torch.manual_seed(0)
    rnn = nn.RNN(4, 8, batch_first=True)
    xx = torch.randn(2, 5, 4)
    ref, _ = rnn(xx)
    assert torch.allclose(my_rnn(xx, rnn.weight_ih_l0, rnn.weight_hh_l0, rnn.bias_ih_l0 + rnn.bias_hh_l0), ref, atol=1e-5)
    assert LSTMClassifier(vocab_size=100, n_classes=3)(torch.randint(0, 100, (4, 12))).shape == (4, 3)
    print("All checks passed!")
