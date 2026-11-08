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
    torch.manual_seed(0)
    rnn = nn.RNN(4, 8, batch_first=True)
    xx = torch.randn(2, 5, 4)
    ref, _ = rnn(xx)
    assert torch.allclose(my_rnn(xx, rnn.weight_ih_l0, rnn.weight_hh_l0, rnn.bias_ih_l0 + rnn.bias_hh_l0), ref, atol=1e-5)
    assert LSTMClassifier(vocab_size=100, n_classes=3)(torch.randint(0, 100, (4, 12))).shape == (4, 3)
    print("All checks passed!")
