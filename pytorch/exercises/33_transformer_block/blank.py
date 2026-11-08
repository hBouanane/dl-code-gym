"""
Exercise 33 - Transformer block (pre-LayerNorm, residuals, MLP)  [attention]

class FeedForward(nn.Module): Linear(d, 4d) -> GELU -> Linear(4d, d) -> Dropout
class Block(nn.Module) (GPT-style, pre-norm):
     x = x + self.attn(self.ln1(x), causal=True)
     x = x + self.ffn(self.ln2(x))
Use your MultiHeadAttention from ex 31 (rewrite it here, don't import!) or nn.MultiheadAttention.
Test: Block(64, 4) on (2, 10, 64) keeps the shape; count its parameters.
Interview: pre-norm vs post-norm? (pre-norm = LayerNorm before the sublayer, more stable training)
Self-check expects these names: Block(d_model, n_heads)

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
    blk = Block(64, 4)
    assert blk(torch.randn(2, 10, 64)).shape == (2, 10, 64)
    d = 64
    assert sum(p.numel() for p in blk.parameters()) == 2 * (2 * d) + (3 * d * d + 3 * d) + (d * d + d) + (4 * d * d + 4 * d) + (4 * d * d + d)
    print("All checks passed!")
