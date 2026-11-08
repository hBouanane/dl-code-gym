"""
Exercise 34 - Tiny character-level GPT: train + generate  [attention]

Put everything together (Karpathy "Let's build GPT" style), from a blank file:
1. Data: use the `text` string below (or any .txt). chars = sorted(set(text));
   stoi / itos dicts; encode(s) -> list[int]; decode(l) -> str. data = tensor(encode(text))
2. get_batch(): random offsets ix; x = data[i:i+block_size], y = data[i+1:i+block_size+1]
3. class GPT(nn.Module): token emb + position emb -> n_layer Blocks -> LayerNorm -> Linear(vocab)
   forward(idx, targets=None) returns (logits, loss) with
   loss = F.cross_entropy(logits.view(B*T, V), targets.view(B*T))
4. generate(idx, max_new_tokens): crop idx to last block_size, take logits[:, -1, :],
   softmax, torch.multinomial, append. Under torch.no_grad().
5. Train with AdamW lr=3e-3 for ~500 steps (block_size=32, batch=32, d=64, 4 heads, 2 layers).
   Loss should drop from ~ln(vocab) to well below 1.5. Then generate 200 chars.
text = ("to be or not to be that is the question whether tis nobler in the mind to suffer "
        "the slings and arrows of outrageous fortune or to take arms against a sea of troubles ") * 20
Self-check expects these names: final_loss (last training loss)

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
    assert final_loss < 1.5
    print("All checks passed!")
