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
"""

import torch
import torch.nn.functional as F
from torch import nn

torch.manual_seed(0)
text = ("to be or not to be that is the question whether tis nobler in the mind to suffer "
        "the slings and arrows of outrageous fortune or to take arms against a sea of troubles ") * 20
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join(itos[i] for i in l)
data = torch.tensor(encode(text))

block_size, batch_size, d_model, n_heads, n_layer = 32, 32, 64, 4, 2
device = "cuda" if torch.cuda.is_available() else "cpu"


def get_batch():
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.n_heads = h
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def attn(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q, k, v = (t.view(B, T, self.n_heads, -1).transpose(1, 2) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(out.transpose(1, 2).contiguous().view(B, T, C))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.blocks = nn.Sequential(*[Block(d_model, n_heads) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        logits = self.head(self.ln_f(self.blocks(x)))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -block_size:])
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


model = GPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
for step in range(500):
    xb, yb = get_batch()
    _, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 100 == 0:
        print(f"step {step} loss {loss.item():.3f}")
final_loss = loss.item()

model.eval()
start = torch.tensor([encode("to be")], device=device)
print(decode(model.generate(start, 200)[0].tolist()))


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    import math
    import torch
    import torch.nn.functional as F
    from torch import nn
    assert final_loss < 1.5
    print("All checks passed!")
