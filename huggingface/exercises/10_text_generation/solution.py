"""
Exercise 10 - Text generation: generate() and a manual greedy loop  [generation]

name = "distilgpt2"; tok = AutoTokenizer...; model = AutoModelForCausalLM...
  1. greedy  -> model.generate(**inputs, max_new_tokens=20, do_sample=False)
  2. sampled -> do_sample=True, temperature=0.8, top_k=50, top_p=0.95 (torch.manual_seed first)
     GPT-2 has no pad token: pass pad_token_id=tok.eos_token_id
  3. MANUAL greedy decoding (interview!): loop 20 times:
        logits = model(input_ids).logits[:, -1, :] ; next_id = logits.argmax(-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_id], dim=-1)
     -> manual result must equal the generate() greedy result
  4. Write sample_next(logits, temperature, top_k) yourself: divide by temperature,
     keep top_k logits (others = -inf), softmax, torch.multinomial.
Self-check expects these names: inputs, greedy, sampled, manual, sample_next
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "distilgpt2"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name)
model.eval()

inputs = tok("The future of artificial intelligence", return_tensors="pt")
with torch.no_grad():
    greedy = model.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=tok.eos_token_id)
    torch.manual_seed(0)
    sampled = model.generate(**inputs, max_new_tokens=20, do_sample=True, temperature=0.8,
                             top_k=50, top_p=0.95, pad_token_id=tok.eos_token_id)
print("greedy :", tok.decode(greedy[0], skip_special_tokens=True))
print("sampled:", tok.decode(sampled[0], skip_special_tokens=True))

input_ids = inputs["input_ids"]
with torch.no_grad():
    for _ in range(20):
        logits = model(input_ids).logits[:, -1, :]
        next_id = logits.argmax(-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_id], dim=-1)
manual = input_ids


def sample_next(logits, temperature=1.0, top_k=50):
    logits = logits / temperature
    v, _ = torch.topk(logits, top_k)
    logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


print(tok.decode(sample_next(logits, 0.8, 10)[0]))


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    import numpy as np
    import torch
    from torch import nn
    assert torch.equal(manual, greedy)
    assert sampled.shape[1] <= inputs["input_ids"].shape[1] + 20
    assert sample_next(torch.tensor([[0., 0., 10., 0.]]), 1.0, 1).item() == 2
    print("All checks passed!")
