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
    assert torch.equal(manual, greedy)
    assert sampled.shape[1] <= inputs["input_ids"].shape[1] + 20
    assert sample_next(torch.tensor([[0., 0., 10., 0.]]), 1.0, 1).item() == 2
    print("All checks passed!")
