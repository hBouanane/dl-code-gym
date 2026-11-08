"""
Exercise 14 - Causal LM loss by hand + perplexity  [generation]

model = AutoModelForCausalLM.from_pretrained("distilgpt2")
text = "The capital of France is Paris. It is known for the Eiffel Tower."
  1. hf_loss -> model(**enc, labels=enc["input_ids"]).loss   (HF shifts the labels internally)
  2. manual  -> shift yourself:
        shift_logits = logits[:, :-1, :]   ; shift_labels = input_ids[:, 1:]
        F.cross_entropy(shift_logits.reshape(-1, V), shift_labels.reshape(-1))
     must equal hf_loss
  3. perplexity -> exp(loss)
  4. Compare the perplexity of a normal sentence vs a shuffled-words version (should be higher).
  5. With padding: labels for pad positions must be -100. Build labels = input_ids.clone();
     labels[attention_mask == 0] = -100
Self-check expects these names: hf_loss, manual, ppl_normal, ppl_shuffled, labels

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
    assert torch.allclose(hf_loss, manual, atol=1e-4)
    assert ppl_shuffled > ppl_normal
    assert (labels == -100).sum() > 0
    print("All checks passed!")
