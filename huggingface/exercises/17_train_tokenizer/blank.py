"""
Exercise 17 - Train a new tokenizer on your own corpus  [tokenizers]

Useful for Darija / Arabic where English tokenizers split everything into tiny pieces.
  1. corpus = a list of sentences (below). Write a generator get_corpus(batch_size=2) yielding lists of texts.
  2. old = AutoTokenizer.from_pretrained("gpt2")
  3. new = old.train_new_from_iterator(get_corpus(), vocab_size=300)
  4. Compare len(old.tokenize(s)) vs len(new.tokenize(s)) for a Darija sentence
  5. new.save_pretrained("darija_tok") then reload it and clean up.
corpus = ["salam labas 3lik", "ana bikhir hamdullah", "wach nta mzyan", "fin ghadi lyoum",
          "bghit nmchi l casablanca", "had lmakla bnina bzaf", "chokran bzaf a sahbi",
          "wach kayn chi jdid", "ana ghadi l lkhdma", "lyoum jaw zwin bzaf"] * 50
Self-check expects these names: s, n_old, n_new, new, reloaded

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
    assert n_new < n_old
    assert reloaded.tokenize(s) == new.tokenize(s)
    print("All checks passed!")
