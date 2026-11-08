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
"""

import shutil
import sys

from transformers import AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8")   # GPT-2 tokens contain 'Ġ' (Windows console fix)

corpus = ["salam labas 3lik", "ana bikhir hamdullah", "wach nta mzyan", "fin ghadi lyoum",
          "bghit nmchi l casablanca", "had lmakla bnina bzaf", "chokran bzaf a sahbi",
          "wach kayn chi jdid", "ana ghadi l lkhdma", "lyoum jaw zwin bzaf"] * 50


def get_corpus(batch_size=2):
    for i in range(0, len(corpus), batch_size):
        yield corpus[i:i + batch_size]


old = AutoTokenizer.from_pretrained("gpt2")
new = old.train_new_from_iterator(get_corpus(), vocab_size=300)

s = "wach nta mzyan bzaf lyoum"
n_old, n_new = len(old.tokenize(s)), len(new.tokenize(s))
print(old.tokenize(s))
print(new.tokenize(s))

new.save_pretrained("darija_tok")
reloaded = AutoTokenizer.from_pretrained("darija_tok")
shutil.rmtree("darija_tok")


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    import numpy as np
    import torch
    from torch import nn
    assert n_new < n_old
    assert reloaded.tokenize(s) == new.tokenize(s)
    print("All checks passed!")
