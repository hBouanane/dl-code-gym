"""
Exercise 13 - Token classification: aligning word labels to subword tokens  [tokenizers]

Classic interview / NER fine-tuning step. Words have labels but the tokenizer splits words
into several subwords.
  tok = AutoTokenizer.from_pretrained("bert-base-cased")   (must be a FAST tokenizer)
  words  = ["Haitam", "lives", "in", "Casablanca", "Morocco"]
  labels = [1, 0, 0, 2, 2]          (0 = O, 1 = B-PER, 2 = B-LOC)
  enc = tok(words, is_split_into_words=True)
  word_ids = enc.word_ids()   -> None for special tokens, else the index of the word
Write align_labels(word_ids, labels) -> list where:
   special tokens (None)          -> -100   (ignored by CrossEntropyLoss!)
   first subword of a word        -> that word's label
   other subwords of the same word-> -100
Then write tokenize_and_align(batch) usable with dataset.map(batched=True)
(use enc.word_ids(batch_index=i)).
Self-check expects these names: align_labels, tokenize_and_align

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
    assert align_labels([None, 0, 0, 1, None], [5, 7]) == [-100, 5, -100, 7, -100]
    out_ = tokenize_and_align({"tokens": [["Haitam", "lives", "in", "Casablanca"], ["Rabat"]], "ner_tags": [[1, 0, 0, 2], [2]]})
    assert [l for l in out_["labels"][0] if l != -100] == [1, 0, 0, 2]
    assert out_["labels"][1][1] == 2
    print("All checks passed!")
