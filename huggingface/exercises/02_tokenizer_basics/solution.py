"""
Exercise 02 - Tokenizer: tokens, ids, special tokens, decode  [tokenizers]

tok = AutoTokenizer.from_pretrained("bert-base-uncased")
  tokens -> tok.tokenize("Tokenization isn't hard!")      (see the ## subwords)
  ids    -> tok.convert_tokens_to_ids(tokens)
  enc    -> tok("Tokenization isn't hard!")  -> dict with input_ids, token_type_ids, attention_mask
            note: enc adds [CLS] at the start and [SEP] at the end
  text   -> tok.decode(enc["input_ids"], skip_special_tokens=True)
  Print tok.cls_token, tok.sep_token, tok.pad_token and their ids, and tok.vocab_size.
  pair   -> tok("How are you?", "I am fine.") ; look at token_type_ids (0 for sentence A, 1 for B)
"""

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("bert-base-uncased")
sentence = "Tokenization isn't hard!"
tokens = tok.tokenize(sentence)
ids = tok.convert_tokens_to_ids(tokens)
enc = tok(sentence)
text = tok.decode(enc["input_ids"], skip_special_tokens=True)

print(tokens)
print(ids)
print(enc)
print(text)
print(tok.cls_token, tok.cls_token_id, tok.sep_token, tok.sep_token_id, tok.pad_token, tok.pad_token_id)
print("vocab size", tok.vocab_size)

pair = tok("How are you?", "I am fine.")
print(tok.convert_ids_to_tokens(pair["input_ids"]))
print(pair["token_type_ids"])


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    import numpy as np
    import torch
    from torch import nn
    assert enc["input_ids"][0] == tok.cls_token_id and enc["input_ids"][-1] == tok.sep_token_id
    assert enc["input_ids"][1:-1] == ids
    assert text == "tokenization isn't hard!"
    assert 1 in pair["token_type_ids"] and pair["token_type_ids"][0] == 0
    print("All checks passed!")
