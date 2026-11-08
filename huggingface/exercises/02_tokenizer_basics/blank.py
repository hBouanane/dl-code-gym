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

How to practice:
  1. Write everything in blank.py from memory (imports included).
  2. Stuck? Open solution.py, read ONE line, close it, keep going. Note where you got stuck.
  3. Run `python blank.py`. The self-check at the bottom prints "All checks passed!" when you're right.
  4. Redo it tomorrow and the day after, until it comes out without peeking.
"""

# ============ YOUR CODE BELOW ============





# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    assert (
        enc["input_ids"][0] == tok.cls_token_id
        and enc["input_ids"][-1] == tok.sep_token_id
    )
    assert enc["input_ids"][1:-1] == ids
    assert text == "tokenization isn't hard!"
    assert 1 in pair["token_type_ids"] and pair["token_type_ids"][0] == 0
    print("All checks passed!")
