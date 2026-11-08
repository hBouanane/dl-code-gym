"""
Exercise 01 - pipeline(): models working in one line  [basics]

Use transformers.pipeline for 3 tasks and print the results:
  clf  -> pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", framework="pt")
          run it on ["I love this course!", "This is the worst movie ever."]
  zsc  -> pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli", framework="pt")
          classify "The match ended 2-1 after extra time" into ["sports", "politics", "cooking"]
  gen  -> pipeline("text-generation", model="distilgpt2"), max_new_tokens=20
Notice the output format of each (list of dicts). Always pin the model name explicitly
(never rely on the default model). Pass framework="pt" so it never tries TensorFlow.
Self-check expects these names: sent (sentiment results), zs (zero-shot result)

How to practice:
  1. Write everything in blank.py from memory (imports included).
  2. Stuck? Open solution.py, read ONE line, close it, keep going. Note where you got stuck.
  3. Run `python blank.py`. The self-check at the bottom prints "All checks passed!" when you're right.
  4. Redo it tomorrow and the day after, until it comes out without peeking.
"""

# ============ YOUR CODE BELOW ============





# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    assert sent[0]["label"] == "POSITIVE" and sent[1]["label"] == "NEGATIVE"
    assert zs["labels"][0] == "sports"
    print("All checks passed!")
