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
"""

import os

os.environ["USE_TF"] = "0"   # never import TensorFlow (avoids Keras 3 errors)

import sys

from transformers import pipeline

sys.stdout.reconfigure(encoding="utf-8")   # generated text may contain unicode (Windows console fix)

clf = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", framework="pt")
sent = clf(["I love this course!", "This is the worst movie ever."])
print(sent)

zsc = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli", framework="pt")
zs = zsc("The match ended 2-1 after extra time", candidate_labels=["sports", "politics", "cooking"])
print(zs["labels"][0], zs["scores"][0])

gen = pipeline("text-generation", model="distilgpt2", framework="pt")
print(gen("Deep learning is", max_new_tokens=20)[0]["generated_text"])


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    import numpy as np
    import torch
    from torch import nn
    assert sent[0]["label"] == "POSITIVE" and sent[1]["label"] == "NEGATIVE"
    assert zs["labels"][0] == "sports"
    print("All checks passed!")
