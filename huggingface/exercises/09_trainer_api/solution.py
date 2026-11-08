"""
Exercise 09 - Fine-tune with Trainer + TrainingArguments  [training]

Same task as ex 07, now with the Trainer (needs: pip install accelerate).
  Tip: if TensorFlow is installed too, put os.environ["USE_TF"] = "0" BEFORE importing transformers.
  1. Tokenized train (2000) / validation (500) datasets from rotten_tomatoes, checkpoint "prajjwal1/bert-tiny"
  2. args = TrainingArguments(output_dir="out_trainer", eval_strategy="epoch",
            learning_rate=5e-4, per_device_train_batch_size=32, per_device_eval_batch_size=64,
            num_train_epochs=3, weight_decay=0.01, save_strategy="no", report_to="none", logging_steps=20)
  3. trainer = Trainer(model=model, args=args, train_dataset=..., eval_dataset=...,
                       tokenizer=tok, data_collator=DataCollatorWithPadding(tok),
                       compute_metrics=compute_metrics)   (rewrite compute_metrics from ex 08)
  4. trainer.train(); metrics = trainer.evaluate(); print(metrics)
  5. preds = trainer.predict(test_ds) -> preds.predictions (logits), preds.label_ids
Now compare with ex 07: what does Trainer do for you? (loop, device, scheduler, eval, logging, saving)
"""

import os

os.environ["USE_TF"] = "0"   # never import TensorFlow (avoids Keras 3 errors)

import numpy as np
from datasets import load_dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainingArguments)

checkpoint = "prajjwal1/bert-tiny"
tok = AutoTokenizer.from_pretrained(checkpoint)
raw = load_dataset("rotten_tomatoes")
tokenize = lambda b: tok(b["text"], truncation=True, max_length=128)
train_ds = raw["train"].shuffle(seed=0).select(range(2000)).map(tokenize, batched=True)
val_ds = raw["validation"].shuffle(seed=0).select(range(500)).map(tokenize, batched=True)
test_ds = raw["test"].select(range(200)).map(tokenize, batched=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    tp = ((preds == 1) & (labels == 1)).sum()
    precision = tp / max((preds == 1).sum(), 1)
    recall = tp / max((labels == 1).sum(), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"accuracy": float((preds == labels).mean()), "f1": float(f1)}


model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
args = TrainingArguments(
    output_dir="out_trainer",
    eval_strategy="epoch",
    learning_rate=5e-4,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="no",
    report_to="none",
    logging_steps=20,
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tok,
    data_collator=DataCollatorWithPadding(tok),
    compute_metrics=compute_metrics,
)
trainer.train()
metrics = trainer.evaluate()
print(metrics)
preds = trainer.predict(test_ds)
print(preds.predictions.shape, preds.metrics)


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    import numpy as np
    import torch
    from torch import nn
    assert metrics["eval_accuracy"] > 0.6
    assert preds.predictions.shape == (200, 2)
    print("All checks passed!")
