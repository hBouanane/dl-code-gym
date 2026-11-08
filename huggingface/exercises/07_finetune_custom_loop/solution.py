"""
Exercise 07 - Fine-tune a classifier with YOUR OWN PyTorch loop  [training]

The most important HF exercise: no Trainer, just the PyTorch loop you know.
  checkpoint = "prajjwal1/bert-tiny" (tiny, trains on CPU in ~1 min)
  1. Data: rotten_tomatoes, 2000 train / 500 validation examples, tokenized (max_length=128),
     columns cleaned like ex 06, DataLoaders with DataCollatorWithPadding (batch 32)
  2. model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
  3. optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5 * 10)  (tiny model -> larger lr ok)
  4. scheduler = get_scheduler("linear", optimizer, num_warmup_steps=0,
                               num_training_steps=epochs * len(train_loader))
  5. Loop 3 epochs:  batch = {k: v.to(device)}; outputs = model(**batch); loss = outputs.loss
                     loss.backward(); optimizer.step(); scheduler.step(); optimizer.zero_grad()
  6. Eval: model.eval(), no_grad, preds = outputs.logits.argmax(-1), accuracy on validation
Expected accuracy for bert-tiny: ~0.65+ (bigger models reach 0.85+).
Self-check expects these names: val_acc
"""

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, get_scheduler

torch.manual_seed(0)
checkpoint = "prajjwal1/bert-tiny"
device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(checkpoint)

raw = load_dataset("rotten_tomatoes")
train_raw = raw["train"].shuffle(seed=0).select(range(2000))
val_raw = raw["validation"].shuffle(seed=0).select(range(500))


def prepare(ds):
    ds = ds.map(lambda b: tok(b["text"], truncation=True, max_length=128), batched=True)
    ds = ds.remove_columns(["text"]).rename_column("label", "labels")
    ds.set_format("torch")
    return ds


collator = DataCollatorWithPadding(tokenizer=tok)
train_loader = DataLoader(prepare(train_raw), batch_size=32, shuffle=True, collate_fn=collator)
val_loader = DataLoader(prepare(val_raw), batch_size=64, collate_fn=collator)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2).to(device)
epochs = 3
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
scheduler = get_scheduler("linear", optimizer, num_warmup_steps=0, num_training_steps=epochs * len(train_loader))


def evaluate():
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            preds = model(**batch).logits.argmax(-1)
            correct += (preds == batch["labels"]).sum().item()
            total += len(preds)
    return correct / total


for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    val_acc = evaluate()
    print(f"epoch {epoch+1} last loss {loss.item():.3f} val acc {val_acc:.3f}")


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    import numpy as np
    import torch
    from torch import nn
    assert val_acc > 0.6
    print("All checks passed!")
