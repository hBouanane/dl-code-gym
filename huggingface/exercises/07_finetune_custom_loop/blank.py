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
    assert val_acc > 0.6
    print("All checks passed!")
