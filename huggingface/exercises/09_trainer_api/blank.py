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
    assert metrics["eval_accuracy"] > 0.6
    assert preds.predictions.shape == (200, 2)
    print("All checks passed!")
