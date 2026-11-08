"""
Exercise 08 - Metrics: accuracy, precision, recall, F1, compute_metrics  [training]

1. Implement from scratch for binary labels (numpy or torch):
     accuracy, precision = TP/(TP+FP), recall = TP/(TP+FN), f1 = 2PR/(P+R)
   Compare with sklearn.metrics.
2. Write compute_metrics(eval_pred) — the function Trainer expects:
     logits, labels = eval_pred ; preds = np.argmax(logits, axis=-1)
     return {"accuracy": ..., "f1": ...}
3. confusion -> 2x2 confusion matrix [[TN, FP], [FN, TP]] built with numpy
y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1] ; y_pred = [1, 0, 0, 1, 0, 1, 1, 0, 1, 0]
Self-check expects these names: metrics(y_true, y_pred) -> (acc, precision, recall, f1), compute_metrics, confusion

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
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
    yt = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
    yp = np.array([1, 0, 0, 1, 0, 1, 1, 0, 1, 0])
    a_, p_, r_, f_ = metrics(yt, yp)
    assert np.isclose(a_, accuracy_score(yt, yp)) and np.isclose(p_, precision_score(yt, yp))
    assert np.isclose(r_, recall_score(yt, yp)) and np.isclose(f_, f1_score(yt, yp))
    assert (confusion == confusion_matrix(yt, yp)).all()
    assert np.isclose(compute_metrics((np.eye(2)[yp] * 3.0, yt))["f1"], f_)
    print("All checks passed!")
