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
"""

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
y_pred = np.array([1, 0, 0, 1, 0, 1, 1, 0, 1, 0])


def metrics(y_true, y_pred):
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    acc = (y_pred == y_true).mean()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return acc, precision, recall, f1


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc, _, _, f1 = metrics(labels, preds)
    return {"accuracy": float(acc), "f1": float(f1)}


confusion = np.zeros((2, 2), dtype=int)
for t, p in zip(y_true, y_pred):
    confusion[t, p] += 1

acc, precision, recall, f1 = metrics(y_true, y_pred)
print(acc, precision, recall, f1, confusion, sep="\n")


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
