from sklearn.metrics import (
    accuracy_score, 
    balanced_accuracy_score
)

import numpy as np


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    acc = accuracy_score(p.label_ids, preds)
    balanced_acc = balanced_accuracy_score(p.label_ids, preds)
    return {
        'accuracy': acc,
        'balanced_accuracy': balanced_acc,
    }
