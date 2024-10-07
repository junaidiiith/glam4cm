from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score, 
    accuracy_score
)

import torch


def compute_classification_metrics(predictions: torch.Tensor, labels: torch.Tensor, multi_class=True):
    scores = torch.argmax(predictions, dim=-1)
    if multi_class:
        f1 = f1_score(labels.numpy(), scores.numpy(), average='weighted')

    else:
        f1 = f1_score(labels.numpy(), scores.numpy())
        
    accuracy = accuracy_score(labels.numpy(), scores.numpy())
    balanced_accuracy = balanced_accuracy_score(labels.numpy(), scores.numpy())

    return {
        'f1-score': f1,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy
    }
