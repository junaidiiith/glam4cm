from sklearn.metrics import (
    accuracy_score, 
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

import torch
import numpy as np


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    acc = accuracy_score(p.label_ids, preds)
    balanced_acc = balanced_accuracy_score(p.label_ids, preds)
    return {
        'accuracy': acc,
        'balanced_accuracy': balanced_acc,
    }


def compute_classification_metrics(preds, labels):
    """
    Compute F1-score, balanced accuracy, precision, and recall for multi-class classification.
    
    Args:
        preds (torch.Tensor): Predictions from the model (logits or probabilities). Shape: [num_samples, num_classes]
        labels (torch.Tensor): Ground truth labels. Shape: [num_samples]
    
    Returns:
        dict: Dictionary containing metrics (F1-score, balanced accuracy, precision, recall).
    """
    # Convert predictions to class labels
    preds = torch.argmax(preds, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()

    metrics = {}

    # F1-score (macro and weighted)
    metrics['f1_macro'] = f1_score(labels, preds, average='macro')
    metrics['f1_weighted'] = f1_score(labels, preds, average='weighted')

    # Balanced Accuracy
    metrics['balanced_accuracy'] = balanced_accuracy_score(labels, preds)

    # Precision (macro and weighted)
    metrics['precision_macro'] = precision_score(labels, preds, average='macro', zero_division=0)
    metrics['precision_weighted'] = precision_score(labels, preds, average='weighted', zero_division=0)

    # Recall (macro and weighted)
    metrics['recall_macro'] = recall_score(labels, preds, average='macro', zero_division=0)
    metrics['recall_weighted'] = recall_score(labels, preds, average='weighted', zero_division=0)

    return metrics
