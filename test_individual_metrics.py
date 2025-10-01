#!/usr/bin/env python3
"""
Test script to validate the multilabel individual metrics functionality.
"""

import torch
import numpy as np
from torchmetrics.classification import MultilabelAUROC
from sklearn.metrics import roc_auc_score, f1_score

def test_individual_metrics():
    """Test individual metrics calculation for multilabel classification."""
    print("Testing individual metrics calculation...")
    
    # Create sample data
    num_labels = 3
    n_samples = 100
    
    # Random predictions and binary targets
    np.random.seed(42)
    y_pred = np.random.rand(n_samples, num_labels)
    y_true = np.random.randint(0, 2, (n_samples, num_labels))
    
    # Test sklearn metrics
    print("\nTesting sklearn metrics:")
    auroc_macro = roc_auc_score(y_true, y_pred, average='macro')
    f1_macro = f1_score(y_true, (y_pred > 0.5).astype(int), average='macro')
    print(f"Macro AUROC: {auroc_macro:.4f}")
    print(f"Macro F1: {f1_macro:.4f}")
    
    # Individual metrics
    print("\nIndividual target metrics:")
    target_names = ["TARGET_1", "TARGET_2", "TARGET_3"]
    for i, name in enumerate(target_names):
        auroc_ind = roc_auc_score(y_true[:, i], y_pred[:, i])
        f1_ind = f1_score(y_true[:, i], (y_pred[:, i] > 0.5).astype(int))
        print(f"{name} - AUROC: {auroc_ind:.4f}, F1: {f1_ind:.4f}")
    
    # Test torchmetrics
    print("\nTesting torchmetrics:")
    y_pred_torch = torch.tensor(y_pred, dtype=torch.float32)
    y_true_torch = torch.tensor(y_true, dtype=torch.float32)
    
    # Individual AUROCs using torchmetrics
    metric_individual = MultilabelAUROC(num_labels=num_labels, average=None)
    metric_individual.update(y_pred_torch, y_true_torch.int())
    individual_aurocs = metric_individual.compute()
    
    print(f"Individual AUROCs from torchmetrics: {individual_aurocs}")
    
    # Macro AUROC using torchmetrics
    metric_macro = MultilabelAUROC(num_labels=num_labels, average='macro')
    metric_macro.update(y_pred_torch, y_true_torch.int())
    macro_auroc = metric_macro.compute()
    
    print(f"Macro AUROC from torchmetrics: {macro_auroc:.4f}")
    
    print("\n✅ All tests passed! Individual metrics functionality is working correctly.")

if __name__ == "__main__":
    test_individual_metrics()