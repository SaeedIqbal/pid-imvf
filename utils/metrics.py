"""
Evaluation Metrics for PID-IMVF.

Mathematical background
-----------------------
This module provides exact, from-scratch implementations of the evaluation 
metrics used to assess the proposed methodology. All computations are performed 
using PyTorch tensor operations to ensure mathematical transparency and 
compatibility with the GPU-accelerated inference pipeline.

1. Accuracy (ACC):
   ACC = (1 / N) * sum_{i=1}^{N} I(y_hat_i == y_i)

2. Balanced Accuracy (BACC):
   BACC = (1 / C) * sum_{c=1}^{C} Recall_c
   where Recall_c = sum(I(y_hat_i == c AND y_i == c)) / sum(I(y_i == c))

3. Macro-F1 Score:
   Precision_c = sum(I(y_hat_i == c AND y_i == c)) / sum(I(y_hat_i == c))
   Recall_c = sum(I(y_hat_i == c AND y_i == c)) / sum(I(y_i == c))
   F1_c = (2 * Precision_c * Recall_c) / (Precision_c + Recall_c + epsilon)
   Macro-F1 = (1 / C) * sum_{c=1}^{C} F1_c
"""

from __future__ import annotations

import torch
from typing import Dict, Any


__all__ = [
    "MetricsCalculator",
]


class MetricsCalculator:
    """
    Computes standard classification metrics (ACC, Macro-F1, BACC) from scratch.
    Designed to evaluate the PID-IMVF model on the original datasets located 
    in /home/phd/datasets/.
    """
    def __init__(self, num_classes: int, epsilon: float = 1e-8):
        """
        Parameters
        ----------
        num_classes : int
            Total number of classes $C$ in the dataset.
        epsilon : float, optional
            Small constant to prevent division by zero in F1 calculations.
        """
        if not isinstance(num_classes, int) or num_classes <= 0:
            raise ValueError(f"num_classes must be a positive integer, got {num_classes}.")
            
        self.num_classes = int(num_classes)
        self.epsilon = float(epsilon)

    def _validate_inputs(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        """
        Validates that the input tensors are 1D, of the same length, 
        and contain integer class indices.
        """
        if not isinstance(y_true, torch.Tensor) or not isinstance(y_pred, torch.Tensor):
            raise TypeError("Inputs must be PyTorch tensors.")
        if y_true.dim() != 1 or y_pred.dim() != 1:
            raise ValueError(f"Inputs must be 1D tensors, got shapes {y_true.shape} and {y_pred.shape}.")
        if y_true.size(0) != y_pred.size(0):
            raise ValueError(f"Input tensors must have the same length, got {y_true.size(0)} and {y_pred.size(0)}.")
        if not y_true.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
            y_true = y_true.long()
        if not y_pred.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
            y_pred = y_pred.long()
            
        return y_true, y_pred

    def compute_accuracy(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """
        Computes Accuracy (ACC).
        
        Mathematical formulation:
        $$ \\text{ACC} = \\frac{1}{N} \\sum_{i=1}^{N} \\mathbb{I}(\\hat{y}_i = y_i) $$
        """
        y_true, y_pred = self._validate_inputs(y_true, y_pred)
        
        correct = torch.sum(y_true == y_pred)
        total = y_true.size(0)
        
        return (correct.float() / total).item()

    def compute_bacc(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """
        Computes Balanced Accuracy (BACC).
        
        Mathematical formulation:
        $$ \\text{BACC} = \\frac{1}{C} \\sum_{c=1}^{C} \\text{Recall}_c $$
        $$ \\text{Recall}_c = \\frac{\\sum_{i=1}^N \\mathbb{I}(\\hat{y}_i = c \\land y_i = c)}{\\sum_{i=1}^N \\mathbb{I}(y_i = c)} $$
        """
        y_true, y_pred = self._validate_inputs(y_true, y_pred)
        
        recalls = []
        for c in range(self.num_classes):
            true_mask = (y_true == c)
            true_count = true_mask.sum().float()
            
            # Due to the strict stratified 10/10/80 splits in MultiViewDataModule, 
            # every class is guaranteed to have at least one sample in the test set.
            if true_count > 0:
                correct_mask = (y_pred == c) & true_mask
                recall = correct_mask.sum().float() / true_count
                recalls.append(recall)
                
        if len(recalls) == 0:
            return 0.0
            
        return torch.stack(recalls).mean().item()

    def compute_macro_f1(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """
        Computes Macro-F1 Score.
        
        Mathematical formulation:
        $$ \\text{Precision}_c = \\frac{\\sum_{i=1}^N \\mathbb{I}(\\hat{y}_i = c \\land y_i = c)}{\\sum_{i=1}^N \\mathbb{I}(\\hat{y}_i = c)} $$
        $$ \\text{Recall}_c = \\frac{\\sum_{i=1}^N \\mathbb{I}(\\hat{y}_i = c \\land y_i = c)}{\\sum_{i=1}^N \\mathbb{I}(y_i = c)} $$
        $$ \\text{F1}_c = \\frac{2 \\cdot \\text{Precision}_c \\cdot \\text{Recall}_c}{\\text{Precision}_c + \\text{Recall}_c + \\epsilon} $$
        $$ \\text{Macro-F1} = \\frac{1}{C} \\sum_{c=1}^{C} \\text{F1}_c $$
        """
        y_true, y_pred = self._validate_inputs(y_true, y_pred)
        
        f1_scores = []
        for c in range(self.num_classes):
            true_mask = (y_true == c)
            pred_mask = (y_pred == c)
            
            true_positives = (true_mask & pred_mask).sum().float()
            total_true = true_mask.sum().float()
            total_pred = pred_mask.sum().float()
            
            # Compute Precision and Recall with epsilon smoothing
            precision = true_positives / (total_pred + self.epsilon)
            recall = true_positives / (total_true + self.epsilon)
            
            # Only include classes that actually exist in the ground truth or predictions
            if (total_true > 0) or (total_pred > 0):
                f1 = (2.0 * precision * recall) / (precision + recall + self.epsilon)
                f1_scores.append(f1)
                
        if len(f1_scores) == 0:
            return 0.0
            
        return torch.stack(f1_scores).mean().item()

    def compute_all_metrics(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
        """
        Computes all evaluation metrics and returns them in a dictionary.
        Reuses the individual computation methods to ensure consistency.
        """
        return {
            'ACC': self.compute_accuracy(y_true, y_pred),
            'Macro-F1': self.compute_macro_f1(y_true, y_pred),
            'BACC': self.compute_bacc(y_true, y_pred)
        }

    @classmethod
    def evaluate_dataloader(cls, model: Any, dataloader: Any, num_classes: int, device: str = 'cpu') -> Dict[str, float]:
        """
        Evaluates the PID-IMVF model on a given DataLoader (e.g., the strict 
        80% inductive test split loaded from /home/phd/datasets/).
        
        Parameters
        ----------
        model : PID_IMVF
            The trained PyTorch Lightning model.
        dataloader : torch.utils.data.DataLoader
            The dataloader yielding batches of (views, targets).
        num_classes : int
            Number of classes in the dataset.
        device : str
            Device to run the evaluation on ('cpu' or 'cuda').
            
        Returns
        -------
        Dict[str, float]
            Dictionary containing ACC, Macro-F1, and BACC.
        """
        calculator = cls(num_classes=num_classes)
        model.eval()
        model.to(device)
        
        all_y_true = []
        all_y_pred = []
        
        with torch.no_grad():
            for views, targets in dataloader:
                # Move original dataset views and targets to the specified device
                views = [v.to(device) for v in views]
                targets = targets.to(device)
                
                # Forward pass through the PID-IMVF architecture
                outputs = model(views, targets)
                logits = outputs['logits']
                
                # Extract predictions (argmax of logits)
                preds = torch.argmax(logits, dim=-1)
                
                # Store predictions and ground truth on CPU for metric computation
                all_y_true.append(targets.cpu())
                all_y_pred.append(preds.cpu())
                
        # Concatenate all batches
        y_true = torch.cat(all_y_true, dim=0)
        y_pred = torch.cat(all_y_pred, dim=0)
        
        # Compute and return all metrics
        return calculator.compute_all_metrics(y_true, y_pred)