"""
Missing View Simulation and Partial Consensus Computation for PID-IMVF.

Mathematical background
-----------------------
To evaluate robustness under missing-view conditions, we simulate the absence 
of specific views at test time using deterministic dropping strategies. 
The partial consensus shared representation is computed strictly over the 
observed views:

    \hat{S}^{(\text{obs})} = \frac{1}{|\mathcal{V}_{\text{obs}}|} \sum_{v \in \mathcal{V}_{\text{obs}}} \hat{S}_v

where \mathcal{V}_{\text{obs}} \subseteq \{1, \dots, V\} is the set of observed views.
The view-specific factors \hat{U}_v are explicitly excluded from the classification 
input to guarantee the missing-view degradation bound established in Theorem 3.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Dict, Any


__all__ = [
    "MissingViewSimulator",
    "MissingViewEvaluator",
]


class MissingViewSimulator:
    """
    Generates deterministic missing-view patterns and computes the partial 
    consensus shared representation \hat{S}^{(\text{obs})}.
    """
    def __init__(self, num_views: int):
        """
        Parameters
        ----------
        num_views : int
            Total number of views $V$ in the dataset.
        """
        if not isinstance(num_views, int) or num_views <= 0:
            raise ValueError(f"num_views must be a positive integer, got {num_views}.")
        self.num_views = int(num_views)

    def get_deterministic_observed_views(self, strategy: str, target_view_idx: int = 0) -> List[int]:
        """
        Returns a deterministic list of observed view indices based on the 
        specified dropping strategy. This avoids any random synthetic evaluation.
        
        Parameters
        ----------
        strategy : str
            The dropping strategy. Options:
            - 'all': No views dropped.
            - 'drop_first': Drops the first view (index 0).
            - 'drop_last': Drops the last view (index V-1).
            - 'drop_specific': Drops the view at target_view_idx.
        target_view_idx : int, optional
            The index of the specific view to drop (used only if strategy is 'drop_specific').
            
        Returns
        -------
        List[int]
            List of observed view indices.
        """
        all_views = list(range(self.num_views))
        
        if strategy == 'all':
            return all_views
        elif strategy == 'drop_first':
            if self.num_views < 2:
                raise ValueError("Cannot drop the first view when num_views < 2.")
            return all_views[1:]
        elif strategy == 'drop_last':
            if self.num_views < 2:
                raise ValueError("Cannot drop the last view when num_views < 2.")
            return all_views[:-1]
        elif strategy == 'drop_specific':
            if target_view_idx < 0 or target_view_idx >= self.num_views:
                raise ValueError(f"target_view_idx must be in [0, {self.num_views - 1}].")
            return [v for v in all_views if v != target_view_idx]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def compute_partial_consensus(self, shared_reps: List[torch.Tensor], observed_views: List[int]) -> torch.Tensor:
        """
        Computes the partial consensus shared representation \hat{S}^{(\text{obs})}.
        
        Mathematical formulation:
        $$ \hat{S}^{(\text{obs})} = \frac{1}{|\mathcal{V}_{\text{obs}}|} \sum_{v \in \mathcal{V}_{\text{obs}}} \hat{S}_v $$
        
        Parameters
        ----------
        shared_reps : List[torch.Tensor]
            List of V tensors, each of shape (B, d_S), representing the shared 
            factors \hat{S}_v from the Factorized INNs.
        observed_views : List[int]
            List of indices of the observed views.
            
        Returns
        -------
        torch.Tensor, shape (B, d_S)
            The partial consensus representation.
        """
        if not observed_views:
            raise ValueError("observed_views cannot be empty. At least one view must be observed.")
            
        if max(observed_views) >= len(shared_reps) or min(observed_views) < 0:
            raise ValueError("observed_views contains invalid indices.")
            
        # Extract the shared representations for the observed views
        observed_shared_reps = [shared_reps[v] for v in observed_views]
        
        # Stack and compute the mean (Euclidean Frechet mean over observed views)
        stacked = torch.stack(observed_shared_reps, dim=0) # Shape: (|V_obs|, B, d_S)
        partial_consensus = stacked.mean(dim=0) # Shape: (B, d_S)
        
        return partial_consensus


class MissingViewEvaluator:
    """
    Evaluates the PID-IMVF model under deterministic missing-view scenarios 
    using the original datasets from /home/phd/datasets/.
    """
    def __init__(self, model: nn.Module, dataloader: DataLoader, num_views: int, num_classes: int, device: str = 'cpu'):
        """
        Parameters
        ----------
        model : nn.Module
            The trained PID-IMVF PyTorch Lightning model.
        dataloader : DataLoader
            The dataloader yielding batches of (views, targets).
        num_views : int
            Total number of views $V$ in the dataset.
        num_classes : int
            Number of target classes $C$.
        device : str
            Device to run the evaluation on ('cpu' or 'cuda').
        """
        self.model = model
        self.dataloader = dataloader
        self.num_views = num_views
        self.num_classes = num_classes
        self.device = device
        
        self.metrics_calculator = MetricsCalculator(num_classes=num_classes)
        self.simulator = MissingViewSimulator(num_views=num_views)

    def evaluate_missing_view_scenario(self, strategy: str, target_view_idx: int = 0) -> Dict[str, float]:
        """
        Evaluates the model under a specific deterministic missing-view scenario.
        
        Parameters
        ----------
        strategy : str
            The dropping strategy (e.g., 'drop_first', 'drop_specific').
        target_view_idx : int, optional
            The index of the specific view to drop (used if strategy is 'drop_specific').
            
        Returns
        -------
        Dict[str, float]
            Dictionary containing ACC, Macro-F1, and BACC under the missing-view condition.
        """
        # 1. Get deterministic observed views
        observed_views = self.simulator.get_deterministic_observed_views(strategy, target_view_idx)
        
        self.model.eval()
        self.model.to(self.device)
        
        all_y_true = []
        all_y_pred = []
        
        with torch.no_grad():
            for views, targets in self.dataloader:
                views = [v.to(self.device) for v in views]
                targets = targets.to(self.device)
                
                # 2. Encode all views to get shared representations
                shared_reps = []
                for encoder, x_v in zip(self.model.encoders, views):
                    s_v, _, _ = encoder(x_v)
                    shared_reps.append(s_v)
                    
                # 3. Compute partial consensus for the observed views
                s_obs = self.simulator.compute_partial_consensus(shared_reps, observed_views)
                
                # 4. Classify using ONLY the partial consensus (Theorem 3)
                logits = self.model.classifier(s_obs)
                preds = torch.argmax(logits, dim=-1)
                
                all_y_true.append(targets.cpu())
                all_y_pred.append(preds.cpu())
                
        y_true = torch.cat(all_y_true, dim=0)
        y_pred = torch.cat(all_y_pred, dim=0)
        
        # 5. Compute and return metrics
        return self.metrics_calculator.compute_all_metrics(y_true, y_pred)