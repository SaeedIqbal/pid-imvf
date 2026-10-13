"""
Partial Information Decomposition (PID) Mathematical Logic for PID-IMVF.

Mathematical background
-----------------------
This module implements the core mathematical operations for the Williams-Beer 
PID framework, specifically tailored for the Barber-Agakov classifier-based 
Mutual Information (MI) bounds.

1. Entropy of Target:
   H(Y) = - \sum_{c=1}^C p(c) \log p(c)

2. Barber-Agakov MI Bound:
   I(Z; Y) \ge H(Y) - \text{CE}(q_\phi(Y|Z))

3. Williams-Beer Redundancy:
   \text{Red}(\mathbf{Z}; Y) = \min_{v \in \{1, \dots, V\}} I(Z_v; Y)

4. Uniqueness:
   \text{Unq}(Z_v; Y | Z_{\setminus v}) = I(Z_v; Y) - \text{Red}

5. Synergy (V-source):
   \text{Syn}(\mathbf{Z}; Y) = I(\mathbf{Z}; Y) - \text{Red} - \sum_{v} \text{Unq}(Z_v; Y | Z_{\setminus v})

6. Pairwise Synergy (for V > 2):
   \text{Syn}(Z_i, Z_j; Y) = I(Z_i, Z_j; Y) - \text{Red}(Z_i, Z_j; Y) - \text{Unq}(Z_i; Y|Z_j) - \text{Unq}(Z_j; Y|Z_i)
   The global unique and synergistic information are aggregated by averaging across all \binom{V}{2} pairs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import List, Dict
from itertools import combinations


__all__ = [
    "PIDMathCalculator",
]


class PIDMathCalculator:
    """
    Computes the Partial Information Decomposition (PID) atoms using the 
    Williams-Beer framework and Barber-Agakov MI bounds.
    """
    def __init__(self, num_classes: int, epsilon: float = 1e-8):
        """
        Parameters
        ----------
        num_classes : int
            Total number of classes $C$ in the dataset.
        epsilon : float, optional
            Small constant to prevent log(0) and division by zero.
        """
        if not isinstance(num_classes, int) or num_classes <= 0:
            raise ValueError(f"num_classes must be a positive integer, got {num_classes}.")
            
        self.num_classes = int(num_classes)
        self.epsilon = float(epsilon)

    def compute_entropy(self, y: torch.Tensor) -> torch.Tensor:
        """
        Computes the empirical entropy of the target labels H(Y).
        
        Mathematical formulation:
        $$ H(Y) = - \\sum_{c=1}^C p(c) \\log p(c) $$
        """
        with torch.no_grad():
            counts = torch.bincount(y, minlength=self.num_classes).float()
            probs = counts / y.size(0)
            probs = probs.clamp(min=self.epsilon)
            entropy = -torch.sum(probs * torch.log(probs))
        return entropy

    def compute_mi_bound(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes the Barber-Agakov lower bound on Mutual Information.
        
        Mathematical formulation:
        $$ I(Z; Y) \\ge H(Y) - \\text{CE}(q_\\phi(Y|Z)) $$
        """
        h_y = self.compute_entropy(y)
        ce = F.cross_entropy(logits, y)
        return h_y - ce

    def compute_williams_beer_redundancy(self, marginal_mis: torch.Tensor) -> torch.Tensor:
        """
        Computes the Williams-Beer redundancy.
        
        Mathematical formulation:
        $$ \\text{Red}(\\mathbf{Z}; Y) = \\min_{v \\in \\{1, \\dots, V\\}} I(Z_v; Y) $$
        """
        return torch.min(marginal_mis)

    def compute_uniqueness(self, marginal_mis: torch.Tensor, redundancy: torch.Tensor) -> torch.Tensor:
        """
        Computes the uniqueness for each view.
        
        Mathematical formulation:
        $$ \\text{Unq}(Z_v; Y | Z_{\\setminus v}) = I(Z_v; Y) - \\text{Red} $$
        """
        return marginal_mis - redundancy

    def compute_synergy_v_source(self, joint_mi: torch.Tensor, redundancy: torch.Tensor, total_uniqueness: torch.Tensor) -> torch.Tensor:
        """
        Computes the exact V-source synergy.
        
        Mathematical formulation:
        $$ \\text{Syn}(\\mathbf{Z}; Y) = I(\\mathbf{Z}; Y) - \\text{Red} - \\sum_{v} \\text{Unq} $$
        """
        return joint_mi - redundancy - total_uniqueness

    def compute_pairwise_pid(self, mi_i: torch.Tensor, mi_j: torch.Tensor, mi_ij: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Computes the PID atoms for a specific pair of views (i, j).
        
        Mathematical formulation:
        $$ \\text{Red}(Z_i, Z_j; Y) = \\min(I(Z_i; Y), I(Z_j; Y)) $$
        $$ \\text{Unq}(Z_i; Y|Z_j) = I(Z_i; Y) - \\text{Red} $$
        $$ \\text{Unq}(Z_j; Y|Z_i) = I(Z_j; Y) - \\text{Red} $$
        $$ \\text{Syn}(Z_i, Z_j; Y) = I(Z_i, Z_j; Y) - \\text{Red} - \\text{Unq}(Z_i) - \\text{Unq}(Z_j) $$
        """
        red = torch.min(mi_i, mi_j)
        unq_i = mi_i - red
        unq_j = mi_j - red
        syn = mi_ij - red - unq_i - unq_j
        
        return {
            'redundancy': red,
            'uniqueness_i': unq_i,
            'uniqueness_j': unq_j,
            'synergy': syn
        }

    def aggregate_pairwise_pid(self, marginal_mis: torch.Tensor, pairwise_joint_mis: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Aggregates the pairwise PID atoms across all \\binom{V}{2} pairs.
        The global unique information for view v is the average of its uniqueness 
        across all pairs containing v.
        The global synergy is the average of the pairwise synergies.
        
        Parameters
        ----------
        marginal_mis : torch.Tensor, shape (V,)
            Marginal MI bounds for each view.
        pairwise_joint_mis : torch.Tensor, shape (V, V)
            Matrix of joint MI bounds for all pairs (i, j). 
            Diagonal elements should be the marginal MIs.
        """
        V = marginal_mis.size(0)
        if V < 2:
            raise ValueError("Pairwise PID requires at least 2 views.")
            
        pairs = list(combinations(range(V), 2))
        
        global_uniqueness = torch.zeros(V, device=marginal_mis.device)
        pair_counts = torch.zeros(V, device=marginal_mis.device)
        synergies = []
        
        for i, j in pairs:
            mi_i = marginal_mis[i]
            mi_j = marginal_mis[j]
            mi_ij = pairwise_joint_mis[i, j]
            
            pid_pair = self.compute_pairwise_pid(mi_i, mi_j, mi_ij)
            
            global_uniqueness[i] += pid_pair['uniqueness_i']
            global_uniqueness[j] += pid_pair['uniqueness_j']
            pair_counts[i] += 1
            pair_counts[j] += 1
            
            synergies.append(pid_pair['synergy'])
            
        # Average uniqueness for each view
        global_uniqueness = global_uniqueness / pair_counts.clamp(min=1.0)
        
        # Average synergy across all pairs
        global_synergy = torch.stack(synergies).mean()
        
        return {
            'global_uniqueness': global_uniqueness,
            'global_synergy': global_synergy
        }