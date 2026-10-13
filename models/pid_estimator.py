"""
Differentiable Neural PID Lattice Estimator for PID-IMVF.

Mathematical background
-----------------------
Instead of using unstable adversarial critics (e.g., MINE/NWJ) or intractable 
convex optimization (e.g., I_BROJA), we employ the Barber-Agakov classifier-based 
bound to estimate Mutual Information (MI) for continuous, high-dimensional variables.

1. MI Bound (Barber-Agakov):
   I(Z; Y) >= H(Y) - CE(q_phi(Y|Z))
   where H(Y) is the empirical entropy of the target, and CE is the cross-entropy 
   of a standard classifier head q_phi.

2. Williams-Beer Redundancy:
   Red(Z_1, ..., Z_V; Y) = min_v I(Z_v; Y)

3. Uniqueness:
   Unq(Z_v; Y | Z_{\setminus v}) = I(Z_v; Y) - Red

4. Synergy:
   Syn(Z; Y) = I(Z_1, ..., Z_V; Y) - Red - sum_v Unq(Z_v; Y | Z_{\setminus v})

This formulation requires exactly V + 1 lightweight classifier heads, 
ensuring strict O(N V d) scalability and avoiding min-max optimization.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from dataclasses import dataclass


__all__ = [
    "PIDAtoms",
    "ClassifierHead",
    "NeuralPID_Estimator",
]


# ===========================================================================
# Structured output
# ===========================================================================

@dataclass
class PIDAtoms:
    """
    Container for the Partial Information Decomposition (PID) atoms.

    Attributes
    ----------
    redundancy : torch.Tensor, scalar
        Williams-Beer redundancy: min_v I(Z_v; Y).
    uniqueness : torch.Tensor, shape (V,)
        Per-view uniqueness: I(Z_v; Y) - Red.
    synergy : torch.Tensor, scalar
        Higher-order synergy: I(Z; Y) - Red - sum(Unq).
    total_joint_mi : torch.Tensor, scalar
        Joint mutual information I(Z_1, ..., Z_V; Y).
    marginal_mis : torch.Tensor, shape (V,)
        Marginal mutual information bounds I(Z_v; Y) for each view.
    """
    redundancy: torch.Tensor
    uniqueness: torch.Tensor
    synergy: torch.Tensor
    total_joint_mi: torch.Tensor
    marginal_mis: torch.Tensor


# ===========================================================================
# Classifier Head
# ===========================================================================

class ClassifierHead(nn.Module):
    """
    Standard MLP classifier head q_phi(Y | Z) used to compute the 
    Barber-Agakov MI bound via cross-entropy.
    """
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z : torch.Tensor, shape (B, input_dim)
        
        Returns
        -------
        torch.Tensor, shape (B, num_classes)
            Raw logits (unnormalized).
        """
        return self.net(z)


# ===========================================================================
# Neural PID Estimator
# ===========================================================================

class NeuralPID_Estimator(nn.Module):
    """
    Differentiable Neural PID Lattice Estimator.
    
    Computes the PID atoms using V view-specific classifiers and 1 joint 
    classifier, completely avoiding adversarial min-max optimization.
    """
    def __init__(self, view_dims: List[int], num_classes: int, hidden_dim: int = 256):
        """
        Parameters
        ----------
        view_dims : List[int]
            List of dimensions for the pre-consensus representations [S_hat_v, U_hat_v].
        num_classes : int
            Number of target classes C.
        hidden_dim : int
            Hidden dimension for the classifier MLPs.
        """
        super().__init__()
        self.num_views = len(view_dims)
        self.num_classes = num_classes
        
        if self.num_views == 0:
            raise ValueError("view_dims must contain at least one dimension.")
            
        # V view-specific classifiers for marginal MI bounds
        self.view_classifiers = nn.ModuleList([
            ClassifierHead(dim, num_classes, hidden_dim) for dim in view_dims
        ])
        
        # 1 joint classifier for the full joint MI bound
        joint_dim = sum(view_dims)
        self.joint_classifier = ClassifierHead(joint_dim, num_classes, hidden_dim)
        
    def compute_entropy_y(self, y: torch.Tensor) -> torch.Tensor:
        """
        Computes the empirical entropy of the target labels H(Y).
        H(Y) = - sum_c p(c) log p(c)
        
        Note: Computed in torch.no_grad() as it is a constant for a fixed batch 
        and does not require gradient flow through the bincount operation.
        """
        batch_size = y.size(0)
        with torch.no_grad():
            counts = torch.bincount(y, minlength=self.num_classes).float()
            probs = counts / batch_size
            # Clamp to avoid log(0)
            probs = probs.clamp(min=1e-8)
            entropy = -torch.sum(probs * torch.log(probs))
        return entropy

    def compute_mi_bound(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes the Barber-Agakov lower bound on Mutual Information.
        I(Z; Y) >= H(Y) - CE(q_phi(Y|Z))
        """
        h_y = self.compute_entropy_y(y)
        ce = F.cross_entropy(logits, y)
        return h_y - ce

    def forward(self, pre_consensus_reps: List[torch.Tensor], y: torch.Tensor) -> PIDAtoms:
        """
        Computes the full PID lattice for the given pre-consensus representations.
        
        Parameters
        ----------
        pre_consensus_reps : List[torch.Tensor]
            List of V tensors, each of shape (B, d_S + d_U_v), representing 
            the concatenated [S_hat_v, U_hat_v] before consensus averaging.
        y : torch.Tensor
            Target labels of shape (B,).
            
        Returns
        -------
        PIDAtoms
            Container with redundancy, uniqueness, synergy, and MI estimates.
        """
        if len(pre_consensus_reps) != self.num_views:
            raise ValueError(
                f"Expected {self.num_views} pre-consensus representations, "
                f"got {len(pre_consensus_reps)}."
            )
            
        # 1. Compute marginal MI bounds for each view using view-specific classifiers
        marginal_mis = []
        for v, classifier in enumerate(self.view_classifiers):
            logits_v = classifier(pre_consensus_reps[v])
            mi_v = self.compute_mi_bound(logits_v, y)
            marginal_mis.append(mi_v)
        marginal_mis = torch.stack(marginal_mis) # Shape: (V,)
        
        # 2. Compute joint MI bound using the concatenated representation
        joint_rep = torch.cat(pre_consensus_reps, dim=-1)
        joint_logits = self.joint_classifier(joint_rep)
        joint_mi = self.compute_mi_bound(joint_logits, y)
        
        # 3. Compute PID atoms based on the Williams-Beer framework
        # Redundancy: minimum of marginal MIs
        redundancy = torch.min(marginal_mis)
        
        # Uniqueness: marginal MI minus redundancy
        uniqueness = marginal_mis - redundancy # Shape: (V,)
        total_uniqueness = torch.sum(uniqueness)
        
        # Synergy: joint MI minus redundancy and total uniqueness
        synergy = joint_mi - redundancy - total_uniqueness
        
        return PIDAtoms(
            redundancy=redundancy,
            uniqueness=uniqueness,
            synergy=synergy,
            total_joint_mi=joint_mi,
            marginal_mis=marginal_mis
        )

    def extra_repr(self) -> str:
        """Human-readable module signature for torch.nn repr."""
        return (
            f"num_views={self.num_views}, "
            f"num_classes={self.num_classes}, "
            f"estimator='Barber-Agakov'"
        )