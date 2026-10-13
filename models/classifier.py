"""
Downstream classification head for PID-IMVF.

Mathematical background
-----------------------
To guarantee missing-view robustness, the classification head h_phi is strictly 
restricted to operate on the partial consensus shared representation S_hat^(obs).

1. Partial Consensus (computed by ConsensusModule):
   S_hat^(obs) = (1 / |V_obs|) * sum_{v in V_obs} S_hat_v

2. Classification:
   Y_hat = h_phi(S_hat^(obs))

By explicitly excluding the view-specific factors U_hat_v from the input, 
this architecture enforces the missing-view robustness bound established in 
Theorem 3. This prevents the classifier from relying on fragile, view-dependent 
synergistic information that is irrecoverably lost during test-time view dropout, 
ensuring equitable predictions even under severe observation constraints.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


__all__ = [
    "ConsensusClassifier",
]


class ConsensusClassifier(nn.Module):
    """
    Downstream classification head strictly restricted to the partial consensus 
    shared representation $\hat{S}^{(\text{obs})}$.
    """
    def __init__(
        self, 
        d_S: int, 
        num_classes: int, 
        hidden_dims: Optional[List[int]] = None, 
        dropout_rate: float = 0.5
    ):
        """
        Parameters
        ----------
        d_S : int
            Dimension of the shared latent subspace.
        num_classes : int
            Number of target classes $C$.
        hidden_dims : List[int], optional
            Dimensions of the hidden layers in the classification MLP. 
            Defaults to [256, 128].
        dropout_rate : float, optional
            Dropout probability for regularization. Defaults to 0.5.
        """
        super().__init__()
        self.d_S = int(d_S)
        self.num_classes = int(num_classes)
        
        if hidden_dims is None:
            hidden_dims = [256, 128]
            
        self.hidden_dims = hidden_dims
        self.dropout_rate = float(dropout_rate)
        
        # Build the classification network h_phi
        layers = []
        in_dim = self.d_S
        for h_dim in self.hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=self.dropout_rate))
            in_dim = h_dim
            
        # Final classification layer mapping to logits
        layers.append(nn.Linear(in_dim, self.num_classes))
        
        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize weights using Kaiming normal initialization for ReLU activations,
        ensuring stable gradient flow during the optimization of the joint objective.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, s_obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the classifier.
        
        Parameters
        ----------
        s_obs : torch.Tensor, shape (batch_size, d_S)
            The partial consensus shared representation $\hat{S}^{(\text{obs})}$.
            Note: View-specific factors $\hat{U}_v$ are intentionally excluded 
            from this input to guarantee missing-view robustness.
            
        Returns
        -------
        torch.Tensor, shape (batch_size, num_classes)
            Raw, unnormalized logits for each class.
        """
        if s_obs.dim() != 2:
            raise ValueError(
                f"Expected 2D input tensor (batch_size, d_S), got {s_obs.dim()}D tensor."
            )
        if s_obs.shape[1] != self.d_S:
            raise ValueError(
                f"Expected input dimension d_S={self.d_S}, got {s_obs.shape[1]}."
            )
            
        return self.network(s_obs)

    def predict(self, s_obs: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels from the partial consensus representation.
        
        Parameters
        ----------
        s_obs : torch.Tensor, shape (batch_size, d_S)
            
        Returns
        -------
        torch.Tensor, shape (batch_size,)
            Predicted class indices (argmax of logits).
        """
        logits = self.forward(s_obs)
        return torch.argmax(logits, dim=-1)

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the Cross-Entropy loss for the classification task.
        
        Mathematical formulation:
        $\mathcal{L}_{\text{CE}} = -\frac{1}{N} \sum_{i=1}^{N} \log \left( \frac{\exp(\hat{y}_{i, y_i})}{\sum_{c=1}^{C} \exp(\hat{y}_{i, c})} \right)$
        
        Parameters
        ----------
        logits : torch.Tensor, shape (batch_size, num_classes)
            Raw output from the forward pass.
        targets : torch.Tensor, shape (batch_size,)
            Ground truth class labels (long tensor).
            
        Returns
        -------
        torch.Tensor, scalar
            The computed cross-entropy loss.
        """
        if targets.dtype != torch.long:
            targets = targets.long()
        return F.cross_entropy(logits, targets)

    def extra_repr(self) -> str:
        """Human-readable module signature for torch.nn repr."""
        return (
            f"d_S={self.d_S}, "
            f"num_classes={self.num_classes}, "
            f"hidden_dims={self.hidden_dims}, "
            f"dropout_rate={self.dropout_rate}"
        )