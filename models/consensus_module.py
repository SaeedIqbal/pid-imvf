import torch
import torch.nn as nn
from typing import List, Optional
from data.datamodule import MultiViewDataModule
from data.datasets.hw import HWDataset
from models import ConsensusModule, FactorizedINN

class ConsensusModule(nn.Module):
    """
    Euclidean Fréchet mean aggregation for cross-view shared factors (\hat{S}_v).
    
    Mathematical Formulation:
    Under the shared gauge assumption (A5), the inverse mappings operate in a common 
    Euclidean coordinate system. The consensus shared representation \hat{S} is computed 
    via the Euclidean Fréchet mean, which minimizes the sum of squared distances in R^{d_S}:
    
        \hat{S} = \arg\min_{Z \in \mathbb{R}^{d_S}} \sum_{v=1}^V \| Z - \hat{S}_v \|_2^2 
                = \frac{1}{V} \sum_{v=1}^V \hat{S}_v
                
    For missing-view robustness (Theorem 3), the partial consensus is computed over 
    the observed views \mathcal{V}_{obs}:
        \hat{S}^{(obs)} = \frac{1}{|\mathcal{V}_{obs}|} \sum_{v \in \mathcal{V}_{obs}} \hat{S}_v
    """
    
    def __init__(self, num_views: int):
        """
        Args:
            num_views (int): Total number of views V in the multi-view dataset.
        """
        super().__init__()
        self.num_views = num_views

    def forward(self, s_hat_list: List[torch.Tensor], view_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Computes the (partial) consensus of the shared factors.
        
        Args:
            s_hat_list (List[torch.Tensor]): List of V tensors, each of shape (batch_size, d_S).
            view_mask (Optional[torch.Tensor]): Boolean tensor of shape (V,) or (batch_size, V) 
                                                indicating which views are observed. 
                                                If None, full consensus is computed.
                                               
        Returns:
            torch.Tensor: Consensus tensor of shape (batch_size, d_S).
        """
        # Stack along a new view dimension: (batch_size, V, d_S)
        s_hat_stacked = torch.stack(s_hat_list, dim=1)
        
        if view_mask is not None:
            # Ensure mask is broadcastable to (batch_size, V, d_S)
            if view_mask.dim() == 1:
                view_mask = view_mask.unsqueeze(0)  # (1, V)
            
            # Expand to (batch_size, V, 1) for broadcasting with d_S
            view_mask = view_mask.unsqueeze(-1).float() 
            
            # Mask the stacked tensor (zero out unobserved views)
            s_hat_masked = s_hat_stacked * view_mask
            
            # Sum over views and divide by the number of observed views
            sum_s = torch.sum(s_hat_masked, dim=1)  # (batch_size, d_S)
            
            # Count observed views per sample, keepdim for safe broadcasting
            count_v = torch.sum(view_mask, dim=1, keepdim=True).clamp(min=1.0)  # (batch_size, 1)
            
            s_consensus = sum_s / count_v
        else:
            # Full consensus: simple mean over the view dimension
            s_consensus = torch.mean(s_hat_stacked, dim=1)
            
        return s_consensus

    def compute_consensus_loss(self, s_hat_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Computes the consensus loss to enforce cross-view alignment during training.
        
        Mathematical Formulation:
        The structural identifiability loss includes a term to align the shared factors:
            \mathcal{L}_{consensus} = \sum_{i \neq j}^V \| \hat{S}_i - \hat{S}_j \|_2^2
            
        Note: This is mathematically equivalent to minimizing the variance of the views 
        around their mean, scaled by 2V:
            \sum_{i \neq j} \| \hat{S}_i - \hat{S}_j \|^2 = 2V \sum_{v=1}^V \| \hat{S}_v - \bar{S} \|^2
            
        We compute the latter as it is more computationally efficient (O(V) instead of O(V^2)).
        
        Args:
            s_hat_list (List[torch.Tensor]): List of V tensors, each of shape (batch_size, d_S).
            
        Returns:
            torch.Tensor: Scalar consensus loss.
        """
        # Stack: (batch_size, V, d_S)
        s_hat_stacked = torch.stack(s_hat_list, dim=1)
        
        # Compute the mean (full consensus) across views
        s_mean = torch.mean(s_hat_stacked, dim=1, keepdim=True)  # (batch_size, 1, d_S)
        
        # Compute squared L2 differences from the mean for each view
        sq_diff = torch.sum((s_hat_stacked - s_mean) ** 2, dim=-1)  # (batch_size, V)
        
        # Sum over views and average over the batch
        loss = torch.mean(torch.sum(sq_diff, dim=-1))
        
        return loss

    import torch.nn as nn




# 1. Load the original dataset and read its true view dimensions.
dataset = HWDataset(root_dir="/home/phd/datasets/hw")
view_dims = [int(view.shape[1]) for view in dataset.views]

# 2. One factorized INN per view.
d_S, d_U = 4, 4
encoders = nn.ModuleList(
    [FactorizedINN(input_dim=d_v, d_S=d_S, d_U=d_U) for d_v in view_dims]
)
consensus = ConsensusModule(d_S=d_S)

# 3. Strictly inductive training split (10% / 10% / 80%).
datamodule = MultiViewDataModule(
    dataset_class=HWDataset,
    root_dir="/home/phd/datasets/hw",
    batch_size=256,
)
datamodule.setup()

for views, _ in datamodule.train_dataloader():
    shared_reps, unique_reps = [], []

    for encoder, x_v in zip(encoders, views):
        s_v, u_v, _ = encoder(x_v)
        shared_reps.append(s_v)
        unique_reps.append(u_v)

    output = consensus(shared_reps, unique_reps=unique_reps)

    # Classifier receives ONLY the consensus shared factor.
    # classifier_logits = classifier(output.shared)

    # PID estimator receives ONLY the pre-consensus factors.
    # pid_atoms = pid_estimator(output.pre_consensus)

    loss = output.gauge_loss
    loss.backward()