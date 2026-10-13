import torch
import torch.nn as nn
from data.datamodule import MultiViewDataModule
from data.datasets.hw import HWDataset
from models import FactorizedINN, ConsensusModule, NeuralPID_Estimator

# 1. Setup Data (Original HW Dataset)
datamodule = MultiViewDataModule(
    dataset_class=HWDataset,
    root_dir="/home/phd/datasets/hw",
    batch_size=256,
)
datamodule.setup()

# 2. Initialize Models
# Read actual view dimensions from the dataset
dataset = HWDataset(root_dir="/home/phd/datasets/hw")
view_dims_raw = [int(view.shape[1]) for view in dataset.views]

d_S, d_U = 4, 4
# Pre-consensus dimensions are d_S + d_U
pre_consensus_dims = [d_S + d_U for _ in view_dims_raw]
num_classes = 10 # HW has 10 classes

encoders = nn.ModuleList([
    FactorizedINN(input_dim=d_v, d_S=d_S, d_U=d_U) for d_v in view_dims_raw
])
consensus = ConsensusModule(d_S=d_S)
pid_estimator = NeuralPID_Estimator(view_dims=pre_consensus_dims, num_classes=num_classes)

# 3. Forward Pass & PID Estimation
for views, labels in datamodule.train_dataloader():
    shared_reps, unique_reps = [], []
    
    # Encode views
    for encoder, x_v in zip(encoders, views):
        s_v, u_v, _ = encoder(x_v)
        shared_reps.append(s_v)
        unique_reps.append(u_v)
        
    # Get pre-consensus representations for PID
    consensus_output = consensus(shared_reps, unique_reps=unique_reps)
    
    # Compute PID Atoms
    pid_atoms = pid_estimator(consensus_output.pre_consensus, labels)
    
    print(f"Redundancy: {pid_atoms.redundancy.item():.4f}")
    print(f"Synergy: {pid_atoms.synergy.item():.4f}")
    
    # The PID atoms can now be used to construct the L_PID loss term
    # L_pid = pid_atoms.redundancy + pid_atoms.uniqueness.sum() + pid_atoms.synergy
    # (-L_pid is maximized in the joint objective)
    break