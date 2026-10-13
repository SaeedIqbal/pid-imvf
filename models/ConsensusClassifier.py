import torch
import torch.nn as nn
from data.datamodule import MultiViewDataModule
from data.datasets.hw import HWDataset
from models import FactorizedINN, ConsensusModule, ConsensusClassifier

# 1. Setup Data (Original HW Dataset)
datamodule = MultiViewDataModule(
    dataset_class=HWDataset,
    root_dir="/home/phd/datasets/hw",
    batch_size=256,
)
datamodule.setup()

# 2. Initialize Models
dataset = HWDataset(root_dir="/home/phd/datasets/hw")
view_dims_raw = [int(view.shape[1]) for view in dataset.views]

d_S, d_U = 4, 4
num_classes = 10 # HW has 10 classes

encoders = nn.ModuleList([
    FactorizedINN(input_dim=d_v, d_S=d_S, d_U=d_U) for d_v in view_dims_raw
])
consensus = ConsensusModule(d_S=d_S)
classifier = ConsensusClassifier(d_S=d_S, num_classes=num_classes)

# 3. Forward Pass & Classification
for views, labels in datamodule.train_dataloader():
    shared_reps, unique_reps = [], []
    
    # Encode views
    for encoder, x_v in zip(encoders, views):
        s_v, u_v, _ = encoder(x_v)
        shared_reps.append(s_v)
        unique_reps.append(u_v)
        
    # Get partial consensus (S_hat^(obs))
    # Note: unique_reps are passed to consensus ONLY for the PID estimator path.
    # The classifier strictly receives ONLY the 'shared' output.
    consensus_output = consensus(shared_reps, unique_reps=unique_reps)
    
    # Classify using ONLY the shared consensus representation
    logits = classifier(consensus_output.shared)
    
    # Compute classification loss
    loss_ce = classifier.compute_loss(logits, labels)
    
    print(f"Classification Loss (CE): {loss_ce.item():.4f}")
    print(f"Predictions shape: {classifier.predict(consensus_output.shared).shape}")
    
    break