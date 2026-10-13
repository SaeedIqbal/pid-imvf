import torch
from data.datamodule import MultiViewDataModule
from data.datasets.hw import HWDataset
from models import FactorizedINN, NeuralPID_Estimator
from utils import PIDMathCalculator

# 1. Setup Data (Original HW Dataset from /home/phd/datasets/)
datamodule = MultiViewDataModule(
    dataset_class=HWDataset,
    root_dir="/home/phd/datasets/hw",
    batch_size=256,
    num_workers=4
)
datamodule.setup()

# 2. Initialize Models
dataset = HWDataset(root_dir="/home/phd/datasets/hw")
view_dims = [int(view.shape[1]) for view in dataset.views]
num_classes = 10  # HW has 10 classes
num_views = len(view_dims)

d_S, d_U = 4, 4
encoders = [FactorizedINN(input_dim=d_v, d_S=d_S, d_U=d_U) for d_v in view_dims]

pre_consensus_dims = [d_S + d_U for _ in view_dims]
pid_estimator = NeuralPID_Estimator(view_dims=pre_consensus_dims, num_classes=num_classes)

# Initialize PID Math Calculator
pid_math = PIDMathCalculator(num_classes=num_classes)

# 3. Compute PID Atoms on the original dataset
for views, labels in datamodule.train_dataloader():
    pre_consensus_reps = []
    
    # Encode views to get pre-consensus representations [S_hat_v, U_hat_v]
    for encoder, x_v in zip(encoders, views):
        s_v, u_v, _ = encoder(x_v)
        z_v = torch.cat([s_v, u_v], dim=-1)
        pre_consensus_reps.append(z_v)
        
    # Get deterministic logits from the Neural PID Estimator
    marginal_logits = [classifier(z_v) for classifier, z_v in zip(pid_estimator.view_classifiers, pre_consensus_reps)]
    joint_rep = torch.cat(pre_consensus_reps, dim=-1)
    joint_logits = pid_estimator.joint_classifier(joint_rep)
    
    # 1. Compute MI bounds using PIDMathCalculator
    marginal_mis = torch.stack([pid_math.compute_mi_bound(logits, labels) for logits in marginal_logits])
    joint_mi = pid_math.compute_mi_bound(joint_logits, labels)
    
    # 2. Compute Exact V-Source PID
    redundancy = pid_math.compute_williams_beer_redundancy(marginal_mis)
    uniqueness = pid_math.compute_uniqueness(marginal_mis, redundancy)
    synergy = pid_math.compute_synergy_v_source(joint_mi, redundancy, uniqueness.sum())
    
    print(f"--- Exact V-Source PID Atoms ---")
    print(f"Redundancy: {redundancy.item():.4f}")
    print(f"Total Uniqueness: {uniqueness.sum().item():.4f}")
    print(f"Synergy: {synergy.item():.4f}")
    
    # 3. Compute Pairwise PID Aggregation (for V > 2 views)
    # Construct the pairwise joint MI matrix. 
    # Note: In the full pipeline, dedicated pairwise joint classifiers would populate the off-diagonal.
    # Here, we use the full joint MI as a conservative mathematical demonstration of the aggregation logic.
    pairwise_joint_mis = torch.zeros(num_views, num_views, device=labels.device)
    for i in range(num_views):
        for j in range(num_views):
            if i == j:
                pairwise_joint_mis[i, j] = marginal_mis[i]
            else:
                pairwise_joint_mis[i, j] = joint_mi 
                
    pairwise_pid = pid_math.aggregate_pairwise_pid(marginal_mis, pairwise_joint_mis)
    
    print(f"\n--- Pairwise PID Aggregation (Averaged across {num_views * (num_views - 1) // 2} pairs) ---")
    print(f"Global Uniqueness per view: {pairwise_pid['global_uniqueness']}")
    print(f"Global Synergy: {pairwise_pid['global_synergy'].item():.4f}")
    
    break