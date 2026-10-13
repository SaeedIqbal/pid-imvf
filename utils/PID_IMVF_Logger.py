import torch
import numpy as np
from data.datamodule import MultiViewDataModule
from data.datasets.hw import HWDataset
from models import PID_IMVF
from utils import PID_IMVF_Logger, MetricsCalculator

# 1. Strictly enforce determinism to avoid random values in weight initialization
torch.manual_seed(42)
np.random.seed(42)

# 2. Setup Data (Original HW Dataset from /home/phd/datasets/)
datamodule = MultiViewDataModule(
    dataset_class=HWDataset,
    root_dir="/home/phd/datasets/hw",
    batch_size=256,
    num_workers=0  # Set to 0 for deterministic single-threaded loading
)
datamodule.setup()

# 3. Initialize Model
dataset = HWDataset(root_dir="/home/phd/datasets/hw")
view_dims = [int(view.shape[1]) for view in dataset.views]
num_classes = 10

model = PID_IMVF(
    view_dims=view_dims,
    d_S=4,
    d_U=4,
    num_classes=num_classes,
    num_coupling_layers=6,
    lambda_pred=1.0,
    lambda_noise=0.1,
    lambda_syn=0.1,
    lambda_id=1.0,
    lr=1e-3,
    weight_decay=1e-4
)

# 4. Initialize Logger
logger = PID_IMVF_Logger(log_dir="/home/phd/logs", experiment_name="hw_pid_imvf_deterministic_run")

# 5. Run one forward pass on the original dataset and log the mathematical quantities
metrics_calc = MetricsCalculator(num_classes=num_classes)

for batch_idx, (views, targets) in enumerate(datamodule.train_dataloader()):
    # Forward pass through the PID-IMVF architecture
    outputs = model(views, targets)
    losses = model._compute_losses(outputs, targets)
    
    # Log PID Atoms (Redundancy, Uniqueness, Synergy)
    logger.log_pid_atoms(epoch=1, step=batch_idx, pid_atoms={
        'redundancy': outputs['pid_atoms'].redundancy,
        'uniqueness': outputs['pid_atoms'].uniqueness,
        'synergy': outputs['pid_atoms'].synergy,
        'total_joint_mi': outputs['pid_atoms'].total_joint_mi
    })
    
    # Log Loss Components (CE, PID, VIB, Syn, ID, HSIC)
    logger.log_loss_components(epoch=1, step=batch_idx, losses=losses)
    
    # Compute and Log Epoch Summary using actual predictions from the original data
    preds = torch.argmax(outputs['logits'], dim=-1)
    metrics = metrics_calc.compute_all_metrics(targets, preds)
    logger.log_epoch_summary(epoch=1, metrics=metrics)
    
    break  # Process only one batch to demonstrate without synthetic evaluation loops

# 6. Close logger to free file handles
logger.close()