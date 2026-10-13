import torch
from data.datamodule import MultiViewDataModule
from data.datasets.hw import HWDataset
from models import FactorizedINN
from utils import HSICCalculator

# 1. Setup Data (Original HW Dataset from /home/phd/datasets/)
datamodule = MultiViewDataModule(
    dataset_class=HWDataset,
    root_dir="/home/phd/datasets/hw",
    batch_size=256,
    num_workers=4
)
datamodule.setup()

# 2. Initialize Models and HSIC Calculator
# Read actual view dimensions directly from the original dataset
dataset = HWDataset(root_dir="/home/phd/datasets/hw")
view_dims = [int(view.shape[1]) for view in dataset.views]

d_S, d_U = 4, 4
# Initialize the Factorized INN for the first view
encoder_v1 = FactorizedINN(input_dim=view_dims[0], d_S=d_S, d_U=d_U)

# Initialize the HSIC Calculator with median heuristic for scale-invariance
hsic_calculator = HSICCalculator(median_heuristic=True)

# 3. Compute HSIC on the original dataset features
for views, _ in datamodule.train_dataloader():
    # Encode the first view into shared (S_hat) and unique (U_hat) factors
    s_v, u_v, _ = encoder_v1(views[0])
    
    # Compute HSIC between shared and unique factors
    # Mathematically: HSIC(\hat{S}_v, \hat{U}_v)
    hsic_val = hsic_calculator(s_v, u_v)
    
    print(f"HSIC between S_hat and U_hat for View 1: {hsic_val.item():.6f}")
    print("Target: HSIC should approach 0.0 as the model enforces strict nonlinear independence.")
    
    # This value is directly added to the joint objective as part of L_id
    # loss_id = hsic_val + gauge_penalty
    break