import torch
from data.datamodule import MultiViewDataModule
from data.datasets.hw import HWDataset
from models import PID_IMVF
from utils import MissingViewEvaluator

# 1. Setup Data (Original HW Dataset from /home/phd/datasets/)
datamodule = MultiViewDataModule(
    dataset_class=HWDataset,
    root_dir="/home/phd/datasets/hw",
    batch_size=256,
    num_workers=4
)
datamodule.setup()

# 2. Load the trained PID-IMVF model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = PID_IMVF.load_from_checkpoint("checkpoints/pid_imvf_hw.ckpt")
model.to(device)

# 3. Initialize the Missing View Evaluator
num_views = 6  # HW dataset has 6 views
num_classes = 10

evaluator = MissingViewEvaluator(
    model=model,
    dataloader=datamodule.test_dataloader(),
    num_views=num_views,
    num_classes=num_classes,
    device=device
)

# 4. Evaluate under different deterministic missing-view scenarios
print("--- Full View Baseline ---")
full_metrics = evaluator.evaluate_missing_view_scenario(strategy='all')
print(f"Macro-F1: {full_metrics['Macro-F1'] * 100:.2f}%\n")

print("--- Drop View 2 (Specific) ---")
drop_v2_metrics = evaluator.evaluate_missing_view_scenario(strategy='drop_specific', target_view_idx=1)
print(f"Macro-F1: {drop_v2_metrics['Macro-F1'] * 100:.2f}%\n")

print("--- Drop Last View ---")
drop_last_metrics = evaluator.evaluate_missing_view_scenario(strategy='drop_last')
print(f"Macro-F1: {drop_last_metrics['Macro-F1'] * 100:.2f}%")