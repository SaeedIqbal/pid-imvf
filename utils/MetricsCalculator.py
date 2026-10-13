import torch
from data.datamodule import MultiViewDataModule
from data.datasets.animals import AnimalsDataset
from models import PID_IMVF
from utils import MetricsCalculator

# 1. Setup Data (Original Animals Dataset from /home/phd/datasets/)
datamodule = MultiViewDataModule(
    dataset_class=AnimalsDataset,
    root_dir="/home/phd/datasets/animals",
    batch_size=256,
    num_workers=4
)
datamodule.setup()

# 2. Load the trained PID-IMVF model
# (Assuming model is loaded from a checkpoint)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = PID_IMVF.load_from_checkpoint("checkpoints/pid_imvf_animals.ckpt")
model.to(device)

# 3. Evaluate on the strict inductive test split (80%)
test_metrics = MetricsCalculator.evaluate_dataloader(
    model=model,
    dataloader=datamodule.test_dataloader(),
    num_classes=50,  # Animals dataset has 50 classes
    device=device
)

print("--- Test Set Evaluation Results ---")
print(f"Accuracy (ACC):   {test_metrics['ACC'] * 100:.2f}%")
print(f"Macro-F1 Score:   {test_metrics['Macro-F1'] * 100:.2f}%")
print(f"Balanced Acc (BACC): {test_metrics['BACC'] * 100:.2f}%")