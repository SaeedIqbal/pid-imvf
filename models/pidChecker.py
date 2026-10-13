import pytorch_lightning as pl
from data.datamodule import MultiViewDataModule
from data.datasets.hw import HWDataset
from models import PID_IMVF

# 1. Setup Data (Original HW Dataset from /home/phd/datasets/)
datamodule = MultiViewDataModule(
    dataset_class=HWDataset,
    root_dir="/home/phd/datasets/hw",
    batch_size=256,
    num_workers=4
)
datamodule.setup()

# 2. Initialize Model
# Read actual view dimensions directly from the original dataset
dataset = HWDataset(root_dir="/home/phd/datasets/hw")
view_dims = [int(view.shape[1]) for view in dataset.views]
num_classes = 10  # HW has 10 classes

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

# 3. Train the Model
# PyTorch Lightning handles the training loop, logging, and device placement automatically
trainer = pl.Trainer(
    max_epochs=50,
    accelerator='auto',
    devices=1,
    log_every_n_steps=10
)

trainer.fit(model, datamodule)

# 4. Evaluate on the strict inductive test split (80%)
trainer.test(model, datamodule)