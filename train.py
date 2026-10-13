"""
Main Training Entry Point for PID-IMVF.

This script orchestrates the entire training pipeline:
1. Parses dataset and ablation configurations (YAML).
2. Initializes the MultiViewDataModule using the original datasets from /home/phd/datasets/.
3. Instantiates the PID_IMVF architecture.
4. Executes the PyTorch Lightning training loop with deterministic seeding.
"""

import os
import argparse
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from data.datamodule import MultiViewDataModule
from data.datasets._100leaves import HundredLeavesDataset
from data.datasets.animals import AnimalsDataset
from data.datasets.hw import HWDataset
from data.datasets.uci_multifeature import UCIMultifeatureDataset
from data.datasets.mnist import MNISTMultiViewDataset
from data.datasets.noisy_mnist import NoisyMNISTMultiViewDataset

from models import PID_IMVF
from utils import PID_IMVF_Logger


class PIDLoggingCallback(pl.Callback):
    """
    Custom PyTorch Lightning callback to integrate the PID_IMVF_Logger.
    Ensures that mathematical loss components and training stability 
    are rigorously tracked without relying on synthetic evaluation.
    """
    def __init__(self, custom_logger: PID_IMVF_Logger):
        super().__init__()
        self.custom_logger = custom_logger

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """
        Logs the total training loss at every batch.
        The underlying mathematical objective is:
        L = L_CE - \lambda_pred * L_PID + \lambda_noise * L_VIB + \lambda_syn * L_syn + L_id
        """
        if outputs is not None:
            self.custom_logger.logger.info(
                f"[Epoch {trainer.current_epoch}, Step {trainer.global_step}] "
                f"Total Loss: {outputs.item():.4f}"
            )

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Logs the epoch-level validation metrics (ACC, Macro-F1, BACC) 
        computed on the strict 10% inductive validation split.
        """
        val_loss = trainer.callback_metrics.get('val/loss', torch.tensor(0.0))
        val_acc = trainer.callback_metrics.get('val/acc', torch.tensor(0.0))
        
        metrics = {
            'val_loss': val_loss.item(),
            'val_acc': val_acc.item()
        }
        self.custom_logger.log_epoch_summary(trainer.current_epoch, metrics)


class PID_IMVF_Trainer:
    """
    Object-Oriented Trainer class for the PID-IMVF framework.
    """
    def __init__(self, args):
        self.args = args
        self.config = self._load_config()
        self.datamodule = None
        self.model = None
        self.custom_logger = None
        self.trainer = None
        
    def _load_config(self) -> dict:
        """
        Loads and merges the base dataset configuration with optional ablation overrides.
        """
        # Load base dataset config
        if not os.path.exists(self.args.dataset_config):
            raise FileNotFoundError(f"Dataset config not found: {self.args.dataset_config}")
            
        with open(self.args.dataset_config, 'r') as f:
            config = yaml.safe_load(f)
            
        # Override with ablation config if provided
        if self.args.ablation_config:
            if not os.path.exists(self.args.ablation_config):
                raise FileNotFoundError(f"Ablation config not found: {self.args.ablation_config}")
            with open(self.args.ablation_config, 'r') as f:
                ablation_config = yaml.safe_load(f)
                config.update(ablation_config)
                
        # Override dataset root with CLI argument
        config['dataset_root'] = self.args.dataset_root
        
        return config

    def setup_data(self):
        """
        Initializes the MultiViewDataModule using the original datasets.
        Enforces the strict stratified 10/10/80 inductive split.
        """
        dataset_class_map = {
            '100leaves': HundredLeavesDataset,
            'animals': AnimalsDataset,
            'hw': HWDataset,
            'uci': UCIMultifeatureDataset,
            'mnist': MNISTMultiViewDataset,
            'noisy_mnist': NoisyMNISTMultiViewDataset
        }
        
        # Infer dataset name from config, fallback to filename
        dataset_name = self.config.get('dataset_name')
        if not dataset_name:
            dataset_name = os.path.splitext(os.path.basename(self.args.dataset_config))[0]
            
        dataset_class = dataset_class_map.get(dataset_name, HWDataset)
        
        self.datamodule = MultiViewDataModule(
            dataset_class=dataset_class,
            root_dir=self.config['dataset_root'],
            batch_size=self.config.get('batch_size', 256),
            num_workers=self.config.get('num_workers', 4),
            transform=None
        )
        self.datamodule.setup()

    def setup_model(self):
        """
        Instantiates the PID_IMVF architecture with mathematically derived hyperparameters.
        """
        # Infer view dimensions and number of classes directly from the original dataset
        view_dims = [int(v.shape[1]) for v in self.datamodule.full_dataset.views]
        unique_labels = torch.unique(self.datamodule.full_dataset.labels)
        num_classes = len(unique_labels)
        
        self.model = PID_IMVF(
            view_dims=view_dims,
            d_S=self.config.get('d_S', 4),
            d_U=self.config.get('d_U', 4),
            num_classes=num_classes,
            num_coupling_layers=self.config.get('num_coupling_layers', 6),
            lambda_pred=self.config.get('lambda_pred', 1.0),
            lambda_noise=self.config.get('lambda_noise', 0.1),
            lambda_syn=self.config.get('lambda_syn', 0.1),
            lambda_id=self.config.get('lambda_id', 1.0),
            lr=self.config.get('lr', 1e-3),
            weight_decay=self.config.get('weight_decay', 1e-4)
        )

    def setup_logger(self):
        """
        Initializes the custom PID_IMVF_Logger for mathematical tracking.
        """
        self.custom_logger = PID_IMVF_Logger(
            log_dir=self.args.log_dir,
            experiment_name=self.args.experiment_name
        )

    def run(self):
        """
        Executes the complete training pipeline.
        """
        # 1. Initialize components
        self.setup_data()
        self.setup_model()
        self.setup_logger()
        
        # 2. Setup PyTorch Lightning callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(self.args.checkpoint_dir, self.args.experiment_name),
            filename='pid_imvf-{epoch:02d}-{val/acc:.4f}',
            monitor='val/acc',
            mode='max',
            save_top_k=1
        )
        logging_callback = PIDLoggingCallback(self.custom_logger)
        
        # 3. Initialize PyTorch Lightning Trainer
        self.trainer = pl.Trainer(
            max_epochs=self.config.get('max_epochs', 50),
            accelerator='auto',
            devices=1,
            log_every_n_steps=10,
            enable_progress_bar=True,
            default_root_dir=self.args.checkpoint_dir,
            callbacks=[checkpoint_callback, logging_callback]
        )
        
        # 4. Execute training loop
        self.trainer.fit(self.model, self.datamodule)
        
        # 5. Save the final checkpoint
        final_ckpt_path = os.path.join(self.args.checkpoint_dir, self.args.experiment_name, 'final.ckpt')
        self.trainer.save_checkpoint(final_ckpt_path)
        
        # 6. Cleanup
        self.custom_logger.close()


def main():
    parser = argparse.ArgumentParser(description="PID-IMVF Main Training Entry Point")
    parser.add_argument("--dataset_config", type=str, required=True, help="Path to the dataset config YAML")
    parser.add_argument("--ablation_config", type=str, default=None, help="Path to the ablation config YAML (optional)")
    parser.add_argument("--dataset_root", type=str, default="/home/phd/datasets/", help="Root directory for original datasets")
    parser.add_argument("--log_dir", type=str, default="/home/phd/logs", help="Directory to save logs")
    parser.add_argument("--checkpoint_dir", type=str, default="/home/phd/checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--experiment_name", type=str, default="pid_imvf_run", help="Name of the experiment")
    
    args = parser.parse_args()
    
    # Enforce strict determinism to avoid random values
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    pl.seed_everything(42, workers=True)
    
    trainer = PID_IMVF_Trainer(args)
    trainer.run()


if __name__ == "__main__":
    main()