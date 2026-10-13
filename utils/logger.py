"""
Custom Logging Utility for PID-IMVF.

Mathematical background
-----------------------
This module provides a structured logging interface to track the core 
mathematical quantities of the PID-IMVF framework during training. 
It logs the Partial Information Decomposition (PID) atoms, the components 
of the joint optimization objective, and the epoch-level evaluation metrics.

1. PID Atoms Tracked:
   Redundancy: \\text{Red}(\\mathbf{Z}; Y) = \\min_{v} I(Z_v; Y)
   Uniqueness: \\text{Unq}(Z_v; Y | Z_{\\setminus v}) = I(Z_v; Y) - \\text{Red}
   Synergy: \\text{Syn}(\\mathbf{Z}; Y) = I(\\mathbf{Z}; Y) - \\text{Red} - \\sum_{v} \\text{Unq}

2. Joint Objective Tracked:
   \\mathcal{L} = \\mathcal{L}_{\\text{CE}} - \\lambda_{\\text{pred}}\\mathcal{L}_{\\text{PID}} + \\lambda_{\\text{noise}}\\mathcal{L}_{\\text{VIB}} + \\lambda_{\\text{syn}}\\mathcal{L}_{\\text{syn}} + \\mathcal{L}_{\\text{id}}

3. Evaluation Metrics Tracked:
   Accuracy (ACC): \\frac{1}{N} \\sum_{i=1}^{N} \\mathbb{I}(\\hat{y}_i = y_i)
   Macro-F1: \\frac{1}{C} \\sum_{c=1}^{C} \\frac{2 \\cdot P_c \\cdot R_c}{P_c + R_c}
   Balanced Accuracy (BACC): \\frac{1}{C} \\sum_{c=1}^{C} \\text{Recall}_c
"""

from __future__ import annotations

import os
import csv
import logging
import torch
from typing import Dict, Any, List


__all__ = [
    "PID_IMVF_Logger",
]


class PID_IMVF_Logger:
    """
    Custom logging utility for the PID-IMVF framework.
    Tracks PID atom magnitudes, loss components, and training stability 
    using the original datasets from /home/phd/datasets/.
    """
    def __init__(self, log_dir: str, experiment_name: str):
        """
        Initializes the logger, setting up the directory structure, 
        standard Python logging, and CSV files for structured metric tracking.
        
        Parameters
        ----------
        log_dir : str
            Base directory to store the logs.
        experiment_name : str
            Name of the specific experiment/run.
        """
        self.log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Setup standard Python logging
        self.logger = logging.getLogger(f"PID_IMVF_{experiment_name}")
        self.logger.setLevel(logging.INFO)
        
        # Prevent duplicate handlers if re-initialized
        if not self.logger.handlers:
            fh = logging.FileHandler(os.path.join(self.log_dir, 'training.log'))
            fh.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
        
        # CSV paths for structured metric tracking
        self.pid_csv_path = os.path.join(self.log_dir, 'pid_atoms.csv')
        self.loss_csv_path = os.path.join(self.log_dir, 'loss_components.csv')
        
        self._initialize_csv(self.pid_csv_path, ['epoch', 'step', 'redundancy', 'uniqueness', 'synergy', 'joint_mi'])
        self._initialize_csv(self.loss_csv_path, ['epoch', 'step', 'loss_ce', 'loss_pid', 'loss_vib', 'loss_syn', 'loss_id', 'loss_hsic'])
        
        self.logger.info(f"Logger initialized for experiment: {experiment_name}")

    def _initialize_csv(self, filepath: str, headers: List[str]):
        """
        Creates a CSV file with the specified headers if it doesn't exist.
        Reused by both PID and Loss CSV initializations.
        """
        if not os.path.exists(filepath):
            with open(filepath, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    def _append_to_csv(self, filepath: str, row: List[Any]):
        """
        Appends a row of data to the specified CSV file.
        Reused by all logging methods to write structured data.
        """
        with open(filepath, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def log_pid_atoms(self, epoch: int, step: int, pid_atoms: Dict[str, torch.Tensor]):
        """
        Logs the Partial Information Decomposition (PID) atoms.
        
        Mathematical definitions tracked:
        Redundancy: \\text{Red}(\\mathbf{Z}; Y) = \\min_{v} I(Z_v; Y)
        Uniqueness: \\text{Unq}(Z_v; Y | Z_{\\setminus v}) = I(Z_v; Y) - \\text{Red}
        Synergy: \\text{Syn}(\\mathbf{Z}; Y) = I(\\mathbf{Z}; Y) - \\text{Red} - \\sum_{v} \\text{Unq}
        """
        red = pid_atoms['redundancy'].item()
        unq = pid_atoms['uniqueness'].sum().item()
        syn = pid_atoms['synergy'].item()
        joint_mi = pid_atoms['total_joint_mi'].item()
        
        self._append_to_csv(self.pid_csv_path, [epoch, step, red, unq, syn, joint_mi])
        self.logger.info(f"[Epoch {epoch}, Step {step}] PID Atoms -> Red: {red:.4f}, Unq: {unq:.4f}, Syn: {syn:.4f}, Joint MI: {joint_mi:.4f}")

    def log_loss_components(self, epoch: int, step: int, losses: Dict[str, torch.Tensor]):
        """
        Logs the components of the joint optimization objective.
        
        Mathematical objective tracked:
        \\mathcal{L} = \\mathcal{L}_{\\text{CE}} - \\lambda_{\\text{pred}}\\mathcal{L}_{\\text{PID}} + \\lambda_{\\text{noise}}\\mathcal{L}_{\\text{VIB}} + \\lambda_{\\text{syn}}\\mathcal{L}_{\\text{syn}} + \\mathcal{L}_{\\text{id}}
        """
        row = [epoch, step]
        for key in ['loss_ce', 'loss_pid', 'loss_vib', 'loss_syn', 'loss_id', 'loss_hsic']:
            row.append(losses[key].item())
            
        self._append_to_csv(self.loss_csv_path, row)
        self.logger.info(f"[Epoch {epoch}, Step {step}] Losses -> CE: {losses['loss_ce'].item():.4f}, Total: {losses['loss'].item():.4f}")

    def log_epoch_summary(self, epoch: int, metrics: Dict[str, float]):
        """
        Logs the summary metrics for a completed epoch.
        
        Tracks:
        Accuracy (ACC): \\frac{1}{N} \\sum_{i=1}^{N} \\mathbb{I}(\\hat{y}_i = y_i)
        Macro-F1: \\frac{1}{C} \\sum_{c=1}^{C} \\frac{2 \\cdot P_c \\cdot R_c}{P_c + R_c}
        Balanced Accuracy (BACC): \\frac{1}{C} \\sum_{c=1}^{C} \\text{Recall}_c
        """
        summary_str = f"--- Epoch {epoch} Summary --- "
        for k, v in metrics.items():
            summary_str += f"{k}: {v:.4f} | "
        self.logger.info(summary_str.strip())

    def close(self):
        """Closes all logging handlers to free up file resources."""
        for handler in self.logger.handlers:
            handler.close()
        self.logger.info("Logger closed and resources freed.")