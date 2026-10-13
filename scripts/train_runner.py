#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Object-oriented training / evaluation runner for PID-IMVF.

Manuscript objective implemented end-to-end:

    L = L_CE - lambda_pred * L_PID
          + lambda_noise * L_VIB
          + lambda_syn   * L_syn
          + L_id

with

    L_PID = Red(Z; Y) + sum_v Unq(Z_v; Y | Z_-v) + Syn(Z; Y)
    Red   = min_v I(Z_v; Y)
    I(Z; Y) >= H(Y) - CE(q_phi(Y | Z))          (Barber-Agakov)
    L_id  = mean_v HSIC(S_hat_v, U_hat_v) + L_gauge

and the missing-view consensus

    S_hat^(obs) = (1 / |V_obs|) * sum_{v in V_obs} S_hat_v .

All data are read from the original benchmarks under --data-root.
No synthetic samples and no hard-coded evaluation scores are used.
"""

from __future__ import annotations

import os

# Must be set before importing torch for deterministic CUDA reductions.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONHASHSEED", "0")

import argparse
import importlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import PID_IMVF
from utils import MetricsCalculator, MissingViewSimulator, PID_IMVF_Logger


# ===========================================================================
# 1. Dataset specification and registry
# ===========================================================================

@dataclass(frozen=True)
class DatasetSpec:
    """Static description of an original benchmark (never its values)."""

    name: str
    module: str
    class_name: str
    directory: str
    d_S: int
    d_U: int
    expected_views: int
    expected_classes: int
    needs_train_flag: bool = False
    mat_file: Optional[str] = None


class DatasetRegistry:
    """
    Resolves dataset names to the loaders defined in data/datasets/.

    Latent dimensions respect the immersion constraint of assumption (A3):
        d_S + d_U <= min_v d_v
    """

    _SPECS: Dict[str, DatasetSpec] = {
        "100leaves": DatasetSpec(
            "100leaves", "data.datasets._100leaves", "HundredLeavesDataset",
            "100leaves", d_S=32, d_U=32,
            expected_views=3, expected_classes=100, mat_file="100leaves.mat",
        ),
        "animals": DatasetSpec(
            "animals", "data.datasets.animals", "AnimalsDataset",
            "animals", d_S=32, d_U=64,
            expected_views=2, expected_classes=50, mat_file="awa2_features.mat",
        ),
        "hw": DatasetSpec(
            "hw", "data.datasets.hw", "HWDataset",
            "hw", d_S=4, d_U=2,
            expected_views=6, expected_classes=10, mat_file="mfeat.mat",
        ),
        "mnist": DatasetSpec(
            "mnist", "data.datasets.mnist", "MNISTMultiViewDataset",
            "mnist", d_S=4, d_U=4,
            expected_views=3, expected_classes=10, needs_train_flag=True,
        ),
        "noisy_mnist": DatasetSpec(
            "noisy_mnist", "data.datasets.noisy_mnist", "NoisyMNISTMultiViewDataset",
            "noisy_mnist", d_S=4, d_U=64,
            expected_views=2, expected_classes=10, needs_train_flag=True,
        ),
        "uci": DatasetSpec(
            "uci", "data.datasets.uci_multifeature", "UCIMultifeatureDataset",
            "uci_multifeature", d_S=4, d_U=2,
            expected_views=3, expected_classes=10, mat_file="mfeat.mat",
        ),
    }

    @classmethod
    def names(cls) -> List[str]:
        return list(cls._SPECS.keys())

    @classmethod
    def get(cls, name: str) -> DatasetSpec:
        if name not in cls._SPECS:
            raise KeyError(
                f"Unknown dataset '{name}'. Available: {', '.join(cls.names())}"
            )
        return cls._SPECS[name]

    @classmethod
    def preflight(cls, spec: DatasetSpec, data_root: str) -> Path:
        """
        Fail loudly if the ORIGINAL data is absent.
        Never downloads, synthesises, or substitutes a dataset.
        """
        directory = Path(data_root) / spec.directory

        if not directory.is_dir():
            raise FileNotFoundError(
                f"[{spec.name}] Original dataset directory not found: {directory}"
            )

        if spec.mat_file is not None:
            if not (directory / spec.mat_file).is_file():
                raise FileNotFoundError(
                    f"[{spec.name}] Expected original file "
                    f"{directory / spec.mat_file}. Place the unmodified "
                    f"UCI/AwA2 .mat export there."
                )
            return directory

        # MNIST-family benchmarks: require the real torchvision archive on disk
        # so that the loaders never trigger a network download.
        raw = directory / "MNIST" / "raw"
        required_stems = (
            "train-images-idx3-ubyte",
            "train-labels-idx1-ubyte",
            "t10k-images-idx3-ubyte",
            "t10k-labels-idx1-ubyte",
        )
        for stem in required_stems:
            if not any(
                (raw / stem).exists(),
                (raw / f"{stem}.gz").exists(),
            ):
                raise FileNotFoundError(
                    f"[{spec.name}] Missing original MNIST artifact "
                    f"{raw / stem}[.gz]. The runner does not download data."
                )
        return directory

    @classmethod
    def instantiate(cls, spec: DatasetSpec, data_root: str) -> Dataset:
        module = importlib.import_module(spec.module)
        dataset_class = getattr(module, spec.class_name)
        kwargs = {"root_dir": str(Path(data_root) / spec.directory)}
        if spec.needs_train_flag:
            kwargs["train"] = True
        return dataset_class(**kwargs)


# ===========================================================================
# 2. Label canonicalisation and dataset adaptation
# ===========================================================================

class ClassIndexMapper:
    """
    Deterministically maps original label ids to {0, ..., C-1}.

    UCI and AwA2 .mat exports frequently use 1-based labels. Cross-entropy
    requires 0-based contiguous indices; this mapping is a relabelling of
    the ORIGINAL targets, not a modification of the data.
    """

    def __init__(self) -> None:
        self._values: Optional[torch.Tensor] = None
        self.num_classes: int = 0

    def fit(self, labels: torch.Tensor) -> "ClassIndexMapper":
        values = torch.unique(labels.to(torch.long))
        self._values = values
        self.num_classes = int(values.numel())
        return self

    def transform(self, labels: torch.Tensor) -> torch.Tensor:
        if self._values is None:
            raise RuntimeError("ClassIndexMapper.fit() must be called first.")
        flat = labels.to(torch.long).reshape(-1)
        return torch.searchsorted(self._values, flat)


class MultiViewDatasetAdapter(Dataset):
    """
    Uniform adapter over the six original multi-view loaders.

    Responsibilities:
      * recover the true label vector (including torchvision-style datasets);
      * canonicalise label indices;
      * flatten each view to R^{d_v} so that g_v receives (B, d_v);
      * expose the true per-view dimensions.
    """

    def __init__(self, raw_dataset: Dataset) -> None:
        self.raw = raw_dataset
        self._raw_labels = self._extract_labels(raw_dataset)

        if self._raw_labels.numel() != len(raw_dataset):
            raise ValueError("Label count does not match the number of samples.")

        self.mapper = ClassIndexMapper().fit(self._raw_labels)
        self.labels = self.mapper.transform(self._raw_labels)

        first_sample = raw_dataset[0]
        views, _ = first_sample
        self.view_dims: List[int] = [
            int(torch.as_tensor(view).numel()) for view in views
        ]
        self.num_views = len(self.view_dims)

        if self.num_views < 2:
            raise ValueError("PID-IMVF requires at least two views (V >= 2).")

    @staticmethod
    def _extract_labels(raw_dataset: Dataset) -> torch.Tensor:
        if hasattr(raw_dataset, "labels"):
            values = np.asarray(raw_dataset.labels).reshape(-1)
            return torch.as_tensor(values, dtype=torch.long)

        base = getattr(raw_dataset, "base_dataset", None)
        if base is not None and hasattr(base, "targets"):
            values = np.asarray(list(base.targets)).reshape(-1)
            return torch.as_tensor(values, dtype=torch.long)

        # Deterministic scan of the ORIGINAL targets (no augmentation needed
        # for the labels themselves).
        values = [int(raw_dataset[i][1]) for i in range(len(raw_dataset))]
        return torch.as_tensor(values, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.raw)

    def __getitem__(self, index: int) -> Tuple[List[torch.Tensor], int]:
        views, _ = self.raw[int(index)]
        flattened = [
            torch.as_tensor(view, dtype=torch.float32).reshape(-1)
            for view in views
        ]
        target = int(self.labels[int(index)].item())
        return flattened, target


# ===========================================================================
# 3. Deterministic stratified 10 / 10 / 80 split
# ===========================================================================

class DeterministicStratifiedSplitter:
    """
    RNG-free stratified split.

    For each class c with n_c ordered sample indices, systematic sampling
    selects
        p_j = floor((2j + 1) * n_c / (2 k)),   j = 0, ..., k-1,
    where k = floor(fraction * n_c). This spreads the selected indices
    across the whole class instead of taking a contiguous prefix.

    The same procedure is applied to the remainder for the validation set;
    everything left becomes the test set. No random permutation is used.
    """

    def __init__(
        self, train_fraction: float = 0.10, val_fraction: float = 0.10
    ) -> None:
        if not (0.0 < train_fraction < 1.0 and 0.0 < val_fraction < 1.0):
            raise ValueError("Split fractions must lie in (0, 1).")
        if train_fraction + val_fraction >= 1.0:
            raise ValueError("train_fraction + val_fraction must be < 1.")
        self.train_fraction = float(train_fraction)
        self.val_fraction = float(val_fraction)

    @staticmethod
    def _systematic_take(
        pool: List[int], count: int
    ) -> Tuple[List[int], List[int]]:
        if count <= 0:
            return [], list(pool)
        if count > len(pool):
            raise ValueError("Cannot select more indices than available.")

        n = len(pool)
        positions = {((2 * j + 1) * n) // (2 * count) for j in range(count)}
        selected = [pool[p] for p in sorted(positions)]
        remainder = [
            value for position, value in enumerate(pool)
            if position not in positions
        ]
        return selected, remainder

    def split(
        self, labels: torch.Tensor
    ) -> Tuple[List[int], List[int], List[int]]:
        groups: Dict[int, List[int]] = {}
        for index, label in enumerate(labels.tolist()):
            groups.setdefault(int(label), []).append(index)

        train_idx: List[int] = []
        val_idx: List[int] = []
        test_idx: List[int] = []

        for label in sorted(groups):
            pool = groups[label]
            n = len(pool)
            n_train = max(1, math.floor(self.train_fraction * n))
            n_val = max(1, math.floor(self.val_fraction * n))

            if n_train + n_val >= n:
                raise ValueError(
                    f"Class {label} has only {n} samples; the 10/10/80 "
                    f"protocol requires at least {n_train + n_val + 1}."
                )

            taken_train, remainder = self._systematic_take(pool, n_train)
            taken_val, taken_test = self._systematic_take(remainder, n_val)

            train_idx.extend(taken_train)
            val_idx.extend(taken_val)
            test_idx.extend(taken_test)

        return sorted(train_idx), sorted(val_idx), sorted(test_idx)


# ===========================================================================
# 4. Safe multi-view data loading
# ===========================================================================

class MultiViewLoaderFactory:
    """
    Builds DataLoaders from an adapter and an explicit index list.

    Batch size is clamped to the split size, which is essential for the
    10% training split of 100leaves (160 samples with B=256 would otherwise
    produce zero batches when drop_last=True).
    """

    def __init__(
        self,
        adapter: MultiViewDatasetAdapter,
        batch_size: int,
        num_workers: int,
        seed: int,
        pin_memory: bool,
    ) -> None:
        if batch_size < 2:
            raise ValueError("batch_size must be >= 2 for BatchNorm training.")
        self.adapter = adapter
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.seed = int(seed)
        self.pin_memory = bool(pin_memory)

    def make(
        self, indices: Sequence[int], shuffle: bool, drop_last: bool
    ) -> DataLoader:
        if len(indices) == 0:
            raise ValueError("Cannot build a DataLoader from an empty split.")

        subset = Subset(self.adapter, list(indices))
        batch_size = min(self.batch_size, len(subset))

        if drop_last and len(subset) < 2:
            raise ValueError("Training split must contain at least two samples.")

        generator = torch.Generator().manual_seed(self.seed)

        return DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
            generator=generator,
            persistent_workers=False,
        )


# ===========================================================================
# 5. Experiment configuration
# ===========================================================================

@dataclass
class ExperimentConfig:
    dataset: str
    data_root: str = "/home/phd/datasets"
    output_root: str = "outputs"
    log_root: str = "logs"

    epochs: int = 200
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    num_workers: int = 4

    lambda_pred: float = 1.0
    lambda_noise: float = 0.1
    lambda_syn: float = 0.1
    lambda_id: float = 1.0

    num_coupling_layers: int = 6
    accelerator: str = "auto"
    devices: int = 1
    missing_index: int = 1
    all_single_drops: bool = False

    eval_only: bool = False
    checkpoint: Optional[str] = None

    @classmethod
    def from_namespace(cls, namespace: argparse.Namespace) -> "ExperimentConfig":
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in vars(namespace).items()
                      if key in allowed})


class CommandLineInterface:
    """Parses the runner arguments and dispatches the requested mode."""

    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(
            description="PID-IMVF: original-data, object-oriented training runner."
        )
        self._add_arguments()

    def _add_arguments(self) -> None:
        p = self.parser
        p.add_argument("--dataset", type=str, default=None,
                       choices=DatasetRegistry.names())
        p.add_argument("--data-root", type=str, default="/home/phd/datasets")
        p.add_argument("--output-root", type=str, default="outputs")
        p.add_argument("--log-root", type=str, default="logs")

        p.add_argument("--epochs", type=int, default=200)
        p.add_argument("--batch-size", type=int, default=256)
        p.add_argument("--lr", type=float, default=1e-3)
        p.add_argument("--weight-decay", type=float, default=1e-4)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--num-workers", type=int, default=4)

        p.add_argument("--lambda-pred", type=float, default=1.0)
        p.add_argument("--lambda-noise", type=float, default=0.1)
        p.add_argument("--lambda-syn", type=float, default=0.1)
        p.add_argument("--lambda-id", type=float, default=1.0)

        p.add_argument("--num-coupling-layers", type=int, default=6)
        p.add_argument("--accelerator", type=str, default="auto")
        p.add_argument("--devices", type=int, default=1)
        p.add_argument("--missing-index", type=int, default=1)
        p.add_argument("--all-single-drops", action="store_true")

        p.add_argument("--eval-only", action="store_true")
        p.add_argument("--checkpoint", type=str, default=None)

        p.add_argument("--aggregate", action="store_true",
                       help="Aggregate existing results.json files into a table.")
        p.add_argument("--list-datasets", action="store_true")

    def run(self) -> int:
        args = self.parser.parse_args()

        if args.list_datasets:
            print("\n".join(DatasetRegistry.names()))
            return 0

        if args.aggregate:
            ResultsAggregator(args.output_root).print_tables()
            return 0

        if args.dataset is None:
            self.parser.error("--dataset is required unless --aggregate is used.")

        config = ExperimentConfig.from_namespace(args)
        TrainingExperiment(config).run()
        return 0


# ===========================================================================
# 6. Missing-view robustness evaluation (Theorem 3)
# ===========================================================================

class MissingViewRobustnessEvaluator:
    """
    Evaluates h_phi(S_hat^(obs)) with deterministic view removal.

        S_hat^(obs) = (1 / |V_obs|) sum_{v in V_obs} S_hat_v

    Only the encoders of OBSERVED views are executed. The unique factors
    U_hat_v are never passed to the classifier.
    """

    def __init__(
        self,
        model: PID_IMVF,
        adapter: MultiViewDatasetAdapter,
        test_indices: Sequence[int],
        loader_factory: MultiViewLoaderFactory,
        num_classes: int,
        device: torch.device,
    ) -> None:
        self.model = model
        self.adapter = adapter
        self.test_indices = list(test_indices)
        self.loader_factory = loader_factory
        self.device = device
        self.metrics = MetricsCalculator(num_classes=num_classes)
        self.simulator = MissingViewSimulator(num_views=adapter.num_views)

    @torch.no_grad()
    def _predict(
        self, loader: DataLoader, observed_views: Sequence[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        all_targets: List[torch.Tensor] = []
        all_predictions: List[torch.Tensor] = []

        for views, targets in loader:
            views = [view.to(self.device) for view in views]
            targets = targets.to(self.device)

            shared_reps: List[torch.Tensor] = []
            for view_index in observed_views:
                shared, _, _ = self.model.encoders[int(view_index)](
                    views[int(view_index)]
                )
                shared_reps.append(shared)

            partial_consensus = self.simulator.compute_partial_consensus(
                shared_reps, list(range(len(shared_reps)))
            )
            logits = self.model.classifier(partial_consensus)
            predictions = torch.argmax(logits, dim=-1)

            all_targets.append(targets.detach().cpu())
            all_predictions.append(predictions.detach().cpu())

        return torch.cat(all_targets, dim=0), torch.cat(all_predictions, dim=0)

    def run(
        self, missing_index: int = 1, all_single_drops: bool = False
    ) -> Dict[str, Dict[str, float]]:
        num_views = self.adapter.num_views
        if not 0 <= missing_index < num_views:
            raise ValueError(
                f"missing_index must lie in [0, {num_views - 1}]."
            )

        loader = self.loader_factory.make(
            self.test_indices, shuffle=False, drop_last=False
        )

        scenarios: Dict[str, List[int]] = {
            "full_view": list(range(num_views)),
            f"drop_view_{missing_index + 1}": [
                v for v in range(num_views) if v != missing_index
            ],
        }

        if all_single_drops:
            for v in range(num_views):
                key = f"drop_view_{v + 1}"
                if key not in scenarios:
                    scenarios[key] = [u for u in range(num_views) if u != v]

        results: Dict[str, Dict[str, float]] = {}
        for name, observed in scenarios.items():
            if not observed:
                raise ValueError(
                    "Cannot evaluate a scenario with zero observed views."
                )
            targets, predictions = self._predict(loader, observed)
            results[name] = self.metrics.compute_all_metrics(targets, predictions)

        reference = f"drop_view_{missing_index + 1}"
        results["degradation"] = {
            metric: (
                results["full_view"][metric] - results[reference][metric]
            ) * 100.0
            for metric in ("ACC", "Macro-F1", "BACC")
        }
        return results


# ===========================================================================
# 7. End-to-end experiment orchestration
# ===========================================================================

class TrainingExperiment:
    """
    Orchestrates: original data -> adapter -> deterministic split ->
    Factorized INN -> consensus -> PID estimator -> classifier -> metrics.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.spec = DatasetRegistry.get(config.dataset)

        self.output_dir = Path(config.output_root) / config.dataset
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if config.accelerator == "cpu" or not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda", 0)

        self.adapter: Optional[MultiViewDatasetAdapter] = None
        self.splitter_result: Optional[Tuple[List[int], List[int], List[int]]] = None
        self.loader_factory: Optional[MultiViewLoaderFactory] = None

    # ------------------------------------------------------------------
    # 7.1 Data preparation
    # ------------------------------------------------------------------
    def _prepare_data(self) -> None:
        DatasetRegistry.preflight(self.spec, self.config.data_root)
        raw_dataset = DatasetRegistry.instantiate(self.spec, self.config.data_root)
        self.adapter = MultiViewDatasetAdapter(raw_dataset)

        if self.adapter.num_views != self.spec.expected_views:
            raise ValueError(
                f"[{self.spec.name}] Expected {self.spec.expected_views} views, "
                f"found {self.adapter.num_views} in the original data."
            )
        if self.adapter.mapper.num_classes != self.spec.expected_classes:
            raise ValueError(
                f"[{self.spec.name}] Expected {self.spec.expected_classes} "
                f"classes, found {self.adapter.mapper.num_classes}."
            )

        latent_dim = self.spec.d_S + self.spec.d_U
        smallest_view = min(self.adapter.view_dims)
        if latent_dim > smallest_view:
            raise ValueError(
                f"Assumption (A3) violated: d_S + d_U = {latent_dim} exceeds "
                f"the smallest original view dimension {smallest_view}. "
                f"Reduce d_S/d_U for dataset '{self.spec.name}'."
            )
        if self.adapter.mapper.num_classes < self.spec.d_S + 1:
            raise ValueError(
                "Assumption (A2) violated: |supp(Y)| must be >= d_S + 1."
            )

        splitter = DeterministicStratifiedSplitter(0.10, 0.10)
        self.splitter_result = splitter.split(self.adapter.labels)

        self.loader_factory = MultiViewLoaderFactory(
            adapter=self.adapter,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            seed=self.config.seed,
            pin_memory=(self.device.type == "cuda"),
        )

    # ------------------------------------------------------------------
    # 7.2 Model construction and training
    # ------------------------------------------------------------------
    def _build_model(self) -> PID_IMVF:
        assert self.adapter is not None
        return PID_IMVF(
            view_dims=list(self.adapter.view_dims),
            d_S=self.spec.d_S,
            d_U=self.spec.d_U,
            num_classes=self.adapter.mapper.num_classes,
            num_coupling_layers=self.config.num_coupling_layers,
            lambda_pred=self.config.lambda_pred,
            lambda_noise=self.config.lambda_noise,
            lambda_syn=self.config.lambda_syn,
            lambda_id=self.config.lambda_id,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

    def _train(self, model: PID_IMVF) -> str:
        assert self.loader_factory is not None
        train_idx, val_idx, _ = self.splitter_result

        train_loader = self.loader_factory.make(
            train_idx, shuffle=True, drop_last=True
        )
        val_loader = self.loader_factory.make(
            val_idx, shuffle=False, drop_last=False
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath=str(self.checkpoint_dir),
            filename="pid-imvf-epoch{epoch:03d}",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        )
        early_stopping = EarlyStopping(
            monitor="val/loss", mode="min", patience=30, min_delta=1e-4
        )

        lightning_logger = CSVLogger(
            save_dir=str(self.output_dir), name="lightning_csv"
        )

        trainer = pl.Trainer(
            max_epochs=self.config.epochs,
            accelerator=self.config.accelerator,
            devices=self.config.devices,
            precision="32-true",
            deterministic=True,
            benchmark=False,
            num_sanity_val_steps=0,
            default_root_dir=str(self.output_dir),
            callbacks=[checkpoint_callback, early_stopping],
            logger=lightning_logger,
            enable_model_summary=False,
        )

        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        best_path = checkpoint_callback.best_model_path
        last_path = checkpoint_callback.last_model_path
        if not best_path and not last_path:
            raise RuntimeError("Training produced no checkpoint.")
        return best_path or last_path

    def _load_model(self, checkpoint_path: str) -> PID_IMVF:
        if not Path(checkpoint_path).is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        model = PID_IMVF.load_from_checkpoint(
            checkpoint_path, map_location=self.device
        )
        model.to(self.device)
        model.eval()
        return model

    # ------------------------------------------------------------------
    # 7.3 Diagnostics on real batches
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _diagnose(self, model: PID_IMVF) -> Dict[str, float]:
        assert self.loader_factory is not None
        _, _, test_idx = self.splitter_result
        loader = self.loader_factory.make(test_idx, shuffle=False, drop_last=False)
        views, targets = next(iter(loader))
        views = [view.to(self.device) for view in views]
        targets = targets.to(self.device)

        model.eval()
        outputs = model(views, targets)
        atoms = outputs["pid_atoms"]
        losses = model._compute_losses(outputs, targets)

        return {
            "redundancy_nats": float(atoms.redundancy.item()),
            "uniqueness_nats": float(atoms.uniqueness.sum().item()),
            "synergy_nats": float(atoms.synergy.item()),
            "joint_mi_nats": float(atoms.total_joint_mi.item()),
            "loss_ce": float(losses["loss_ce"].item()),
            "loss_pid": float(losses["loss_pid"].item()),
            "loss_vib": float(losses["loss_vib"].item()),
            "loss_syn": float(losses["loss_syn"].item()),
            "loss_id": float(losses["loss_id"].item()),
        }

    # ------------------------------------------------------------------
    # 7.4 Public entry point
    # ------------------------------------------------------------------
    def run(self) -> Dict[str, object]:
        started = time.time()

        pl.seed_everything(self.config.seed, workers=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self._prepare_data()
        assert self.adapter is not None and self.loader_factory is not None

        train_idx, val_idx, test_idx = self.splitter_result
        num_classes = self.adapter.mapper.num_classes

        pid_logger = PID_IMVF_Logger(
            log_dir=self.config.log_root,
            experiment_name=self.config.dataset,
        )

        if self.config.eval_only:
            if not self.config.checkpoint:
                raise ValueError("--eval-only requires --checkpoint.")
            checkpoint_path = self.config.checkpoint
        else:
            model = self._build_model()
            checkpoint_path = self._train(model)

        model = self._load_model(checkpoint_path)

        test_loader = self.loader_factory.make(
            test_idx, shuffle=False, drop_last=False
        )

        full_metrics = MetricsCalculator.evaluate_dataloader(
            model=model,
            dataloader=test_loader,
            num_classes=num_classes,
            device=str(self.device),
        )

        robustness = MissingViewRobustnessEvaluator(
            model=model,
            adapter=self.adapter,
            test_indices=test_idx,
            loader_factory=self.loader_factory,
            num_classes=num_classes,
            device=self.device,
        ).run(
            missing_index=self.config.missing_index,
            all_single_drops=self.config.all_single_drops,
        )

        diagnostics = self._diagnose(model)

        # Report percentages, matching the manuscript tables.
        full_percent = {k: v * 100.0 for k, v in full_metrics.items()}
        missing_percent = {
            name: {k: v * 100.0 for k, v in values.items()}
            for name, values in robustness.items()
        }

        results: Dict[str, object] = {
            "dataset": self.config.dataset,
            "num_classes": num_classes,
            "num_views": self.adapter.num_views,
            "view_dims": list(self.adapter.view_dims),
            "d_S": self.spec.d_S,
            "d_U": self.spec.d_U,
            "split_sizes": {
                "train": len(train_idx),
                "val": len(val_idx),
                "test": len(test_idx),
            },
            "seed": self.config.seed,
            "config": asdict(self.config),
            "full_view": full_percent,
            "missing_view": missing_percent,
            "pid_diagnostics": diagnostics,
            "checkpoint": str(checkpoint_path),
            "elapsed_seconds": float(time.time() - started),
        }

        pid_logger.log_pid_atoms(
            epoch=self.config.epochs,
            step=0,
            pid_atoms={
                "redundancy": torch.tensor(diagnostics["redundancy_nats"]),
                "uniqueness": torch.tensor([diagnostics["uniqueness_nats"]]),
                "synergy": torch.tensor(diagnostics["synergy_nats"]),
                "total_joint_mi": torch.tensor(diagnostics["joint_mi_nats"]),
            },
        )
        pid_logger.log_epoch_summary(epoch=self.config.epochs, metrics=full_percent)
        pid_logger.close()

        results_path = self.output_dir / "results.json"
        with results_path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, ensure_ascii=False)

        print(f"\n[{self.config.dataset}] original samples: {len(self.adapter)}")
        print(f"  view dimensions : {self.adapter.view_dims}")
        print(f"  classes         : {num_classes}")
        print(f"  split           : {results['split_sizes']}")
        print(f"  ACC / Macro-F1 / BACC (%): "
              f"{full_percent['ACC']:.2f} / "
              f"{full_percent['Macro-F1']:.2f} / "
              f"{full_percent['BACC']:.2f}")
        print(f"  results written : {results_path}")

        return results


# ===========================================================================
# 8. Result aggregation for the manuscript tables
# ===========================================================================

class ResultsAggregator:
    """Reads the real results.json files and prints a LaTeX comparison table."""

    def __init__(self, output_root: str) -> None:
        self.output_root = Path(output_root)

    def collect(self) -> List[Dict[str, object]]:
        records: List[Dict[str, object]] = []
        for path in sorted(self.output_root.glob("*/results.json")):
            with path.open("r", encoding="utf-8") as handle:
                records.append(json.load(handle))
        return records

    def print_tables(self) -> None:
        records = self.collect()
        if not records:
            print("No results.json files found. Run train_all.sh first.")
            return

        print(r"\begin{tabular}{lcccccc}")
        print(r"\toprule")
        print(r"\textbf{Dataset} & \textbf{ACC} & \textbf{Macro-F1} "
              r"& \textbf{BACC} & \textbf{Missing F1} & "
              r"\textbf{$\Delta$F1} & \textbf{Syn} \\")
        print(r"\midrule")

        for record in records:
            full = record["full_view"]
            missing_container = record["missing_view"]
            reference_key = next(
                (key for key in missing_container if key.startswith("drop_view_")),
                None,
            )
            missing_f1 = (
                missing_container[reference_key]["Macro-F1"]
                if reference_key else float("nan")
            )
            delta = missing_container.get("degradation", {}).get(
                "Macro-F1", float("nan")
            )
            synergy = record.get("pid_diagnostics", {}).get(
                "synergy_nats", float("nan")
            )

            print(
                f"{record['dataset']} & "
                f"{full['ACC']:.2f} & "
                f"{full['Macro-F1']:.2f} & "
                f"{full['BACC']:.2f} & "
                f"{missing_f1:.2f} & "
                f"{delta:.2f} & "
                f"{synergy:.3f} \\\\"
            )

        print(r"\bottomrule")
        print(r"\end{tabular}")


# ===========================================================================
# 9. Entry point
# ===========================================================================

def main() -> int:
    return CommandLineInterface().run()


if __name__ == "__main__":
    raise SystemExit(main())