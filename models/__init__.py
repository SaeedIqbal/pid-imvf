"""
Core neural architectures for PID-IMVF.

Modules
-------
factorized_inn :
    Submanifold-restricted invertible encoder g_v, producing the
    block-diagonal factorization (S_hat_v, U_hat_v).
consensus_module :
    Euclidean Frechet aggregation, shared-gauge alignment penalty,
    and partial-consensus inference under missing views.
pid_estimator :
    Differentiable Neural PID Lattice Estimator using Barber-Agakov 
    classifier-based MI bounds to compute Redundancy, Uniqueness, and Synergy.
classifier :
    Downstream classification head strictly restricted to S_hat^(obs).
pid_imvf :
    Main PyTorch Lightning Module orchestrating the entire PID-IMVF pipeline.
"""

from .factorized_inn import (
    AffineCouplingLayer,
    FactorizedINN,
    SubmanifoldProjector,
)

from .consensus_module import (
    ConsensusModule,
    ConsensusOutput,
    squared_distance,
    squared_norm,
)

from .pid_estimator import (
    NeuralPID_Estimator,
    ClassifierHead as PIDClassifierHead,
    PIDAtoms,
)

from .classifier import ConsensusClassifier

from .pid_imvf import PID_IMVF, compute_hsic

__all__ = [
    "AffineCouplingLayer",
    "FactorizedINN",
    "SubmanifoldProjector",
    "ConsensusModule",
    "ConsensusOutput",
    "squared_distance",
    "squared_norm",
    "NeuralPID_Estimator",
    "PIDClassifierHead",
    "PIDAtoms",
    "ConsensusClassifier",
    "PID_IMVF",
    "compute_hsic",
]