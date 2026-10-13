"""
Utility functions and mathematical helpers for PID-IMVF.

Modules
-------
metrics :
    Rigorous, from-scratch implementations of ACC, Macro-F1, and BACC.
hsic :
    Hilbert-Schmidt Independence Criterion (HSIC) calculator using 
    Gaussian RBF kernels with median heuristic for scale-invariance.
pid_math :
    Williams-Beer PID mathematical logic, including Barber-Agakov MI bounds, 
    redundancy, uniqueness, synergy, and pairwise aggregation for V > 2 views.
missing_view_sim :
    Deterministic missing-view simulation and partial consensus computation 
    to evaluate robustness under severe observation constraints.
stats :
    Paired Wilcoxon signed-rank test with normal approximation and 
    Bonferroni correction for multiple hypothesis testing.
"""

from .metrics import MetricsCalculator
from .hsic import HSICCalculator
from .pid_math import PIDMathCalculator
from .missing_view_sim import MissingViewSimulator, MissingViewEvaluator
from .stats import StatisticalSignificanceTester

__all__ = [
    "MetricsCalculator",
    "HSICCalculator",
    "PIDMathCalculator",
    "MissingViewSimulator",
    "MissingViewEvaluator",
    "StatisticalSignificanceTester",
]