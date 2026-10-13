"""
Statistical Significance Testing for PID-IMVF.

Mathematical background
-----------------------
1. Paired Wilcoxon Signed-Rank Test:
   Given paired samples X and Y (e.g., Macro-F1 scores of PID-IMVF and a baseline 
   across R seeds), compute differences D_i = X_i - Y_i.
   Remove zero differences. Let N be the number of non-zero differences.
   Rank the absolute differences |D_i| from 1 to N.
   W+ = sum of ranks where D_i > 0.
   Under the null hypothesis, the expected value is mu_W = N(N+1)/4.
   The variance is sigma_W^2 = N(N+1)(2N+1)/24 - sum(t_j^3 - t_j)/48, 
   where t_j is the number of ties in the j-th group.
   For large N, Z = (W+ - mu_W) / sigma_W follows a standard normal distribution.
   p-value = 2 * (1 - Phi(|Z|)), where Phi is the standard normal CDF.
   
2. Bonferroni Correction:
   To control the family-wise error rate across m hypothesis tests, 
   the adjusted significance level is alpha_adj = alpha / m.
   Alternatively, the corrected p-value is p_corrected = min(p * m, 1.0).
"""

from __future__ import annotations

import torch
import numpy as np
from typing import List, Dict, Tuple, Union


__all__ = [
    "StatisticalSignificanceTester",
]


class StatisticalSignificanceTester:
    """
    Computes statistical significance using the Paired Wilcoxon Signed-Rank Test 
    and applies the Bonferroni correction for multiple comparisons.
    """
    def __init__(self, alpha: float = 0.05):
        """
        Parameters
        ----------
        alpha : float, default=0.05
            The family-wise significance level.
        """
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
        self.alpha = float(alpha)
        
    def _normal_cdf(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the cumulative distribution function (CDF) of the standard normal distribution.
        
        Mathematical formulation:
        $$ \\Phi(x) = \\frac{1}{2} \\left[ 1 + \\text{erf}\\left( \\frac{x}{\\sqrt{2}} \\right) \\right] $$
        """
        return 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))

    def _rank_data(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ranks the elements of a 1D tensor. Ties are assigned the average rank.
        """
        unique_vals, inverse_indices, counts = torch.unique(x, return_inverse=True, return_counts=True)
        cum_counts = torch.cumsum(counts, dim=0)
        start_ranks = cum_counts - counts + 1
        end_ranks = cum_counts
        avg_ranks = (start_ranks + end_ranks) / 2.0
        ranks = avg_ranks[inverse_indices]
        return ranks

    def wilcoxon_signed_rank_test(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
        """
        Performs the paired Wilcoxon signed-rank test.
        
        Parameters
        ----------
        x : torch.Tensor, shape (N,)
            Performance scores of the proposed method (e.g., PID-IMVF) across N seeds/folds.
        y : torch.Tensor, shape (N,)
            Performance scores of the baseline method across N seeds/folds.
            
        Returns
        -------
        Tuple[float, float]
            (W_plus statistic, two-sided p-value).
        """
        if x.dim() != 1 or y.dim() != 1:
            raise ValueError("Inputs must be 1D tensors.")
        if x.size(0) != y.size(0):
            raise ValueError("Inputs must have the same length.")
            
        d = x - y
        
        # Remove zero differences
        non_zero_mask = d != 0
        d = d[non_zero_mask]
        N = d.numel()
        
        if N == 0:
            return 0.0, 1.0
            
        abs_d = torch.abs(d)
        ranks = self._rank_data(abs_d)
        
        # Sum of ranks for positive differences
        W_plus = torch.sum(ranks[d > 0])
        
        # Mean and variance of W+ under null hypothesis
        mu_W = N * (N + 1) / 4.0
        
        # Variance with tie correction
        unique_vals, counts = torch.unique(abs_d, return_counts=True)
        tie_correction = torch.sum(counts**3 - counts) / 48.0
        sigma_W_sq = N * (N + 1) * (2 * N + 1) / 24.0 - tie_correction
        
        sigma_W = torch.sqrt(torch.clamp(sigma_W_sq, min=1e-8))
        
        # Normal approximation (two-sided)
        Z = (W_plus - mu_W) / sigma_W
        p_value = 2.0 * (1.0 - self._normal_cdf(torch.abs(Z)))
        
        return W_plus.item(), p_value.item()

    def bonferroni_correction(self, p_values: List[float]) -> Tuple[List[float], float]:
        """
        Applies the Bonferroni correction to a list of p-values.
        
        Mathematical formulation:
        $$ p_{\\text{corrected}} = \\min(p_i \\times m, 1.0) $$
        $$ \\alpha_{\\text{adj}} = \\frac{\\alpha}{m} $$
        where m is the number of hypothesis tests.
        
        Parameters
        ----------
        p_values : List[float]
            List of uncorrected p-values.
            
        Returns
        -------
        Tuple[List[float], float]
            (List of corrected p-values, adjusted significance level alpha_adj).
        """
        m = len(p_values)
        if m == 0:
            return [], self.alpha
            
        corrected_p_values = [min(p * m, 1.0) for p in p_values]
        adjusted_alpha = self.alpha / m
        
        return corrected_p_values, adjusted_alpha

    def evaluate_statistical_significance(self, proposed_scores: Dict[str, torch.Tensor], 
                                          baseline_scores: Dict[str, torch.Tensor], 
                                          dataset_names: List[str]) -> Dict[str, Dict[str, Union[float, bool]]]:
        """
        Evaluates the statistical significance of the proposed method against multiple baselines 
        across multiple datasets, applying the Bonferroni correction.
        
        Parameters
        ----------
        proposed_scores : Dict[str, torch.Tensor]
            Dictionary mapping dataset name to a 1D tensor of performance scores (e.g., Macro-F1) 
            for the proposed method across R seeds.
        baseline_scores : Dict[str, torch.Tensor]
            Dictionary mapping baseline name to a dictionary mapping dataset name to a 1D tensor 
            of performance scores across R seeds.
        dataset_names : List[str]
            List of dataset names to evaluate.
            
        Returns
        -------
        Dict[str, Dict[str, Union[float, bool]]]
            Nested dictionary: {baseline_name: {dataset_name: {'p_value': float, 'p_corrected': float, 'is_significant': bool}}}
        """
        m = len(baseline_scores) * len(dataset_names)
        all_p_values = []
        
        # First pass: compute all uncorrected p-values
        results = {}
        for baseline_name, baseline_datasets in baseline_scores.items():
            results[baseline_name] = {}
            for ds_name in dataset_names:
                if ds_name not in proposed_scores or ds_name not in baseline_datasets:
                    raise ValueError(f"Missing scores for dataset {ds_name}.")
                    
                prop_scores = proposed_scores[ds_name]
                base_scores = baseline_datasets[ds_name]
                
                _, p_val = self.wilcoxon_signed_rank_test(prop_scores, base_scores)
                all_p_values.append(p_val)
                results[baseline_name][ds_name] = {'p_value': p_val}
                
        # Second pass: apply Bonferroni correction
        corrected_p_values, adjusted_alpha = self.bonferroni_correction(all_p_values)
        
        idx = 0
        for baseline_name in baseline_scores.keys():
            for ds_name in dataset_names:
                p_corr = corrected_p_values[idx]
                is_sig = p_corr < self.alpha
                results[baseline_name][ds_name]['p_corrected'] = p_corr
                results[baseline_name][ds_name]['is_significant'] = is_sig
                idx += 1
                
        return results