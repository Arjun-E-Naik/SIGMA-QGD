"""
Statistical analysis routines for benchmark comparisons.
"""

from __future__ import annotations

from typing import List, Optional, Sequence
import numpy as np
from scipy import stats as scipy_stats


def wilcoxon_pairwise(
    sigma_gaps: List[float],
    baseline_gaps: List[float],
    baseline_name: str = "baseline",
) -> Optional[dict]:
    """
    Paired Wilcoxon signed-rank test comparing two sets of energy gaps.

    Returns dict containing test statistic, p-value, mean difference, sample size,
    and relative direction. Returns None if n < 3.
    """
    n_m = min(len(sigma_gaps), len(baseline_gaps))
    if n_m < 3:
        return None
    s_gaps = np.array(sigma_gaps[:n_m])
    b_gaps = np.array(baseline_gaps[:n_m])
    diffs = s_gaps - b_gaps
    try:
        w_res = scipy_stats.wilcoxon(diffs, alternative="two-sided")
        stat = float(np.asarray(w_res[0]))
        pval = float(np.asarray(w_res[1]))
    except ValueError:
        stat, pval = 0.0, 1.0
    mean_diff = float(np.mean(diffs))
    direction = "SIGMA better" if mean_diff < 0 else "Baseline better"
    return dict(
        baseline=baseline_name,
        statistic=round(float(stat), 4),
        p_value=round(float(pval), 6),
        mean_diff=round(mean_diff, 6),
        n=n_m,
        direction=direction,
    )


def holm_bonferroni_correction(p_values: List[float]) -> List[float]:
    """
    Holm-Bonferroni step-down correction for multiple testing.

    Parameters
    ----------
    p_values : list of float
        Raw unadjusted p-values.

    Returns
    -------
    corrected : list of float
        Adjusted p-values preserving original input ordering.
    """
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    corrected: List[float] = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, pval) in enumerate(indexed):
        adjusted = min((m - rank) * pval, 1.0)
        adjusted = max(adjusted, running_max)
        running_max = adjusted
        corrected[orig_idx] = round(adjusted, 8)
    return corrected


def fdr_bh_correction(p_values: List[float]) -> List[float]:
    """
    Benjamini-Hochberg False Discovery Rate (FDR) correction.
    """
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted: List[float] = [0.0] * m
    running_min = 1.0
    for rank in range(m - 1, -1, -1):
        orig_idx, pval = indexed[rank]
        adj = min(pval * m / (rank + 1), 1.0)
        adj = min(adj, running_min)
        running_min = adj
        adjusted[orig_idx] = round(adj, 8)
    return adjusted


def effect_size_summary(
    sigma_gaps: List[float],
    baseline_gaps: List[float],
    baseline_name: str = "baseline",
) -> dict:
    """
    Computes Cohen's d effect size and 95% confidence interval on the mean gap difference.
    """
    s = np.array(sigma_gaps)
    b = np.array(baseline_gaps)
    n_m = min(len(s), len(b))
    diffs = s[:n_m] - b[:n_m]
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1)) if n_m > 1 else float("nan")
    cohens_d = mean_diff / std_diff if std_diff > 0 else float("nan")
    sem = std_diff / np.sqrt(n_m) if n_m > 1 else float("nan")
    t_crit = float(scipy_stats.t.ppf(0.975, df=n_m - 1)) if n_m > 1 else float("nan")
    ci_lo = mean_diff - t_crit * sem
    ci_hi = mean_diff + t_crit * sem
    return dict(
        baseline=baseline_name,
        mean_diff=round(mean_diff, 6),
        cohens_d=round(cohens_d, 4) if cohens_d == cohens_d else None,
        ci_95_lo=round(ci_lo, 6) if ci_lo == ci_lo else None,
        ci_95_hi=round(ci_hi, 6) if ci_hi == ci_hi else None,
        n=n_m,
    )


def momentum_alignment_rate(
    energy_hist: Optional[Sequence[float]],
    grad_hist: Sequence[np.ndarray | Sequence[float]],
    momentum_hist: Sequence[np.ndarray | Sequence[float]],
) -> float:
    """
    Fraction of coordinate steps where sign(gradient) == sign(momentum).
    """
    total = 0
    aligned = 0
    for g, m in zip(grad_hist, momentum_hist):
        g = np.asarray(g)
        m = np.asarray(m)
        mask = (g != 0) & (m != 0)
        total += int(np.sum(mask))
        aligned += int(np.sum(np.sign(g[mask]) == np.sign(m[mask])))
    return aligned / total if total > 0 else float("nan")


def correlation_summary(x: np.ndarray, y: np.ndarray) -> dict:
    """
    Pearson (log-space) and Spearman rank correlations.
    """
    eps = 1e-30
    lx = np.log(np.asarray(x) + eps)
    ly = np.log(np.asarray(y) + eps)
    try:
        p_res = scipy_stats.pearsonr(lx, ly)
        pearson_r = float(np.asarray(p_res[0]))
    except Exception:
        pearson_r = float("nan")
    try:
        s_res = scipy_stats.spearmanr(x, y)
        spearman_r = float(np.asarray(s_res[0]))
    except Exception:
        spearman_r = float("nan")
    return dict(
        pearson_log=round(float(pearson_r), 6),
        spearman=round(float(spearman_r), 6),
    )
