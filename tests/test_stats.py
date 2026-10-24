"""
Statistical function unit tests.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sigma_qgd.analysis.stats import (
    wilcoxon_pairwise, holm_bonferroni_correction,
    fdr_bh_correction, effect_size_summary, momentum_alignment_rate,
)


def test_holm_basic():
    raw       = [0.01, 0.04, 0.10, 0.20, 0.35]
    corrected = holm_bonferroni_correction(raw)
    assert all(c >= r for c, r in zip(corrected, raw)), \
        "Holm must be conservative (corrected >= raw)"
    assert all(0.0 <= c <= 1.0 for c in corrected), \
        "All corrected p-values must be in [0, 1]"
    print("  PASS: test_holm_basic")


def test_holm_single():
    assert holm_bonferroni_correction([0.05]) == [0.05]
    print("  PASS: test_holm_single")


def test_holm_empty():
    assert holm_bonferroni_correction([]) == []
    print("  PASS: test_holm_empty")


def test_fdr_basic():
    raw       = [0.01, 0.04, 0.10]
    corrected = fdr_bh_correction(raw)
    assert all(0.0 <= c <= 1.0 for c in corrected)
    print("  PASS: test_fdr_basic")


def test_wilcoxon_too_small():
    result = wilcoxon_pairwise([0.1, 0.2], [0.3, 0.4])
    assert result is None, "Should return None for n<3"
    print("  PASS: test_wilcoxon_too_small")


def test_wilcoxon_clearly_significant():
    """Clearly different distributions should yield small p."""
    sigma_gaps = [0.001] * 20
    adam_gaps  = [0.5]   * 20
    result = wilcoxon_pairwise(sigma_gaps, adam_gaps, "Adam")
    assert result is not None
    assert result["p_value"] < 0.01, \
        f"Expected p < 0.01, got {result['p_value']}"
    assert result["direction"] == "SIGMA better"
    print(f"  PASS: test_wilcoxon_clearly_significant (p={result['p_value']:.6f})")


def test_effect_size_direction():
    sigma_gaps = [0.01] * 10
    base_gaps  = [0.50] * 10
    result = effect_size_summary(sigma_gaps, base_gaps)
    assert result["mean_diff"] < 0, "mean_diff should be negative (SIGMA better)"
    assert result["cohens_d"] is not None
    print("  PASS: test_effect_size_direction")


def test_momentum_alignment_rate():
    import numpy as np
    # Fully aligned: sign(g) == sign(m) always
    grads    = [np.array([1.0, -1.0])] * 10
    momenta  = [np.array([1.0, -1.0])] * 10
    rate     = momentum_alignment_rate(None, grads, momenta)
    assert abs(rate - 1.0) < 1e-9, f"Expected 1.0, got {rate}"
    # Fully anti-aligned
    momenta2 = [np.array([-1.0, 1.0])] * 10
    rate2    = momentum_alignment_rate(None, grads, momenta2)
    assert abs(rate2 - 0.0) < 1e-9, f"Expected 0.0, got {rate2}"
    print("  PASS: test_momentum_alignment_rate")


if __name__ == "__main__":
    test_holm_basic()
    test_holm_single()
    test_holm_empty()
    test_fdr_basic()
    test_wilcoxon_too_small()
    test_wilcoxon_clearly_significant()
    test_effect_size_direction()
    test_momentum_alignment_rate()
    print("\n  All stats tests passed.")
