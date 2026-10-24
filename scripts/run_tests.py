"""
Run all unit tests for the modular SIGMA-QGD package.

Usage:
    python scripts/run_tests.py
    python scripts/run_tests.py --quick    (fast smoke tests only)
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from sigma_qgd.problems.hamiltonians import build_tfim
from sigma_qgd.core.evaluator import CostEvaluator, ShotNoiseCostEvaluator
from sigma_qgd.optimizers.registry import OPTIMIZER_REGISTRY, LR_SWEEP_METHODS, KNOWN_METHODS
from sigma_qgd.analysis.stats import (
    wilcoxon_pairwise, holm_bonferroni_correction, momentum_alignment_rate
)


def test_registry_completeness():
    """Every method name in style.BENCHMARK_METHODS must be in the registry."""
    from sigma_qgd.style import BENCHMARK_METHODS
    missing = [m for m in BENCHMARK_METHODS if m not in OPTIMIZER_REGISTRY]
    assert not missing, f"Missing from registry: {missing}"
    print("[test_registry_completeness] PASSED")


def test_lr_sweep_methods_have_supports_lr():
    """All LR_SWEEP_METHODS must have supports_lr=True."""
    for name in LR_SWEEP_METHODS:
        opt = OPTIMIZER_REGISTRY[name]
        assert opt.supports_lr, f"{name}.supports_lr should be True"
    print("[test_lr_sweep_methods_have_supports_lr] PASSED")


def test_cost_and_gradient_matches_separate():
    """cost_and_gradient must match separate __call__ + gradient()."""
    n, r = 2, 1
    H  = build_tfim(n)
    th = np.random.uniform(-0.5, 0.5, n * r)
    cf1 = CostEvaluator(n, r, H); c1 = cf1(th.copy()); g1 = cf1.gradient(th.copy())
    cf2 = CostEvaluator(n, r, H); c2, g2 = cf2.cost_and_gradient(th.copy())
    assert abs(c1 - c2) < 1e-6, f"Cost mismatch: {c1} vs {c2}"
    assert np.allclose(g1, g2, atol=1e-6), f"Gradient mismatch"
    print("[test_cost_and_gradient_matches_separate] PASSED")


def test_circuit_count_sigma(n=3, r=1, max_steps=5):
    """SIGMA-QGD must accumulate exactly (2p+1)*steps circuits."""
    from sigma_qgd.optimizers.sigma_qgd import SigmaQGD
    H = build_tfim(n); p = n * r
    theta0 = np.random.uniform(-0.1, 0.1, p)
    cf = CostEvaluator(n, r, H)
    result, _, _ = SigmaQGD().run(cf, theta0, max_steps=max_steps,
                                    n_qubits=n, reps=r)
    expected = (2 * p + 1) * result.n_steps
    assert result.n_circuits == expected, \
        f"SIGMA circuit count {result.n_circuits} != {expected}"
    assert all(c == 2 * p + 1 for c in result.circuits_hist), \
        f"Per-step circuit counts wrong: {result.circuits_hist}"
    print(f"[test_circuit_count_sigma] PASSED  (p={p}, steps={result.n_steps}, "
          f"circuits={result.n_circuits}, expected={expected})")


def test_circuit_count_hessian_methods(n=3, r=1, max_steps=5):
    """H-QNG and Diag-Hessian must use exactly 2p+1 circuits/step."""
    H = build_tfim(n); p = n * r
    theta0 = np.random.uniform(-0.1, 0.1, p)
    for name in ("H-QNG", "Diag-Hessian"):
        cf  = CostEvaluator(n, r, H)
        opt = OPTIMIZER_REGISTRY[name]
        res, _, _ = opt.run(cf, theta0.copy(), max_steps=max_steps)
        assert all(c == 2 * p + 1 for c in res.circuits_hist), \
            f"{name}: circuits/step wrong: {res.circuits_hist}"
        print(f"[test_circuit_count_hessian_methods:{name}] PASSED")


def test_diag_qng_circuit_count(n=3, r=1, max_steps=3):
    """Diag-QNG must use exactly 3p+2 circuits/step."""
    H = build_tfim(n); p = n * r
    theta0 = np.random.uniform(-0.1, 0.1, p)
    cf  = CostEvaluator(n, r, H)
    res, _, _ = OPTIMIZER_REGISTRY["Diag-QNG"].run(cf, theta0.copy(), max_steps=max_steps)
    expected_per_step = 3 * p + 2
    assert all(c == expected_per_step for c in res.circuits_hist), \
        f"Diag-QNG circuits/step wrong: {res.circuits_hist}, expected {expected_per_step}"
    print(f"[test_diag_qng_circuit_count] PASSED  (3p+2={expected_per_step})")


def test_shot_noise_actually_varies():
    """Shot-noise evaluator must produce stochastic results."""
    n, r = 3, 1
    H = build_tfim(n); theta = np.random.uniform(-0.5, 0.5, n * r)
    cf100  = ShotNoiseCostEvaluator(n, r, H, n_shots=100)
    vals   = [cf100(theta) for _ in range(5)]
    assert np.std(vals) > 0, "Shot noise evaluator returned deterministic results"
    print(f"[test_shot_noise_actually_varies] PASSED  std={np.std(vals):.4f}")


def test_holm_bonferroni_monotone():
    """Holm-corrected p-values must always be >= raw p-values."""
    raw = [0.01, 0.04, 0.10, 0.20, 0.35]
    corrected = holm_bonferroni_correction(raw)
    for r, c in zip(raw, corrected):
        assert c >= r, f"Corrected {c} < raw {r} — Holm must be conservative"
    print("[test_holm_bonferroni_monotone] PASSED")


def test_holm_bonferroni_example():
    """Verify against known Holm-Bonferroni example."""
    raw = [0.01, 0.04, 0.03, 0.20, 0.35]
    corrected = holm_bonferroni_correction(raw)
    # Smallest: 0.01 * 5 = 0.05; next: 0.03 * 4 = 0.12; etc.
    assert corrected[0] == 0.05, f"Expected 0.05, got {corrected[0]}"
    print("[test_holm_bonferroni_example] PASSED")


def test_wilcoxon_n_less_than_3():
    """wilcoxon_pairwise must return None when n < 3."""
    result = wilcoxon_pairwise([0.1, 0.2], [0.3, 0.4], "test")
    assert result is None, "Should return None for n<3"
    print("[test_wilcoxon_n_less_than_3] PASSED")


def test_registry_unknown_method():
    """get_optimizer must raise ValueError for unknown method names."""
    from sigma_qgd.optimizers.registry import get_optimizer
    try:
        get_optimizer("NotAMethod")
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass
    print("[test_registry_unknown_method] PASSED")


def test_sigma_improves_on_random(n=3, r=1, max_steps=30):
    """SIGMA-QGD must improve over random initialisation on 3-qubit TFIM."""
    H = build_tfim(n)
    theta0 = np.random.uniform(-np.pi / 8, np.pi / 8, n * r)
    cf = CostEvaluator(n, r, H)
    from sigma_qgd.optimizers.sigma_qgd import SigmaQGD
    result, _, _ = SigmaQGD().run(cf, theta0.copy(), max_steps=max_steps,
                                    n_qubits=n, reps=r)
    init_e = cf(theta0)
    assert result.energy_opt < init_e, \
        f"SIGMA did not improve: init={init_e:.4f} opt={result.energy_opt:.4f}"
    print(f"[test_sigma_improves_on_random] PASSED  "
          f"init={init_e:.4f} opt={result.energy_opt:.4f}")


ALL_TESTS = [
    test_registry_completeness,
    test_lr_sweep_methods_have_supports_lr,
    test_cost_and_gradient_matches_separate,
    test_circuit_count_sigma,
    test_circuit_count_hessian_methods,
    test_diag_qng_circuit_count,
    test_shot_noise_actually_varies,
    test_holm_bonferroni_monotone,
    test_holm_bonferroni_example,
    test_wilcoxon_n_less_than_3,
    test_registry_unknown_method,
    test_sigma_improves_on_random,
]

QUICK_TESTS = [
    test_registry_completeness,
    test_lr_sweep_methods_have_supports_lr,
    test_holm_bonferroni_monotone,
    test_holm_bonferroni_example,
    test_wilcoxon_n_less_than_3,
    test_registry_unknown_method,
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Run only fast (no-quantum) tests")
    args = parser.parse_args()

    tests = QUICK_TESTS if args.quick else ALL_TESTS
    passed = 0; failed = 0

    print(f"\n  Running {len(tests)} {'quick ' if args.quick else ''}tests...\n")
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"[FAILED] {test_fn.__name__}: {e}")

    print(f"\n  {passed}/{len(tests)} tests passed.", end=" ")
    if failed:
        print(f"{failed} FAILED."); sys.exit(1)
    else:
        print("All PASSED.")


if __name__ == "__main__":
    main()
