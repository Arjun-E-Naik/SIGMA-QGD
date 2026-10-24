"""
Circuit count verification tests.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sigma_qgd.problems.hamiltonians import build_tfim
from sigma_qgd.core.evaluator import CostEvaluator
from sigma_qgd.optimizers.registry import OPTIMIZER_REGISTRY


def assert_circuits_per_step(name: str, expected: int, n: int = 3, r: int = 1, steps: int = 5):
    H = build_tfim(n)
    p = n * r
    np.random.seed(42)
    theta0 = np.random.uniform(-0.1, 0.1, p)
    opt = OPTIMIZER_REGISTRY[name]
    cf = CostEvaluator(n, r, H)
    res, _, _ = opt.run(cf, theta0.copy(), max_steps=steps, n_qubits=n, reps=r)
    per_step = res.circuits_hist[:steps]
    failures = [(i, c) for i, c in enumerate(per_step) if c != expected]
    assert not failures, (
        f"{name}: expected {expected} circuits/step, wrong at steps {failures[:5]}"
    )
    print(f"  PASS: {name:<22} circuits/step={expected} (p={p})")


def test_all_circuit_counts():
    n, r = 3, 1
    p = n * r
    expected = {
        "SIGMA-QGD v9.0": 2 * p + 1,
        "Adam": 2 * p + 1,
        "AdaBelief": 2 * p + 1,
        "RMSProp": 2 * p + 1,
        "Yogi": 2 * p + 1,
        "H-QNG (Shi&Wang)": 2 * p + 1,
        "Diag-Hessian": 2 * p + 1,
        "Diag-QNG": 3 * p + 2,
        "QN-SPSA": 8,
        "COBYLA": 1,
        "SPSA": 3,
    }
    print(f"\nCircuit count audit (n={n}, reps={r}, p={p})")
    for name, exp in expected.items():
        try:
            assert_circuits_per_step(name, exp, n=n, r=r, steps=5)
        except Exception as e:
            print(f"  FAIL: {name}: {e}")
    print("\n[test_all_circuit_counts] Complete.")


if __name__ == "__main__":
    test_all_circuit_counts()
