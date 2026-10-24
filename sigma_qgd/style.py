"""
Plot styling configurations and optimizer hyperparameter definitions.
"""

from __future__ import annotations

COLOURS = {
    "SIGMA-QGD v9.0": "#00C9A7",
    "Adam": "#E24B4A",
    "COBYLA": "#708090",
    "SPSA": "#8B4513",
    "Diag-QNG": "#6A0DAD",
    "QN-SPSA": "#FF6F00",
    "AdaBelief": "#1F77B4",
    "RMSProp": "#BCBD22",
    "Yogi": "#17BECF",
    "H-QNG (Shi&Wang)": "#D62728",
    "Hess-Mag": "#D62728",
    "Diag-Hessian": "#8C564B",
    "CMA-ES": "#2CA02C",
}

LINES = {
    "SIGMA-QGD v9.0": "-",
    "Adam": "-.",
    "COBYLA": "--",
    "SPSA": ":",
    "Diag-QNG": "-",
    "QN-SPSA": "--",
    "AdaBelief": "-.",
    "RMSProp": ":",
    "Yogi": "--",
    "H-QNG (Shi&Wang)": "-.",
    "Hess-Mag": "-.",
    "Diag-Hessian": "--",
    "CMA-ES": "-",
}

LW = {
    "SIGMA-QGD v9.0": 3.0,
    "Adam": 1.8,
    "COBYLA": 1.5,
    "SPSA": 1.5,
    "Diag-QNG": 2.0,
    "QN-SPSA": 1.8,
    "AdaBelief": 1.8,
    "RMSProp": 1.5,
    "Yogi": 1.5,
    "H-QNG (Shi&Wang)": 1.8,
    "Hess-Mag": 1.8,
    "Diag-Hessian": 1.8,
    "CMA-ES": 2.0,
}

MARKERS = {
    "SIGMA-QGD v9.0": "D",
    "Adam": "X",
    "COBYLA": "H",
    "SPSA": "p",
    "Diag-QNG": "^",
    "QN-SPSA": "v",
    "AdaBelief": "P",
    "RMSProp": "s",
    "Yogi": "*",
    "H-QNG (Shi&Wang)": "<",
    "Hess-Mag": "<",
    "Diag-Hessian": "h",
    "CMA-ES": "o",
}

LR_SWEEP_METHODS = [
    "SIGMA-QGD v9.0",
    "Adam",
    "AdaBelief",
    "RMSProp",
    "Yogi",
    "H-QNG (Shi&Wang)",
    "Hess-Mag",
    "Diag-Hessian",
]

BENCHMARK_METHODS = [
    "SIGMA-QGD v9.0",
    "Adam",
    "COBYLA",
    "SPSA",
    "Diag-QNG",
    "QN-SPSA",
    "AdaBelief",
    "RMSProp",
    "Yogi",
    "H-QNG (Shi&Wang)",
    "Diag-Hessian",
]

TOP2_BASELINE_SET = ["SIGMA-QGD v9.0", "Adam", "Diag-QNG"]

DEFAULT_LR: float = 0.1
DEFAULT_SEEDS: int = 20
ABLATION_MAX_SEEDS: int = 15

OPTIMAL_LRS = {
    "SIGMA-QGD v9.0": 0.1,
    "Adam": 0.1,
    "AdaBelief": 0.1,
    "Yogi": 0.1,
    "RMSProp": 0.01,
    "Diag-Hessian": 0.1,
    "H-QNG (Shi&Wang)": 0.001,
    "Hess-Mag": 0.001,
    "H-QNG": 0.001,
}

OPTIMAL_LRS_BY_HAM = {
    "tfim": {
        "SIGMA-QGD v9.0": 0.1,
        "Adam": 0.1,
        "AdaBelief": 0.1,
        "Yogi": 0.1,
        "RMSProp": 0.01,
        "Diag-Hessian": 0.1,
        "H-QNG (Shi&Wang)": 0.001,
        "Hess-Mag": 0.001,
        "H-QNG": 0.001,
    },
    "xxz": {
        "SIGMA-QGD v9.0": 0.1,
        "Adam": 0.1,
        "AdaBelief": 0.1,
        "Yogi": 0.1,
        "RMSProp": 0.05,
        "Diag-Hessian": 0.1,
        "H-QNG (Shi&Wang)": 0.001,
        "Hess-Mag": 0.001,
        "H-QNG": 0.001,
    },
}


def get_optimal_lr(method_name: str, ham_type: str = "tfim", n_qubits: int = 10) -> float:
    """Return the tuned learning rate for a method given Hamiltonian and system size."""
    s = str(ham_type).lower()
    ht = "xxz" if ("xxz" in s or "heis" in s) else "tfim"
    nq = int(n_qubits)

    if ht == "xxz" and nq == 12 and method_name == "AdaBelief":
        return 0.01

    if ht in OPTIMAL_LRS_BY_HAM and method_name in OPTIMAL_LRS_BY_HAM[ht]:
        return OPTIMAL_LRS_BY_HAM[ht][method_name]

    return OPTIMAL_LRS.get(method_name, DEFAULT_LR)
