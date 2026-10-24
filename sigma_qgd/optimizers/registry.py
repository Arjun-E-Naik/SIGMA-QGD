"""
Optimizer registry mapping names to optimizer instances.
"""

from __future__ import annotations

from .sigma_qgd import SigmaQGD
from .adam_family import Adam, AdaBelief, RMSProp, Yogi
from .hessian_based import DiagHessian
from .qng_methods import DiagQNG, QNSPSA
from .derivative_free import COBYLA, SPSA, CMAES
from .hamiltonian_qng import HamiltonianQNG

OPTIMIZER_REGISTRY = {
    "SIGMA-QGD v9.0": SigmaQGD(),
    "Adam": Adam(),
    "AdaBelief": AdaBelief(),
    "RMSProp": RMSProp(),
    "Yogi": Yogi(),
    "H-QNG (Shi&Wang)": HamiltonianQNG(),
    "Hess-Mag": HamiltonianQNG(),
    "H-QNG": HamiltonianQNG(),
    "Diag-Hessian": DiagHessian(),
    "Diag-QNG": DiagQNG(),
    "QN-SPSA": QNSPSA(),
    "COBYLA": COBYLA(),
    "SPSA": SPSA(),
    "CMA-ES": CMAES(),
}

LR_SWEEP_METHODS = [
    name for name, opt in OPTIMIZER_REGISTRY.items()
    if opt.supports_lr and name != "H-QNG"
]

KNOWN_METHODS = list(OPTIMIZER_REGISTRY.keys())


def get_optimizer(name: str):
    """Retrieve an optimizer instance by name."""
    if name not in OPTIMIZER_REGISTRY:
        raise ValueError(
            f"Unknown optimizer '{name}'. Available methods: {sorted(KNOWN_METHODS)}"
        )
    return OPTIMIZER_REGISTRY[name]
