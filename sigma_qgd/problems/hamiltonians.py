"""
Hamiltonian constructors and exact ground state solvers for VQE benchmarks.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml


def build_tfim(n: int, J: float = 1.0, h: float = 0.5) -> qml.Hamiltonian:
    """
    Transverse-field Ising model with periodic boundary conditions:
        H = -J ∑_i Z_i Z_{i+1} - h ∑_i X_i
    """
    coeffs = []
    ops = []

    for i in range(n):
        j = (i + 1) % n
        coeffs.append(-J)
        ops.append(qml.PauliZ(i) @ qml.PauliZ(j))

    for i in range(n):
        coeffs.append(-h)
        ops.append(qml.PauliX(i))

    return qml.Hamiltonian(coeffs, ops)


def build_xxz(n: int, Jxy: float = 1.0, Jz: float = 0.5, h: float = 0.1) -> qml.Hamiltonian:
    """
    Heisenberg XXZ model with periodic boundary conditions:
        H = Jxy ∑_i (X_i X_{i+1} + Y_i Y_{i+1}) + Jz ∑_i Z_i Z_{i+1} + h ∑_i Z_i
    """
    coeffs = []
    ops = []

    for i in range(n):
        j = (i + 1) % n
        coeffs += [Jxy, Jxy, Jz]
        ops += [
            qml.PauliX(i) @ qml.PauliX(j),
            qml.PauliY(i) @ qml.PauliY(j),
            qml.PauliZ(i) @ qml.PauliZ(j),
        ]

    for i in range(n):
        coeffs.append(h)
        ops.append(qml.PauliZ(i))

    return qml.Hamiltonian(coeffs, ops)


def exact_gs(H: qml.Hamiltonian) -> float:
    """
    Compute exact minimum eigenvalue of the Hamiltonian via dense diagonalization.
    """
    mat = np.array(qml.matrix(H))
    return float(np.min(np.linalg.eigvalsh(mat)))
