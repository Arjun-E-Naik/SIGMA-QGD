"""
Parameterized quantum circuit (ansatz) definitions.
"""

from __future__ import annotations

import pennylane as qml


def ansatz_function(theta, n: int = 4, reps: int = 3) -> None:
    """
    Hardware-efficient ansatz with Ry single-qubit rotations and CNOT ladder entanglers.

    Parameters
    ----------
    theta : array-like of shape (n * reps,)
        Variational parameters.
    n : int
        Number of qubits.
    reps : int
        Number of variational layers.
    """
    idx = 0
    for _ in range(reps):
        for q in range(n):
            qml.RY(theta[idx], wires=q)
            idx += 1
        for q in range(n - 1):
            qml.CNOT(wires=[q, q + 1])


def build_ansatz(n: int = 4, reps: int = 3) -> tuple:
    """Return ansatz system specifications (qubits, layers)."""
    return (n, reps)
