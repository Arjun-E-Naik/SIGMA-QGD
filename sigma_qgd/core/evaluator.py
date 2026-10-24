"""
Circuit cost evaluation and shot-noise simulation wrappers.
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import pennylane as qml
from pennylane.tape import QuantumScript, QuantumTape
from .device import make_device
from ..problems.ansatz import ansatz_function


class CostEvaluator:
    """
    Wraps a PennyLane device and tracks total circuit evaluations.
    """

    def __init__(
        self,
        n_qubits: int,
        reps: int,
        H: qml.Hamiltonian,
        device=None,
        shots: Optional[int] = None,
        exact_E: Optional[float] = None,
    ):
        self.n_qubits = n_qubits
        self.reps = reps
        self.H = H
        self.n_calls = 0
        self.shots = shots
        self.exact_E = exact_E
        if device is not None:
            self.device = device
            self.status = "external device"
        else:
            self.device, self.status = make_device(n_qubits)

    def _tape(self, theta: np.ndarray) -> QuantumScript:
        with QuantumTape() as tape:
            ansatz_function(theta, self.n_qubits, self.reps)
            qml.expval(self.H)
        if self.shots is not None:
            tape = QuantumScript(tape.operations, tape.measurements, shots=self.shots)
        return tape

    def _batch_execute(self, tapes):
        try:
            return self.device.execute(tapes)
        except Exception:
            pass
        try:
            return self.device.batch_execute(tapes)
        except AttributeError:
            return qml.execute(tapes, self.device, diff_method=None)

    def __call__(self, theta: np.ndarray) -> float:
        """Single energy evaluation (1 circuit)."""
        self.n_calls += 1
        result = self._batch_execute([self._tape(theta)])
        return float(np.asarray(result[0]))

    def gradient(self, theta: np.ndarray) -> np.ndarray:
        """Parameter-shift gradient (2p circuits)."""
        p = len(theta)
        tapes = []
        for i in range(p):
            tp = theta.copy()
            tp[i] += np.pi / 2
            tm = theta.copy()
            tm[i] -= np.pi / 2
            tapes.append(self._tape(tp))
            tapes.append(self._tape(tm))
        results = self._batch_execute(tapes)
        self.n_calls += len(tapes)
        g = np.zeros(p)
        for i in range(p):
            g[i] = 0.5 * (float(np.asarray(results[2 * i])) - float(np.asarray(results[2 * i + 1])))
        return g

    def cost_and_gradient(self, theta: np.ndarray) -> Tuple[float, np.ndarray]:
        """Energy and gradient evaluated jointly (2p+1 circuits total)."""
        cost, g, _ = self._cost_grad_hess(theta)
        return cost, g

    def cost_gradient_and_diag_hessian(
        self, theta: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Energy, gradient, and diagonal Hessian elements (2p+1 circuits total).

        Uses the parameter-shift formula for diagonal Hessian entries:
            h_ii = 0.5 * (E(θ + π/2 e_i) + E(θ - π/2 e_i) - 2 * E(θ))
        """
        return self._cost_grad_hess(theta)

    def _cost_grad_hess(
        self, theta: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        p = len(theta)
        tapes = [self._tape(theta)]
        for i in range(p):
            tp = theta.copy()
            tp[i] += np.pi / 2
            tm = theta.copy()
            tm[i] -= np.pi / 2
            tapes.append(self._tape(tp))
            tapes.append(self._tape(tm))
        results = self._batch_execute(tapes)
        self.n_calls += len(tapes)
        cost = float(np.asarray(results[0]))
        g = np.zeros(p)
        hess_diag = np.zeros(p)
        for i in range(p):
            ev_plus = float(np.asarray(results[1 + 2 * i]))
            ev_minus = float(np.asarray(results[2 + 2 * i]))
            g[i] = 0.5 * (ev_plus - ev_minus)
            hess_diag[i] = 0.5 * (ev_plus + ev_minus - 2.0 * cost)
        return cost, g, hess_diag

    def statevector(self, theta: np.ndarray) -> np.ndarray:
        """Return the statevector of the ansatz state."""
        self.n_calls += 1
        with QuantumTape() as tape:
            ansatz_function(theta, self.n_qubits, self.reps)
            qml.state()
        return np.asarray(self._batch_execute([tape])[0])

    def overlap_call(self, sv_base: np.ndarray, theta_shift: np.ndarray) -> float:
        """Compute the squared state overlap |<ψ|ψ'>|^2."""
        self.n_calls += 1
        with QuantumTape() as tape:
            ansatz_function(theta_shift, self.n_qubits, self.reps)
            qml.state()
        sv_shift = np.asarray(self._batch_execute([tape])[0])
        return float(abs(np.vdot(sv_base, sv_shift)) ** 2)


class ShotNoiseCostEvaluator:
    """
    Cost evaluator with finite measurement shots.
    """

    def __init__(
        self,
        n_qubits: int,
        reps: int,
        H: qml.Hamiltonian,
        n_shots: int = 1024,
        exact_E: Optional[float] = None,
    ):
        self.n_qubits = n_qubits
        self.reps = reps
        self.H = H
        self.n_shots = n_shots
        self.n_calls = 0
        self.exact_E = exact_E
        device, self.status = make_device(n_qubits)
        self.base = CostEvaluator(
            n_qubits, reps, H, device=device, shots=n_shots, exact_E=exact_E
        )

    def __call__(self, theta: np.ndarray) -> float:
        self.n_calls += 1
        val = self.base(theta)
        self.base.n_calls -= 1
        return val

    def gradient(self, theta: np.ndarray) -> np.ndarray:
        p = len(theta)
        g = np.zeros(p)
        for i in range(p):
            p_plus = theta.copy()
            p_plus[i] += np.pi / 2
            p_minus = theta.copy()
            p_minus[i] -= np.pi / 2
            g[i] = 0.5 * (self(p_plus) - self(p_minus))
        return g

    def cost_and_gradient(self, theta: np.ndarray) -> Tuple[float, np.ndarray]:
        cost, g, _ = self.cost_gradient_and_diag_hessian(theta)
        return cost, g

    def cost_gradient_and_diag_hessian(
        self, theta: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        p = len(theta)
        cost = self(theta)
        g = np.zeros(p)
        hess = np.zeros(p)
        for i in range(p):
            tp = theta.copy()
            tp[i] += np.pi / 2
            tm = theta.copy()
            tm[i] -= np.pi / 2
            ev_plus = self(tp)
            ev_minus = self(tm)
            g[i] = 0.5 * (ev_plus - ev_minus)
            hess[i] = 0.5 * (ev_plus + ev_minus - 2.0 * cost)
        return cost, g, hess
