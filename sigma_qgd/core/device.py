"""
Device selection with GPU → CPU fallback and discovery caching.

Priority order:
  1. lightning.gpu   (cuQuantum / cuStateVec — Kaggle / Linux / WSL2)
  2. qrack.simulator (OpenCL / CUDA — Native Windows GPU via pennylane-qrack)
  3. lightning.qubit (Fast C++ OpenMP CPU backend)
  4. default.qubit   (Pure-Python fallback, always available)
"""

from typing import Optional, Tuple
import pennylane as qml
from pennylane.tape import QuantumTape

# Global discovery cache to avoid re-probing on every CostEvaluator instantiation
_DISCOVERED_BACKEND: Optional[str] = None
_DISCOVERED_STATUS: Optional[str] = None


def make_device(
    n_qubits: int,
    device_pref: str = "auto",
    verbose: bool = False,
) -> tuple:
    """
    Return (device, status_string) with automatic GPU → CPU fallback.

    Parameters
    ----------
    n_qubits : int
        Number of qubits for the device.
    device_pref : str, default="auto"
        Desired backend: "auto", "gpu", "cpu", "qrack", "lightning.gpu",
        "lightning.qubit", or "default.qubit".
    verbose : bool, default=False
        If True, prints warning messages when falling back to CPU.

    Returns
    -------
    device : pennylane.Device
    status : str
    """
    global _DISCOVERED_BACKEND, _DISCOVERED_STATUS

    # Fast path: use previously verified backend if auto-detecting
    if device_pref == "auto" and _DISCOVERED_BACKEND is not None:
        try:
            return qml.device(_DISCOVERED_BACKEND, wires=n_qubits), _DISCOVERED_STATUS
        except Exception:
            pass  # Fall through to full probe if cached device fails on new wire count

    pref = device_pref.lower().strip()

    # --- Attempt 1: lightning.gpu (cuQuantum / Linux / Kaggle) ---
    if pref in ("auto", "gpu", "lightning.gpu"):
        try:
            dev = qml.device("lightning.gpu", wires=n_qubits)
            with QuantumTape() as qc:
                qml.Hadamard(wires=0)
                qml.expval(qml.PauliZ(0))
            try:
                dev.execute(qc)
            except Exception:
                qml.execute([qc], dev)
            status = "GPU (lightning.gpu, cuQuantum)"
            if pref == "auto":
                _DISCOVERED_BACKEND = "lightning.gpu"
                _DISCOVERED_STATUS = status
            return dev, status
        except Exception as e_gpu:
            if verbose and pref in ("gpu", "lightning.gpu"):
                print(
                    f"  [Device] lightning.gpu unavailable "
                    f"({e_gpu.__class__.__name__}: {e_gpu})."
                )

    # --- Attempt 2: qrack.simulator (Native Windows GPU via OpenCL/CUDA) ---
    if pref in ("auto", "gpu", "qrack", "qrack.simulator"):
        try:
            dev = qml.device("qrack.simulator", wires=n_qubits)
            with QuantumTape() as qc:
                qml.Hadamard(wires=0)
                qml.expval(qml.PauliZ(0))
            try:
                dev.execute(qc)
            except Exception:
                qml.execute([qc], dev)
            status = "GPU (qrack.simulator, CUDA/OpenCL)"
            if pref == "auto":
                _DISCOVERED_BACKEND = "qrack.simulator"
                _DISCOVERED_STATUS = status
            return dev, status
        except Exception as e_qrack:
            if verbose and pref in ("gpu", "qrack", "qrack.simulator"):
                print(
                    f"  [Device] qrack.simulator unavailable "
                    f"({e_qrack.__class__.__name__}: {e_qrack})."
                )

    # --- Attempt 3: lightning.qubit (Fast C++ OpenMP CPU backend) ---
    if pref in ("auto", "cpu", "lightning.qubit"):
        try:
            dev = qml.device("lightning.qubit", wires=n_qubits)
            status = "CPU (lightning.qubit)"
            if pref == "auto":
                _DISCOVERED_BACKEND = "lightning.qubit"
                _DISCOVERED_STATUS = status
            return dev, status
        except Exception as e_cpu:
            if verbose:
                print(
                    f"  [Device] lightning.qubit unavailable ({e_cpu}). "
                    f"Falling back to default.qubit."
                )

    # --- Fallback: default.qubit (Pure Python) ---
    status = "CPU (default.qubit)"
    if pref == "auto":
        _DISCOVERED_BACKEND = "default.qubit"
        _DISCOVERED_STATUS = status
    return qml.device("default.qubit", wires=n_qubits), status


if __name__ == "__main__":
    dev, status = make_device(10, verbose=True)
    print(f"Chosen device: {dev}")
    print(f"Status: {status}")

