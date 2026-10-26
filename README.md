# SIGMA-QGD: Variance-Aware Preconditioned Optimization for Variational Quantum Algorithms

Official implementation and empirical benchmark suite for **SIGMA-QGD** (Statistical Information Gradient Modulation Algorithm for Quantum Gradient Descent).

---

## Overview

Variational Quantum Algorithms (VQAs) such as the Variational Quantum Eigensolver (VQE) frequently encounter ill-conditioned parameter landscapes and vanishing gradients. While Quantum Natural Gradient (QNG) methods mitigate these challenges by preconditioning updates with the Fubini-Study metric tensor, standard QNG introduces substantial circuit evaluation overhead.

**SIGMA-QGD** addresses this bottleneck by constructing a variance-based diagonal curvature proxy directly from gradient trajectory dynamics using Bessel-corrected Welford accumulation and James-Stein shrinkage. This achieves second-order preconditioning without any additional quantum circuit calls beyond standard parameter-shift gradients ($2p + 1$ circuits per step).

---

## Repository Structure

```
.
├── configs/                  # Experiment configuration files (YAML)
│   ├── benchmark_10q.yaml    # 10-qubit TFIM & XXZ benchmarks
│   ├── benchmark_12q.yaml    # 12-qubit TFIM & XXZ benchmarks
│   ├── lr_sweep.yaml         # Learning rate grid sweep
│   ├── ablation.yaml         # Component ablation study
│   ├── shot_noise.yaml       # Finite-shot noise evaluations
│   ├── scaling.yaml          # System-size scaling (qubit sweep)
│   ├── sensitivity.yaml      # Hyperparameter sensitivity sweep
│   └── cma_sigma.yaml        # Budget-matched CMA-ES comparison
├── figures/                  # Publication-ready figures and plots
├── scripts/                  # Command-line drivers and analysis utilities
│   ├── run_experiment.py     # Main experiment driver
│   ├── generate_plots.py     # Figure generation utility
│   ├── run_tests.py          # Unit and regression test suite
│   └── run_qnspsa_benchmark.py # Dedicated QN-SPSA benchmark runner
├── sigma_data/               # Authoritative experimental CSV data and records
│   ├── benchmark_10q/        # Raw 10-qubit trial results
│   ├── benchmark_12q/        # Raw 12-qubit trial results
│   ├── benchmark_results/    # Consolidated benchmark tables
│   ├── ablation_results/     # Consolidated ablation tables
│   ├── shot_noise/           # Shot-noise robustness data
│   ├── scaling/              # Scaling study records
│   └── sensitivity_sweep_results/ # Hyperparameter sensitivity records
├── sigma_qgd/                # Core library
│   ├── analysis/             # Statistical analysis (Wilcoxon, Holm) and plotting
│   ├── core/                 # Evaluators, device backends, logging, config
│   ├── experiments/          # Benchmark and experiment execution routines
│   ├── optimizers/           # Optimizer implementations and registry
│   └── problems/             # Hamiltonians (TFIM, XXZ) and ansatz builders
├── tests/                    # Test modules for circuit verification and stats
├── run_all_results.py        # Single script to sequentially run all experiments
├── run_all_results.ps1       # PowerShell automation script for all experiments
├── requirements.txt          # Python dependencies
└── README.md
```

---

## Implemented Optimizer Baselines

All baseline optimizers are implemented according to their respective peer-reviewed literature formulations:

| Optimizer | Type | Reference | Circuit Cost / Step |
| :--- | :--- | :--- | :--- |
| **SIGMA-QGD v9.0** | Variance Preconditioned | Novel Framework | $2p + 1$ |
| **Adam** | Adaptive Momentum | Kingma & Ba (ICLR 2015) | $2p + 1$ |
| **AdaBelief** | Adaptive Variance | Zhuang et al. (NeurIPS 2020) | $2p + 1$ |
| **RMSProp** | Moving Average | Tieleman & Hinton (2012) | $2p + 1$ |
| **Yogi** | Sign-Controlled Adaptive | Zaheer et al. (NeurIPS 2018) | $2p + 1$ |
| **H-QNG** | Hamiltonian-Induced Pullback | Shi & Wang (arXiv:2511.14511) | $2p + 1$ |
| **Diag-Hessian** | Exact Diagonal Newton | Wierichs et al. (Quantum 6, 677, 2022) | $2p + 1$ |
| **Diag-QNG** | Diagonal Fubini-Study | Stokes et al. (Quantum 4, 269, 2020) | $3p + 2$ |
| **QN-SPSA** | Stochastic QNG | Gacon et al. (Quantum 5, 567, 2021) | $8$ |
| **COBYLA** | Derivative-Free Simplex | Powell (1994) / SciPy | $1$ |
| **SPSA** | Simultaneous Perturbation | Spall (IEEE TAC 1992, 1998) | $3$ |
| **CMA-ES** | Evolutionary Strategy | Hansen & Ostermeier (2001) | Budget-matched |

---

## Installation

### Prerequisites
- Python 3.9+ (Python 3.10 or 3.11 recommended)
- Virtual environment tool (`venv` or `conda`)

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/sigma-qgd.git
cd sigma-qgd

# Create and activate virtual environment
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

---

## Reproducing Results

### 1. Regenerating All Benchmark Results
To run every experiment across all configuration files in sequence:

```bash
# Cross-platform Python execution:
python run_all_results.py

# Alternatively, on Windows PowerShell:
.\run_all_results.ps1
```

### 2. Running Individual Experiments
To run a specific experiment configuration:

```bash
# 10-qubit TFIM & XXZ Benchmark:
python scripts/run_experiment.py --config configs/benchmark_10q.yaml

# 12-qubit Benchmark:
python scripts/run_experiment.py --config configs/benchmark_12q.yaml

# Learning Rate Grid Sweep:
python scripts/run_experiment.py --config configs/lr_sweep.yaml

# Component Ablation Study:
python scripts/run_experiment.py --config configs/ablation.yaml

# Finite Shot Noise Evaluation:
python scripts/run_experiment.py --config configs/shot_noise.yaml

# Qubit Scaling Study:
python scripts/run_experiment.py --config configs/scaling.yaml
```

### 3. Generating Figures
To re-generate plots from existing CSV data records:

```bash
python scripts/generate_plots.py --results-dir sigma_data/benchmark_results/ --tag tfim_10q
```

---

## Verification & Testing

To execute the test suite (verifying circuit counts, statistical corrections, and optimizer contracts):

```bash
# Fast smoke tests
python scripts/run_tests.py --quick

# Full circuit count and numerical tests
python scripts/run_tests.py
```

---


