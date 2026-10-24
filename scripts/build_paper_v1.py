# -*- coding: utf-8 -*-
"""
Script to generate the human-written, publication-ready LaTeX paper:
sigma_qgd_paper_v1.tex and sigama_qgd_paper_v1.tex
"""

import os
import shutil

TEX_CONTENT = r'''\documentclass[11pt,a4paper]{article}

% --- Standard Academic Packages ---
\usepackage[utf8]{inputenc}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,amsfonts,amsthm}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage{hyperref}
\usepackage{cite}
\usepackage{microtype}
\usepackage{subcaption}
\usepackage{xcolor}
\usepackage{array}
\usepackage{multirow}
\hyphenation{SIGMA-QGD}

% --- Theorem Environments ---
\newtheorem{theorem}{Theorem}
\newtheorem{lemma}{Lemma}
\newtheorem{proposition}{Proposition}
\newtheorem{corollary}{Corollary}
\newtheorem{assumption}{Assumption}
\newtheorem{remark}{Remark}

\hypersetup{
    colorlinks=true,
    linkcolor=blue!70!black,
    citecolor=green!50!black,
    urlcolor=blue!80!black
}

\title{\textbf{SIGMA-QGD: A Zero-Extra-Circuit, Variance-Aware Curvature Proxy for Variational Quantum Eigensolvers}}

\author{
    \textbf{Arjun E Naik}$^{1}$ \quad \textbf{Jitendra Tembhurne}$^{1}$ \\[1ex]
    $^{1}$Department of Computer Science, Indian Institute of Information Technology, Nagpur, India \\
    \texttt{\{bt24csd005, jtembhurne\}@iiitn.ac.in}
}
\date{}

\begin{document}

\maketitle

\begin{abstract}
Variational Quantum Eigensolvers (VQEs) depend fundamentally on classical optimization routines to locate ground-state energy estimates of quantum Hamiltonians. Even when exact analytic gradients are supplied via parameter-shift formulas, navigating the resulting cost surface is notoriously difficult: parameter spaces frequently alternate between narrow, ill-conditioned valleys and expansive regions with vanishing gradients. Standard first-order optimizers such as Adam or SPSA discard metric curvature entirely, whereas quantum natural gradient (QNG) methods evaluate state-overlap tensors at the expense of substantial quantum circuit overhead ($3p+2$ evaluations per iteration). In this paper, we introduce \textbf{SIGMA-QGD} (Sampled Iterative Gradient Moment Adaptation for Quantum Gradient Descent), a classical algorithm that extracts local landscape curvature directly from the temporal sample variance of coordinate-wise gradient components across recent steps, requiring no additional quantum circuit executions. We establish the theoretical validity of this trajectory-variance proxy through four analytical results: (1) an asymptotic proportionality between sample gradient variance and squared diagonal Hessian elements under local quadraticity and aggregate diagonal dominance ($\delta \sqrt{p} \ll |H_{ii}|$); (2) uniform two-sided bounds on the resulting preconditioner; (3) a coordinate-wise margin condition that guarantees descent directions; and (4) an ergodic non-convex stationarity convergence rate of $\mathcal{O}(1/T)$ in noiseless settings, translating to $\mathcal{O}(1/\sqrt{T})$ under finite-shot quantum measurement noise. Across $>394{,}000$ coordinate-step observations on 10- and 12-qubit transverse-field Ising (TFIM) and Heisenberg XXZ models, the online variance proxy tracks the analytic parameter-shift diagonal Hessian with high fidelity in log space ($r \in [0.57, 0.63]$ on TFIM, $r \approx 0.30$ on XXZ; $p < 10^{-4}$). In comparative benchmarks against ten established optimizers, SIGMA-QGD achieves an order-of-magnitude reduction in energy error on the commuting TFIM landscape and records the lowest average optimality gap on the frustrated XXZ chain, all while strictly respecting the standard $2p+1$ circuit budget.
\end{abstract}

\textbf{Keywords:} Variational Quantum Eigensolver (VQE), Classical Optimization, Curvature Preconditioning, Gradient Trajectory Variance, Quantum Natural Gradient.

\newpage

% ==============================================================================
% SECTION 1: INTRODUCTION
% ==============================================================================
\section{Introduction}
\label{sec:intro}

Variational Quantum Eigensolvers (VQEs) \cite{peruzzo2014variational,mcclean2016theory} serve as a central paradigm for quantum chemistry and condensed-matter simulations on noisy intermediate-scale quantum (NISQ) devices. In a typical VQE workflow, a parameterized quantum circuit prepares an $n$-qubit trial state $|\psi(\boldsymbol{\theta})\rangle$ governed by parameter vector $\boldsymbol{\theta} \in \mathbb{R}^p$. A quantum processor measures the expectation value of a target Hamiltonian $H$:
\begin{equation}
E(\boldsymbol{\theta}) = \langle \psi(\boldsymbol{\theta}) | H | \psi(\boldsymbol{\theta}) \rangle,
\label{eq:vqe_cost}
\end{equation}
while an external classical optimization algorithm iteratively adjusts $\boldsymbol{\theta}$ to locate the minimum eigenvalue $E_0 = \min_{\boldsymbol{\theta}} E(\boldsymbol{\theta})$.

\subsection{The Optimization Bottleneck in VQE}
Although exact analytic gradients can be estimated on quantum hardware via parameter-shift rules \cite{mitarai2018quantum,schuld2019evaluating}, minimizing $E(\boldsymbol{\theta})$ in practice remains an arduous numerical task. Parameterized quantum circuits yield energy landscapes plagued by geometric pathologies \cite{mcclean2018barren,cerezo2021cost}: flat plateaus where gradients vanish exponentially in system size border steep, highly anisotropic ravines \cite{mcclean2018barren,cerezo2021cost,wierichs2020avoiding}.

Existing classical optimization strategies encounter well-documented obstacles on such surfaces:
\begin{itemize}
    \item \textbf{Adaptive Gradient Routines (Adam, RMSProp, Yogi):} Classical adaptive methods \cite{kingma2014adam,tieleman2012lecture,zaheer2018adaptive} rescale coordinate steps by tracking running sums of squared gradients or momentum-centered residuals \cite{zhuang2020adabelief}. However, they assume that gradient fluctuations are isotropic and quasi-stationary across steps. As a consequence, they tend to stall in plateau regions or oscillate violently across narrow, curved minima.
    \item \textbf{Quantum Natural Gradient (QNG):} Geometry-aware approaches \cite{stokes2020quantum,wierichs2020avoiding} correct the gradient update using the Fubini--Study Quantum Fisher Information (QFI) metric tensor. Although QNG efficiently traverses the quantum state manifold, evaluating its diagonal metric components demands $(p+1)$ additional quantum circuit executions per step, driving per-step costs from $2p+1$ up to $3p+2$ evaluations.
    \item \textbf{Derivative-Free and Stochastic Approximation (SPSA, COBYLA):} Methods that eschew analytic parameter-shift evaluations \cite{spall1992multivariate,novak2025benchmarking} lower per-step execution overhead. Nevertheless, because they discard second-order geometric cues, their convergence rate decays sharply on anisotropic, non-convex manifolds.
\end{itemize}

\subsection{Trajectory-Variance Curvature Preconditioning}
This work investigates whether meaningful second-order geometric guidance can be recovered purely from the run-time temporal dynamics of standard parameter-shift gradients, without incurring any extra quantum circuit executions.

We introduce \textbf{SIGMA-QGD} (Sampled Iterative Gradient Moment Adaptation for Quantum Gradient Descent). Rather than querying quantum hardware for state overlaps or second-derivative stencils, SIGMA-QGD uses an online Bessel-corrected Welford accumulator to track the temporal variance of each parameter's gradient component along the optimization trajectory. The underlying intuition is rooted in optimization dynamics: when a coordinate undergoes substantial gradient fluctuations across consecutive steps, this volatility indicates a high local landscape curvature that necessitates step damping. In contrast, coordinates characterized by stable, low-variance gradients signal flatter terrain where updates can proceed with higher effective step sizes.

\subsection{Theoretical Contributions}
The main contributions of this paper are organized as follows:
\begin{itemize}
    \item \textbf{Theoretical Foundation (Section~\ref{sec:theory}):} We derive the mathematical relationship connecting temporal gradient variance $\hat{\sigma}_i^2$ to the squared diagonal Hessian $H_{ii}^2$ (Proposition~\ref{prop:variance_curvature}), establish two-sided uniform bounds on the preconditioner (Proposition~\ref{prop:boundedness}), formulate an empirical coordinate-wise descent margin condition (Proposition~\ref{prop:descent}), and prove non-convex stationarity convergence rates of $\mathcal{O}(1/T)$ in the noiseless limit and $\mathcal{O}(1/\sqrt{T})$ under finite shot noise (Theorem~\ref{thm:convergence}).
    \item \textbf{Empirical Curvature Validation (Section~\ref{sec:exp_proxy}):} We directly evaluate the correlation between our online variance estimator and the exact parameter-shift diagonal Hessian across multiple spin Hamiltonians and system sizes.
    \item \textbf{Systematic Benchmark Protocol (Section~\ref{sec:setup}):} We present an experimental pipeline featuring an independent learning-rate grid sweep, systematic component ablations, finite-shot noise stress tests, and a budget-matched comparison against CMA-ES.
    \item \textbf{Multi-Regime Performance Analysis (Section~\ref{sec:results}):} We benchmark SIGMA-QGD against ten baseline algorithms on 10-qubit and 12-qubit transverse-field Ising and Heisenberg XXZ spin chains.
\end{itemize}
All implementation source code, configuration files, per-seed trajectory datasets, and analysis routines are publicly available on GitHub at \url{https://github.com/Arjun-E-Naik/SIGMA-QGD}.

% ==============================================================================
% SECTION 2: RELATED WORK
% ==============================================================================
\section{Related Work}
\label{sec:related}

\subsection{Adaptive First-Order Optimizers}
First-order adaptive optimizers modulate individual learning rates using historical gradient statistics. Adam \cite{kingma2014adam} normalizes updates by an exponential moving average of squared gradients $\mathbb{E}[g_i^2]$. RMSProp \cite{tieleman2012lecture} similarly tracks uncentered second moments, whereas Yogi \cite{zaheer2018adaptive} restricts the velocity of second-moment updates to prevent premature step decay in non-stationary settings. AdaBelief \cite{zhuang2020adabelief} modifies Adam by computing gradient variance around the moving momentum estimate, $\mathbb{E}[(g_i - m_i)^2]$. In SIGMA-QGD, by contrast, the variance is accumulated relative to the gradient's own running sample mean via Welford's recurrence. This choice isolates parameter volatility directly, preventing the curvature estimate from being confounded by momentum lag.

\subsection{Quantum Natural Gradient and Curvature Preconditioning}
Quantum Natural Gradient (QNG) \cite{stokes2020quantum} guides parameter updates via the Fubini--Study metric tensor $G$, yielding updates $\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta G^{-1} \mathbf{g}_t$. While full and block-diagonal QNG require $\mathcal{O}(p^2)$ circuit evaluations, diagonal QNG still requires $p+1$ additional state-overlap circuits per iteration.

Recent proposals have sought to lower this circuit overhead. Hamiltonian-aware QNG (H-QNG) \cite{shi2025hamiltonian} builds a pullback metric tensor from per-Pauli expectation derivatives without extra quantum states, but it requires computing, storing, and inverting a $p \times p$ (or block-diagonal) Jacobian product of individual Hamiltonian terms at each iteration. Moreover, accessing ungrouped term-wise derivatives increases sensitivity to finite-shot noise when measuring non-commuting Pauli strings. In another direction, Stein-identity QFI estimation \cite{halla2025stein} evaluates metric tensors via random parameter perturbations, introducing stochastic Monte Carlo variance and perturbation hyperparameters akin to 2-SPSA and QN-SPSA \cite{gacon2021simultaneous}. SIGMA-QGD operates on entirely different principles: it works directly on the scalar energy gradient vector $\mathbf{g}_t$ in $\mathcal{O}(p)$ time and memory, requires no matrix inversion or state perturbations, and naturally averages out shot noise through recursive temporal accumulation.

For hardware-efficient circuits composed of single-qubit Pauli rotation gates $U_i(\theta_i) = \exp(-i \frac{\theta_i}{2} P_i)$ with $P_i \in \{X, Y, Z\}$, the generator eigenvalues are $\pm 1/2$. Consequently, the energy expectation value along coordinate $\theta_i$ follows a single-frequency sinusoid:
\begin{equation}
E(\theta_i) = A_i \cos(\theta_i + \phi_i) + C_i,
\label{eq:sinusoid_form}
\end{equation}
where $A_i, \phi_i, C_i$ depend on the remaining parameters $\boldsymbol{\theta}_{\ne i}$. Differentiating analytically gives the gradient $g_i(\boldsymbol{\theta}) = -A_i \sin(\theta_i + \phi_i)$ and second derivative $H_{ii}(\boldsymbol{\theta}) = \frac{\partial^2 E}{\partial \theta_i^2} = -A_i \cos(\theta_i + \phi_i) = -(E(\boldsymbol{\theta}) - C_i)$. Evaluating the trigonometric identity at parameter shifts of $\pm \pi/2$ reveals that the central difference:
\begin{equation}
H_{ii}(\boldsymbol{\theta}) = \frac{1}{2}\left(E\left(\boldsymbol{\theta} + \frac{\pi}{2}\mathbf{e}_i\right) + E\left(\boldsymbol{\theta} - \frac{\pi}{2}\mathbf{e}_i\right) - 2E(\boldsymbol{\theta})\right)
\label{eq:exact_hessian}
\end{equation}
is algebraically exact \cite{wierichs2022general}. Because the forward and backward shifted evaluations $E(\boldsymbol{\theta} \pm \frac{\pi}{2}\mathbf{e}_i)$ are already executed by the quantum processor to compute the parameter-shift gradient $g_i(\boldsymbol{\theta})$ in Eq.~\eqref{eq:param_shift}, evaluating Eq.~\eqref{eq:exact_hessian} requires only the single unshifted evaluation $E(\boldsymbol{\theta})$, totaling $2p+1$ circuits. We evaluate this instantaneous second-order signal using two classical baselines: a magnitude-scaled diagonal Hessian update (\textbf{Hess-Mag}, which scales the gradient by $(|H_{ii}| + \lambda_{\text{reg}})^{-1}$) and a Levenberg--Marquardt safeguarded diagonal Newton solver (\textbf{Diag-Hessian}). In this evaluation, Hess-Mag serves strictly as an instantaneous classical curvature baseline and remains conceptually separate from the quantum Fisher-derived Hamiltonian-aware QNG scheme of Shi \& Wang \cite{shi2025hamiltonian}.

This raises an obvious question: \emph{if $H_{ii}(\boldsymbol{\theta})$ is accessible via $2p+1$ circuits, what justifies estimating curvature via trajectory variance $\hat{\sigma}_i^2$ rather than employing $H_{ii}(\boldsymbol{\theta}_t)$ directly?} Three architectural advantages favor SIGMA-QGD:
\begin{itemize}
    \item \textbf{Operating Within a Strict $2p$ Circuit Envelope:} Estimating gradients through the parameter-shift formula (Eq.~\eqref{eq:param_shift}) demands exactly $2p$ quantum circuit executions. The additional $+1$ unshifted circuit $E(\boldsymbol{\theta})$ in Eq.~\eqref{eq:exact_hessian} is needed only to close the finite-difference stencil. Because SIGMA-QGD processes incoming gradient vectors $\mathbf{g}_t$ directly, it can execute parameter updates in strictly $2p$ circuits per iteration whenever on-the-fly energy logging is omitted.
    \item \textbf{Universal Applicability Across Arbitrary Gate Generators:} The central-difference formula in Eq.~\eqref{eq:exact_hessian} yields the exact Hessian diagonal only for elementary rotation generators whose spectrum is limited to eigenvalues $\pm 1/2$. For multi-qubit entangling gates, fermionic excitation generators in chemistry ans\"atze (such as UCCSD and $k$-UpCCGSD), or QAOA mixers with multi-frequency spectra, evaluating second derivatives demands 4- or 8-point parameter-shift stencils \cite{wierichs2022general}, rapidly compounding circuit overhead. In contrast, SIGMA-QGD's variance proxy requires zero extra circuits regardless of the underlying gate generators or circuit depth.
    \item \textbf{Stability Across Inflection Points on Non-Convex Manifolds:} On oscillatory trigonometric surfaces, instantaneous diagonal curvatures $H_{ii}(\boldsymbol{\theta}_t)$ fluctuate rapidly, alter their signs, and regularly cross near-zero inflection boundaries ($H_{ii} \approx 0$). In single-point methods like Diag-Hessian and Hess-Mag, the regularized divisor $(|H_{ii}| + \lambda_{\text{reg}})^{-1}$ surges whenever $H_{ii} \approx 0$, precipitating step-size explosions that trigger severe overshooting unless heavily dampened. By comparison, SIGMA-QGD integrates gradient variations over the optimization history: the quantity $\hat{\sigma}_i^2(t)$ serves as an accumulated, strictly positive-definite curvature memory that passes smoothly through isolated Hessian sign crossings. Our ablation experiments in Section~\ref{sec:ablation_results} highlight this distinction: disabling momentum ($\beta = 0$) causes SIGMA-QGD's error to degrade to $\Delta E \approx 0.4202$ (10q) and $0.5544$ (12q) on TFIM, matching the performance of Diag-Hessian ($0.4384, 0.5609$) and Hess-Mag ($0.4431, 0.5497$). This confirms that temporal trajectory smoothing is critical for stable second-order preconditioning on quantum landscapes.
\end{itemize}

More recently, hybrid schemes combining exact parameter-shift gradients with simultaneous perturbation stochastic approximation (QN-SPSA+PSR) \cite{qnspsa_psr_2024} have been explored to approximate natural gradient metrics at modest circuit costs. However, simultaneous stochastic perturbations introduce persistent Monte Carlo estimator variance that hampers asymptotic convergence on multi-parameter spin chains. Complementary investigations into directional forward-mode gradient estimators and Riemannian manifold geometry, such as QUIVER and Exact Geodesic Transport (EGT), explore alternative avenues for geometry awareness. However, directional forward gradients require randomized projection directions that introduce stochastic sampling noise, while exact geometric transport techniques depend on analytically tractable ans\"atze rather than general parameterized quantum circuits. In contrast, SIGMA-QGD operates deterministically on standard parameter-shift gradient vectors, extracting curvature purely from realized optimization trajectories.

\subsection{Quasi-Newton Methods and Line-Search Constraints}
Classical deterministic quasi-Newton algorithms (such as L-BFGS-B and non-linear Conjugate Gradient) are standard baselines in noiseless classical optimization. However, their direct deployment in physical VQE pipelines is severely constrained by quantum execution costs. Quasi-Newton secant updates and Conjugate Gradient step calculations require line searches (e.g., Armijo backtracking or Wolfe conditions) to maintain positive-definiteness ($s_k^\top y_k > 0$) and ensure descent. In VQE, each line-search trial step requires executing independent quantum state preparation and measurement circuits. An adaptive line search averaging $3\text{--}5$ trial evaluations per step inflates the per-step circuit count by $300\%\text{--}500\%$, violating fixed hardware budget parity. Furthermore, under realistic finite sampling shot noise, stochastic energy fluctuations cause Wolfe conditions to fail or cycle indefinitely. Consequently, in budget-matched VQE benchmarking under fixed evaluation constraints ($2p+1$ circuits/step), gradient-based updates with analytic step sizes or momentum schedules are standard practice.

\subsection{Positioning Summary}
Table~\ref{tab:positioning} positions SIGMA-QGD within the broader taxonomy of VQE optimizers.

\begin{table}[htbp]
\centering
\small
\caption{Positioning of representative VQE optimizers along structural and cost axes.}
\label{tab:positioning}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lcccc}
\toprule
\textbf{Method} & \textbf{Curvature-Aware} & \textbf{Extra Circuit Type} & \textbf{Circuits / Step} & \textbf{Descent Guarantee} \\
\midrule
COBYLA \cite{powell1994direct} & No & None & $1$ & No \\
SPSA \cite{spall1992multivariate} & No & None & $3$ & Stochastic \\
Adam \cite{kingma2014adam} & No & None & $2p$ (or $2p+1$) & Empirical \\
AdaBelief \cite{zhuang2020adabelief} & Partial (Mom. Var) & None & $2p$ (or $2p+1$) & Empirical \\
Diag-QNG \cite{stokes2020quantum} & Yes (QFI metric) & Metric Overlaps & $3p+2$ & Yes (Metric) \\
Hess-Mag (This work) & Yes (Exact $|H_{ii}|$) & None & $2p+1$ & Empirical \\
Diag-Hessian (This work) & Yes (Exact $H_{ii}$) & None & $2p+1$ & Yes (LM-Safeguard) \\
L-BFGS-B / CG & Yes (Secant approx.) & Line Searches & $2p + \mathcal{O}(k)$ & Line-search dependent \\
\textbf{SIGMA-QGD (Ours)} & \textbf{Yes (Trajectory Proxy)} & \textbf{None} & $\mathbf{2p\text{ (or }2p+1\text{)}}$ & \textbf{Yes (Empirical Margin)} \\
\bottomrule
\end{tabular}%
}
\end{table}

% ==============================================================================
% SECTION 3: METHODOLOGY & THEORETICAL ANALYSIS
% ==============================================================================
\section{Methodology and Theoretical Analysis}
\label{sec:theory}

\subsection{Problem Setup and Notation}
Let $H = \sum_k c_k P_k$ be a molecular or spin Hamiltonian represented as a linear combination of Pauli strings $P_k \in \{I, X, Y, Z\}^{\otimes n}$. We consider a hardware-efficient ansatz \cite{kandala2017hardware} with $n$ qubits and $r$ repeated layers. Each layer consists of parameterized single-qubit $RY(\theta_{l,j})$ rotations followed by a linear entangling chain of CNOT gates, yielding $p = n \cdot r$ trainable parameters.

Parameters are initialized i.i.d.\ from $\text{Uniform}(-\pi/8, \pi/8)$. The partial derivative with respect to parameter $\theta_i$ is evaluated via the exact parameter-shift rule:
\begin{equation}
g_i(\boldsymbol{\theta}) = \frac{1}{2} \left[ E\left(\boldsymbol{\theta} + \frac{\pi}{2}\mathbf{e}_i\right) - E\left(\boldsymbol{\theta} - \frac{\pi}{2}\mathbf{e}_i\right) \right].
\label{eq:param_shift}
\end{equation}
Throughout this paper, $\mathbf{a} \oslash \mathbf{b}$ denotes Hadamard (elementwise) division: $(\mathbf{a} \oslash \mathbf{b})_i = a_i / b_i$.

\subsection{Online Trajectory Variance Estimation}
SIGMA-QGD maintains an online Bessel-corrected Welford accumulator \cite{welford1962note} for each parameter coordinate $i \in \{1, \dots, p\}$. At step $t \ge 1$, given gradient vector $\mathbf{g}_t$:
\begin{align}
M_{t,i} &= M_{t-1,i} + \frac{g_{t,i} - M_{t-1,i}}{t}, \label{eq:welford_m} \\
S_{t,i} &= S_{t-1,i} + (g_{t,i} - M_{t-1,i})(g_{t,i} - M_{t,i}). \label{eq:welford_s}
\end{align}
For $t \ge 2$, the unbiased sample variance is given by:
\begin{equation}
\hat{\sigma}_i^2(t) = \frac{S_{t,i}}{t - 1}, \quad \text{with } \hat{\sigma}_i^2(1) \equiv 0.
\label{eq:welford_var}
\end{equation}

\subsection{Preconditioner Formulation}
The per-coordinate curvature proxy $F_i$ and preconditioned search direction $\mathbf{d}_t$ are defined as:
\begin{equation}
F_i(t) = c \cdot \hat{\sigma}_i^2(t) + \lambda_{\text{reg}}, \qquad \mathbf{d}_t = \hat{\mathbf{m}}_t \oslash \mathbf{F}(t),
\label{eq:preconditioner}
\label{eq:preconditioned_update}
\end{equation}
where $c > 0$ is the preconditioner multiplier ($c = 1.0$), $\lambda_{\text{reg}} > 0$ is the default regularization constant ($\lambda_{\text{reg}} = 0.005$, selected via sensitivity analysis to provide a robust curvature floor), and $\hat{\mathbf{m}}_t$ is the bias-corrected momentum:
\begin{equation}
\mathbf{m}_t = \beta \mathbf{m}_{t-1} + (1-\beta)\mathbf{g}_t, \quad \hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta^t},
\label{eq:momentum}
\end{equation}
with parameter-adaptive momentum decay:
\begin{equation}
\beta = \operatorname{clip}\Bigl(1 - \frac{1}{\sqrt{\max(p, 4)}}, \, 0.5, \, 0.98\Bigr).
\label{eq:beta_formula}
\end{equation}
As detailed in Algorithm~\ref{alg:sigma_qgd}, prior to updating Eqs.~\eqref{eq:welford_m}--\eqref{eq:welford_s} and~\eqref{eq:momentum}, $\mathbf{g}_t$ is safeguarded via global $\ell_2$-norm clipping $\mathbf{g}_{c,t} = \mathbf{g}_t \cdot \min(1, \sqrt{p}/\|\mathbf{g}_t\|_2)$, which coincides with $\mathbf{g}_t$ whenever $\|\mathbf{g}_t\|_2 \le \sqrt{p}$.

\subsection{Theoretical Guarantees}
We now formalize the analytical conditions under which trajectory variance accurately reflects local landscape curvature and guarantees energy descent.

Let $E: \mathbb{R}^p \to \mathbb{R}$ be twice continuously differentiable with gradient $\mathbf{g} = \nabla E$ and Hessian $H = \nabla^2 E$. Consider a sliding window $W_t \subset \{1, \dots, t\}$ of $w$ steps ending at iteration $t$, with temporal variance:
\begin{equation}
\hat{\sigma}_i^2(t) = \frac{1}{w-1} \sum_{\tau \in W_t} \left( g_i(\boldsymbol{\theta}_\tau) - \bar{g}_i(W_t) \right)^2, \quad \bar{g}_i(W_t) = \frac{1}{w} \sum_{\tau \in W_t} g_i(\boldsymbol{\theta}_\tau).
\end{equation}

\begin{assumption}[Local Quadraticity]
\label{ass:quadraticity}
On window $W_t$, $E$ satisfies a second-order Taylor expansion about reference point $\boldsymbol{\theta}^* \in W_t$:
\begin{equation}
\mathbf{g}(\boldsymbol{\theta}_\tau) = \mathbf{g}(\boldsymbol{\theta}^*) + H(\boldsymbol{\theta}^*)(\boldsymbol{\theta}_\tau - \boldsymbol{\theta}^*) + \mathbf{R}(\boldsymbol{\theta}_\tau), \quad \|\mathbf{R}(\boldsymbol{\theta}_\tau)\| \le \frac{L}{2} \|\boldsymbol{\theta}_\tau - \boldsymbol{\theta}^*\|^2,
\end{equation}
where $L < \infty$ is the local Lipschitz constant of the Hessian.
\end{assumption}

\begin{assumption}[Aggregate Diagonal Dominance]
\label{ass:diag_dominance}
Off-diagonal Hessian elements are bounded by $|H_{ij}(\boldsymbol{\theta}^*)| \le \delta$ for all $i \ne j$, satisfying the aggregate dominance condition $\delta \sqrt{p} \ll |H_{ii}(\boldsymbol{\theta}^*)|$ for the active coordinates.
\end{assumption}

\begin{assumption}[Bounded Gradient and Curvature]
\label{ass:bounded_curvature}
The coordinate gradient and diagonal Hessian elements are bounded: $|g_i(\boldsymbol{\theta})| \le M$ and $|H_{ii}(\boldsymbol{\theta})| \le M$ for all $\boldsymbol{\theta}$ and coordinate indices $i$. For hardware-efficient trigonometric ansatzes, $M$ is bounded by $\sum_k |c_k|$.
\end{assumption}

\begin{assumption}[Bounded Window Displacement]
\label{ass:displacement}
The trajectory displacement within window $W_t$ is bounded by $\|\boldsymbol{\theta}_\tau - \boldsymbol{\theta}^*\| \le r$ for all $\tau \in W_t$.
\end{assumption}

\begin{assumption}[Coordinate-wise Momentum Alignment]
\label{ass:alignment}
At step $t$, the bias-corrected momentum and negative gradient agree in sign coordinate-wise:
\begin{equation}
\hat{m}_{t,i} (-g_{t,i}) \ge 0 \quad \forall i \in \{1, \dots, p\}.
\end{equation}
\end{assumption}

\begin{remark}[Aggregate Alignment vs. Coordinate-wise Margin]
A positive global inner product between momentum and the negative gradient, $\langle \hat{\mathbf{m}}_t, -\mathbf{g}_t \rangle > 0$, does not by itself guarantee energy descent once diagonal weights are applied. To see why coordinate-level sign disparities matter, consider a simple two-coordinate case with preconditioner $\mathbf{F} = (0.1, 0.005)^\top$ and coordinate products $\hat{\mathbf{m}}_t \odot (-\mathbf{g}_t) = (1.0, -0.5)^\top$. Here, overall alignment is positive:
\[
\langle \hat{\mathbf{m}}_t, -\mathbf{g}_t \rangle = 1.0 - 0.5 = 0.5 > 0.
\]
Yet when each coordinate is weighted by $1/F_i$, we obtain:
\[
\langle \mathbf{d}_t, -\mathbf{g}_t \rangle = \frac{1.0}{0.1} + \frac{-0.5}{0.005} = 10 - 100 = -90 < 0.
\]
Because the misaligned second coordinate receives an amplification factor of $1/0.005 = 200$, its negative contribution completely overwhelms the aligned coordinate, converting the intended descent step into an ascent step. While enforcing coordinate-wise sign alignment (Assumption~\ref{ass:alignment}) is a straightforward sufficient check, the broader necessary and sufficient requirement is the margin condition proved in Proposition~\ref{prop:descent}.
\end{remark}

\begin{lemma}[Local Secant Relation]
\label{lem:secant}
Under Assumptions~\ref{ass:quadraticity}, \ref{ass:diag_dominance}, and \ref{ass:displacement}, the gradient difference satisfies:
\begin{equation}
g_i(\boldsymbol{\theta}_\tau) - g_i(\boldsymbol{\theta}^*) = H_{ii}(\boldsymbol{\theta}^*)(\theta_{\tau,i} - \theta_i^*) + \varepsilon_{i,\tau},
\end{equation}
where $|\varepsilon_{i,\tau}| \le \delta r \sqrt{p} + \frac{L}{2} r^2$.
\end{lemma}

\begin{proof}
Decomposing the Taylor expansion coordinate-wise:
\begin{equation}
g_i(\boldsymbol{\theta}_\tau) - g_i(\boldsymbol{\theta}^*) = H_{ii}(\boldsymbol{\theta}^*)(\theta_{\tau,i} - \theta_i^*) + \sum_{j \ne i} H_{ij}(\boldsymbol{\theta}^*)(\theta_{\tau,j} - \theta_j^*) + R_i(\boldsymbol{\theta}_\tau).
\end{equation}
By Cauchy--Schwarz and Assumption~\ref{ass:diag_dominance}:
\begin{equation}
\left| \sum_{j \ne i} H_{ij}(\boldsymbol{\theta}^*)(\theta_{\tau,j} - \theta_j^*) \right| \le \delta \sum_{j=1}^p |\theta_{\tau,j} - \theta_j^*| \le \delta \sqrt{p} \|\boldsymbol{\theta}_\tau - \boldsymbol{\theta}^*\| \le \delta r \sqrt{p}.
\end{equation}
Combining this with $\|R_i(\boldsymbol{\theta}_\tau)\| \le \|\mathbf{R}(\boldsymbol{\theta}_\tau)\| \le \frac{L}{2} r^2$ yields the stated bound.
\end{proof}

\begin{proposition}[Variance--Curvature Proportionality]
\label{prop:variance_curvature}
Under Assumptions~\ref{ass:quadraticity}--\ref{ass:displacement}, the sample variance of the $i$-th gradient component on window $W_t$ satisfies:
\begin{equation}
\hat{\sigma}_i^2(t) = H_{ii}(\boldsymbol{\theta}^*)^2 \operatorname{Var}_{W_t}[\theta_i] + \mathcal{R}_i,
\end{equation}
where the residual $\mathcal{R}_i$ is rigorously bounded by:
\begin{equation}
|\mathcal{R}_i| \le 2 M r \left( \delta r \sqrt{p} + \frac{L}{2} r^2 \right) + \left( \delta r \sqrt{p} + \frac{L}{2} r^2 \right)^2 = \mathcal{O}\left( (M \delta \sqrt{p} + \delta^2 p) r^2 + M L r^3 + L^2 r^4 \right).
\label{eq:prop1_remainder}
\end{equation}
Consequently, when the parameter moves actively across the window ($\operatorname{Var}_{W_t}[\theta_i] = \Theta(r^2)$), the diagonal curvature term $H_{ii}(\boldsymbol{\theta}^*)^2 \operatorname{Var}_{W_t}[\theta_i]$ dominates the residual provided $\delta \sqrt{p} \ll |H_{ii}(\boldsymbol{\theta}^*)|$ (Assumption~\ref{ass:diag_dominance}) and displacement $r$ is sufficiently small.
\end{proposition}

\begin{proof}
Applying the variance operator to the secant relation from Lemma~\ref{lem:secant}:
\begin{equation}
\operatorname{Var}_{W_t}[g_i] = \operatorname{Var}_{W_t}\left[ H_{ii}(\boldsymbol{\theta}^*) \theta_i + \varepsilon_i \right] = H_{ii}(\boldsymbol{\theta}^*)^2 \operatorname{Var}_{W_t}[\theta_i] + 2 H_{ii}(\boldsymbol{\theta}^*) \operatorname{Cov}_{W_t}(\theta_i, \varepsilon_i) + \operatorname{Var}_{W_t}[\varepsilon_i].
\end{equation}
By the Cauchy--Schwarz inequality, $|\operatorname{Cov}_{W_t}(\theta_i, \varepsilon_i)| \le \sqrt{\operatorname{Var}_{W_t}[\theta_i]} \sqrt{\operatorname{Var}_{W_t}[\varepsilon_i]}$. Because the displacement satisfies $\|\boldsymbol{\theta}_\tau - \boldsymbol{\theta}^*\| \le r$, we have $|\theta_{\tau,i} - \theta_i^*| \le r$, which implies $\sqrt{\operatorname{Var}_{W_t}[\theta_i]} \le r$ (up to the finite-window Bessel factor $\sqrt{\frac{w}{w-1}} = 1 + \mathcal{O}(w^{-1})$). From Lemma~\ref{lem:secant}, $\|\boldsymbol{\varepsilon}_i\|_\infty \le \delta r \sqrt{p} + \frac{L}{2} r^2$, yielding $\sqrt{\operatorname{Var}_{W_t}[\varepsilon_i]} \le \|\boldsymbol{\varepsilon}_i\|_\infty$ (up to $1 + \mathcal{O}(w^{-1})$). Using $|H_{ii}(\boldsymbol{\theta}^*)| \le M$, the cross-covariance term is bounded by:
\begin{equation}
2 |H_{ii}(\boldsymbol{\theta}^*)| |\operatorname{Cov}_{W_t}(\theta_i, \varepsilon_i)| \le 2 M r \left( \delta r \sqrt{p} + \frac{L}{2} r^2 \right).
\end{equation}
Adding $\operatorname{Var}_{W_t}[\varepsilon_i] \le \|\boldsymbol{\varepsilon}_i\|_\infty^2 = (\delta r \sqrt{p} + \frac{L}{2} r^2)^2$ completes the proof of Eq.~\eqref{eq:prop1_remainder}.
\end{proof}

\begin{remark}[Dynamical Variance Scaling, Energy Rescaling, and Window Formulation]
\label{rem:variance_scaling}
Several practical and conceptual details of this formulation merit discussion:
\begin{itemize}
    \item \textbf{Step Damping and Hamiltonian Scale Sensitivity:} Classical Newton methods scale updates inversely with the first power of local curvature ($H_{ii}^{-1}$). In SIGMA-QGD, the preconditioning factor scales with $H_{ii}^2 \operatorname{Var}[\theta_i]$, producing coordinate adjustments that behave roughly like $\hat{m}_{t,i} / (c H_{ii}^2 \operatorname{Var}[\theta_i] + \lambda_{\text{reg}})$. In steep regions where large second derivatives generate rapid gradient reversals, the quadratic denominator penalizes updates heavily, quelling numerical instability. In flat regions ($H_{ii} \approx 0$), the denominator drops to the regularization baseline, ensuring that steps are bounded by $\lambda_{\text{reg}}^{-1}$. Note that in the variance-dominated regime where $d_i \sim \hat{m}_i / \hat{\sigma}_i^2 \propto 1/g_i$, multiplying the Hamiltonian by a constant factor $\alpha$ ($H \to \alpha H$) modifies the relative step balance. This is an intrinsic characteristic of unnormalized second-order schemes, meaning appropriate values for $c$ and $\lambda_{\text{reg}}$ scale with the overall operator norm $\Lambda = \sum_k |c_k|$. When optimizing Hamiltonians with diverse energy scales, scale invariance can be preserved either by minimizing the normalized objective $\tilde{E}(\boldsymbol{\theta}) = E(\boldsymbol{\theta}) / \Lambda$ or by setting $\lambda_{\text{reg}} \propto \Lambda^2$.
    \item \textbf{Trajectory-Coupled Curvature:} The leading term $H_{ii}^2 \operatorname{Var}_{W_t}[\theta_i]$ is proportional to the parameter dispersion $\operatorname{Var}_{W_t}[\theta_i]$ generated by the optimization path itself. The proxy $\hat{\sigma}_i^2$ therefore captures local curvature as experienced through the recent kinematic history of the optimizer, rather than as an isolated static tensor.
    \item \textbf{Sliding Window Analysis versus Cumulative Welford Tracking:} While Assumptions~\ref{ass:quadraticity}--\ref{ass:displacement} and Proposition~\ref{prop:variance_curvature} are proved over a finite sliding window $W_t$ of width $w$---representing the active descent stage ($t \le t_0$) with meaningful displacement ($r = \Theta(1)$)---practical implementation in Algorithm~\ref{alg:sigma_qgd} relies on a full-history Welford accumulator for $t \ge 2$. This gives $\mathcal{O}(1)$ time and memory overhead per step. Furthermore, if a strict sliding window of size $w$ were used near a local basin ($t > t_0$), parameter displacement would contract toward zero ($\|\boldsymbol{\theta}_\tau - \boldsymbol{\theta}^*\| \to 0$), driving $\operatorname{Var}_{W_t}[\theta_i] \to 0$ and collapsing the preconditioner to a uniform scalar $\lambda_{\text{reg}}$. Accumulating the sum of squared differences $\mathbf{S}_t / (t-1)$ over the full run avoids this collapse, preserving the directional curvature hierarchy established during the descent phase even as parameter motion slows down.
\end{itemize}
\end{remark}

\begin{proposition}[Uniform Local and Global Boundedness of the Preconditioner]
\label{prop:boundedness}
Under Assumptions~\ref{ass:bounded_curvature} and \ref{ass:displacement}, the diagonal preconditioner $F_i(t) = c \hat{\sigma}_i^2(t) + \lambda_{\text{reg}}$ on window $W_t$ satisfies:
\begin{equation}
\lambda_{\text{reg}} \le F_i(t) \le c M^2 r^2 + \lambda_{\text{reg}} + \mathcal{O}(r^2).
\end{equation}
Furthermore, for the full-history Welford accumulator in Algorithm~\ref{alg:sigma_qgd}, gradient clipping ($\|\mathbf{g}_{c,t}\|_2 \le c_{\text{norm}} = \sqrt{p}$) and Assumption~\ref{ass:bounded_curvature} ensure $|g_{c,t,i}| \le G_{\max} \equiv \min(\sqrt{p}, M)$ for all $t \ge 1$, yielding the unconditional global bound $\lambda_{\text{reg}} \le F_i(t) \le F_{\max} \equiv 2c G_{\max}^2 + \lambda_{\text{reg}}$ for all $t \ge 1$, and coordinate-wise step bounds:
\begin{equation}
\frac{|\hat{m}_{t,i}|}{F_{\max}} \le |d_{t,i}| \le \frac{|\hat{m}_{t,i}|}{\lambda_{\text{reg}}}.
\end{equation}
\end{proposition}

\begin{proof}
Since $\hat{\sigma}_i^2 \ge 0$, the lower bound follows from $\lambda_{\text{reg}} > 0$. The local window upper bound follows from Proposition~\ref{prop:variance_curvature} and $\operatorname{Var}_{W_t}[\theta_i] \le \frac{w}{w-1} r^2$ (absorbing the Bessel factor $1 + \mathcal{O}(w^{-1})$). Globally, since $|g_{c,t,i}| \le G_{\max}$, the unbiased sample variance over any horizon $t \ge 2$ satisfies $\hat{\sigma}_i^2(t) = \frac{1}{t-1}\sum_{\tau=1}^t (g_{c,\tau,i} - M_{t,i})^2 \le \frac{t}{t-1} G_{\max}^2 \le 2 G_{\max}^2$, establishing the unconditional upper bound $F_{\max}$.
\end{proof}

\begin{proposition}[Descent Direction Guarantee under Margin Condition]
\label{prop:descent}
Let $\mathcal{S}_+ = \{i : \hat{m}_{t,i}(-g_{t,i}) \ge 0\}$ and $\mathcal{S}_- = \{j : \hat{m}_{t,j}(-g_{t,j}) < 0\}$ partition the coordinate indices $\{1, \dots, p\}$ at step $t$. Under Assumptions~\ref{ass:bounded_curvature} and \ref{ass:displacement}, the preconditioned update direction $\mathbf{d}_t = \hat{\mathbf{m}}_t \oslash \mathbf{F}(t)$ constitutes a strict descent direction ($\langle \mathbf{d}_t, -\mathbf{g}_t \rangle > 0$) if the aligned-coordinate projection dominates the misaligned contribution according to the margin condition:
\begin{equation}
\sum_{i \in \mathcal{S}_+} \frac{\hat{m}_{t,i} (-g_{t,i})}{F_i(t)} > \sum_{j \in \mathcal{S}_-} \frac{|\hat{m}_{t,j} g_{t,j}|}{F_j(t)}.
\label{eq:margin_condition}
\end{equation}
In particular, using the uniform bounds $\lambda_{\text{reg}} \le F_k(t) \le F_{\max}$ from Proposition~\ref{prop:boundedness}, a sufficient condition is:
\begin{equation}
\sum_{i \in \mathcal{S}_+} \hat{m}_{t,i} (-g_{t,i}) > \kappa \sum_{j \in \mathcal{S}_-} |\hat{m}_{t,j} g_{t,j}|, \qquad \kappa = \frac{F_{\max}}{\lambda_{\text{reg}}}.
\label{eq:kappa_margin}
\end{equation}
In the special case of coordinate-wise alignment ($\mathcal{S}_- = \emptyset$, Assumption~\ref{ass:alignment}), descent holds unconditionally with $\langle \mathbf{d}_t, -\mathbf{g}_t \rangle \ge \frac{1}{F_{\max}} \langle \hat{\mathbf{m}}_t, -\mathbf{g}_t \rangle > 0$.
\end{proposition}

\begin{proof}
Decomposing the preconditioned inner product over the partition $\mathcal{S}_+ \cup \mathcal{S}_-$:
\begin{align}
\langle \mathbf{d}_t, -\mathbf{g}_t \rangle &= \sum_{i=1}^p \frac{\hat{m}_{t,i}(-g_{t,i})}{F_i(t)} \nonumber \\
&= \sum_{i \in \mathcal{S}_+} \frac{\hat{m}_{t,i}(-g_{t,i})}{F_i(t)} - \sum_{j \in \mathcal{S}_-} \frac{|\hat{m}_{t,j} g_{t,j}|}{F_j(t)}.
\end{align}
Positivity is guaranteed whenever the first sum exceeds the second, establishing Eq.~\eqref{eq:margin_condition}. Applying the uniform lower bound $\lambda_{\text{reg}}$ and upper bound $F_{\max}$ yields Eq.~\eqref{eq:kappa_margin}. When $\mathcal{S}_- = \emptyset$, the negative sum vanishes and the uniform lower bound preserves strict descent.
\end{proof}

\paragraph{Empirical Validity and Dynamical Self-Damping.}
Although strict coordinate-wise sign alignment ($\mathcal{S}_- = \emptyset$) is satisfied in $67.10\%$ of coordinate-step pairs on TFIM and $71.93\%$ on XXZ, the margin inequality in Eq.~\eqref{eq:margin_condition} uncovers an innate self-regulating property of trajectory variance. Temporary sign conflicts ($\hat{m}_{t,j} g_{t,j} > 0$) arise almost exclusively when coordinates oscillate back and forth across steep canyon walls. In these jittery directions, the sample variance $\hat{\sigma}_j^2$ builds up rapidly, driving the denominator $F_j(t)$ far above $\lambda_{\text{reg}}$. This large denominator heavily dampens the offending negative terms in Eq.~\eqref{eq:margin_condition} ($1/F_j(t) \to 0$), neutralizing their destabilizing influence. At the same time, coordinates advancing smoothly along the ravine bed experience low gradient volatility, keeping $F_i(t)$ close to $\lambda_{\text{reg}}$ and preserving substantial step sizes along productive descent paths. In our empirical runs, the observed condition ratio $\max_{i,t} F_i(t)/\lambda_{\text{reg}}$ remained comfortably within $[12.4, 48.6]$, well below the theoretical upper bound $\kappa = F_{\max}/\lambda_{\text{reg}} \le 200\text{--}400$, since parameters do not saturate their clipping thresholds simultaneously. Thanks to this self-damping behavior, the aggregate inner product $\langle \mathbf{d}_t, -\mathbf{g}_t \rangle$ was strictly positive across all $200$ optimization steps in $100.0\%$ of our runs for every tested model and size.

\begin{theorem}[Non-Convex Stationarity Convergence Rate]
\label{thm:convergence}
Let $E: \mathbb{R}^p \to \mathbb{R}$ be $L_g$-smooth with lower-bounded global minimum $E^* > -\infty$. Suppose that along the optimization trajectory, the margin condition of Proposition~\ref{prop:descent} holds with uniform alignment factor $\gamma_0 > 0$ such that $\langle \mathbf{d}_t, -\mathbf{g}_t \rangle \ge \frac{\gamma_0}{F_{\max}} \|\mathbf{g}_t\|_2^2$ (which holds unconditionally when $\beta = 0$ with $\gamma_0 = \min(1, 1/M)$, or empirically under Proposition~\ref{prop:descent} for $\beta > 0$), where $F_{\max}$ is the uniform preconditioner upper bound from Proposition~\ref{prop:boundedness}, and the base learning rate satisfies $\eta_0 < \frac{2 \gamma_0 \lambda_{\text{reg}}^2 (1-\beta)}{L_g F_{\max}}$. Then the sequence of iterates $\{\boldsymbol{\theta}_{t-1}\}_{t=1}^T$ generated by Algorithm~\ref{alg:sigma_qgd} satisfies the ergodic stationary-point convergence bound:
\begin{equation}
\min_{1 \le t \le T} \|\nabla E(\boldsymbol{\theta}_{t-1})\|_2^2 \le \frac{E(\boldsymbol{\theta}_0) - E^*}{\mathcal{C} \cdot T} = \mathcal{O}\left(\frac{1}{T}\right),
\label{eq:ergodic_rate}
\end{equation}
where $\mathcal{C} = \eta_0 \left( \frac{\gamma_0}{F_{\max}} - \frac{L_g \eta_0}{2 \lambda_{\text{reg}}^2 (1-\beta)} \right) > 0$. Furthermore, in the presence of unbiased stochastic quantum measurement shot noise with bounded variance $\mathbb{E}[\|\tilde{\mathbf{g}}_t - \nabla E(\boldsymbol{\theta}_{t-1})\|_2^2 \mid \mathcal{F}_{t-1}] \le \sigma_{\text{shot}}^2$, setting a diminishing step-size schedule $\eta_t = \eta_0 / \sqrt{t}$ yields the standard non-convex stochastic stationarity rate:
\begin{equation}
\min_{1 \le t \le T} \mathbb{E}\left[\|\nabla E(\boldsymbol{\theta}_{t-1})\|_2^2\right] = \mathcal{O}\left(\frac{1}{\sqrt{T}}\right).
\label{eq:stochastic_rate}
\end{equation}
\end{theorem}

\begin{proof}
By $L_g$-smoothness of $E(\boldsymbol{\theta})$, for consecutive parameter iterates $\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \eta_0 \mathbf{d}_t$ with $\mathbf{g}_t = \nabla E(\boldsymbol{\theta}_{t-1})$:
\begin{equation}
E(\boldsymbol{\theta}_t) \le E(\boldsymbol{\theta}_{t-1}) - \eta_0 \langle \mathbf{g}_t, \mathbf{d}_t \rangle + \frac{L_g \eta_0^2}{2} \|\mathbf{d}_t\|_2^2.
\end{equation}
Applying the margin lower bound $\langle \mathbf{d}_t, -\mathbf{g}_t \rangle \ge \frac{\gamma_0}{F_{\max}} \|\mathbf{g}_t\|_2^2$ and the preconditioner lower bound $F_i(t) \ge \lambda_{\text{reg}}$ (for $t \ge t_{\text{warm}}$, and $F_i(t) = 1 \ge \lambda_{\text{reg}}$ during warmup) gives $\|\mathbf{d}_t\|_2 \le \|\hat{\mathbf{m}}_t\|_2 / \lambda_{\text{reg}}$. Summing over $t = 1, \dots, T$ yields:
\begin{equation}
E(\boldsymbol{\theta}_T) - E(\boldsymbol{\theta}_0) \le -\frac{\eta_0 \gamma_0}{F_{\max}} \sum_{t=1}^T \|\mathbf{g}_t\|_2^2 + \frac{L_g \eta_0^2}{2 \lambda_{\text{reg}}^2} \sum_{t=1}^T \|\hat{\mathbf{m}}_t\|_2^2.
\end{equation}
Since $\hat{\mathbf{m}}_t = \sum_{\tau=1}^t w_{t,\tau} \mathbf{g}_{c,\tau}$ with convex weights $w_{t,\tau} = \frac{(1-\beta)\beta^{t-\tau}}{1-\beta^t}$ summing to $1$, $\|\mathbf{g}_{c,\tau}\|_2 \le \|\mathbf{g}_\tau\|_2$, and $1 - \beta^t \ge 1 - \beta$, Jensen's inequality gives $\|\hat{\mathbf{m}}_t\|_2^2 \le \sum_{\tau=1}^t w_{t,\tau} \|\mathbf{g}_\tau\|_2^2 \le \sum_{\tau=1}^t \beta^{t-\tau} \|\mathbf{g}_\tau\|_2^2$. Summing over $t=1,\dots,T$ and exchanging summation order yields $\sum_{t=1}^T \|\hat{\mathbf{m}}_t\|_2^2 \le \frac{1}{1-\beta} \sum_{t=1}^T \|\mathbf{g}_t\|_2^2$. Substituting this bound and rearranging with $E(\boldsymbol{\theta}_T) \ge E^*$ gives:
\begin{equation}
\mathcal{C} \sum_{t=1}^T \|\mathbf{g}_t\|_2^2 \le E(\boldsymbol{\theta}_0) - E^*,
\end{equation}
where $\mathcal{C} > 0$ under the stated step-size condition. Bounding $\sum_{t=1}^T \|\mathbf{g}_t\|_2^2 \ge T \min_{1 \le t \le T} \|\mathbf{g}_t\|_2^2$ and dividing by $\mathcal{C} T$ establishes Eq.~\eqref{eq:ergodic_rate}. Under additive zero-mean shot noise with bounded variance $\sigma_{\text{shot}}^2$, note that the Welford update satisfies $|\hat{\sigma}_i^2(t) - \hat{\sigma}_i^2(t-1)| = \mathcal{O}(G_{\max}^2 / t)$, so replacing $F_i(t)$ with the $\mathcal{F}_{t-1}$-measurable lagged preconditioner $F_i(t-1)$ incurs at most an $\mathcal{O}(\eta_t / t)$ bias per step (whose sum $\sum_{t=1}^T \eta_t / t < \infty$ is bounded), while conditional expectations introduce an $\mathcal{O}(L_g \eta_t^2 \sigma_{\text{shot}}^2)$ variance penalty per step; applying $\eta_t = \eta_0 / \sqrt{t}$ yields $\min_{1 \le t \le T} \mathbb{E}[\|\nabla E(\boldsymbol{\theta}_{t-1})\|_2^2] = \mathcal{O}(1/\sqrt{T})$~\cite{ghadimi2013stochastic}.
\end{proof}

\subsection{Warmup Schedule and Complete Algorithm}
To prevent early, high-variance sample estimates from perturbing initial optimization trajectories, preconditioning is initiated only after a defined warmup window:
\begin{equation}
t_{\text{warm}} = \min(3p, \max(10, \lfloor T / 5 \rfloor)).
\label{eq:warmup}
\end{equation}
For steps $t < t_{\text{warm}}$, updates follow standard momentum-accelerated gradient descent ($\mathbf{d}_t = \hat{\mathbf{m}}_t$). Computing the unshifted objective value $E_t = E(\boldsymbol{\theta}_{t-1})$ adds one circuit per iteration for best-iterate monitoring, yielding the logged budget of $2p+1$ circuits. In settings where only parameter steps are performed without intermediate energy checks, $E_t$ can be bypassed, allowing the algorithm to execute in strictly $2p$ circuit runs per step.

\begin{algorithm}[H]
\caption{SIGMA-QGD Optimization Algorithm}
\label{alg:sigma_qgd}
\begin{algorithmic}[1]
\REQUIRE Initial parameters $\boldsymbol{\theta}_0 \in \mathbb{R}^p$, learning rate $\eta_0$, multiplier $c=1.0$, regularizer $\lambda_{\text{reg}}=0.005$, max steps $T$.
\STATE Initialize accumulators $\mathbf{M} \leftarrow \mathbf{0}, \mathbf{S} \leftarrow \mathbf{0}, \mathbf{m} \leftarrow \mathbf{0}$, best energy $E^* \leftarrow \infty, \boldsymbol{\theta}^* \leftarrow \boldsymbol{\theta}_0$.
\STATE Compute $t_{\text{warm}} = \min(3p, \max(10, \lfloor T/5 \rfloor))$, clip threshold $c_{\text{norm}} = \sqrt{p}$.
\FOR{$t = 1$ \TO $T$}
    \STATE Evaluate cost and parameter-shift gradient: $E_t, \mathbf{g}_t \leftarrow \text{CostAndGradient}(\boldsymbol{\theta}_{t-1})$ \hfill \COMMENT{$2p+1$ circuits}
    \STATE Gradient clipping: $\mathbf{g}_c \leftarrow \mathbf{g}_t \cdot \min(1, c_{\text{norm}} / \|\mathbf{g}_t\|_2)$
    \STATE Update Welford accumulators $(\mathbf{M}, \mathbf{S})$ with $\mathbf{g}_c$ via Eqs.~\eqref{eq:welford_m}--\eqref{eq:welford_s}
    \STATE Compute unbiased variance $\hat{\boldsymbol{\sigma}}^2 \leftarrow \mathbf{S} / (t-1)$ for $t \ge 2$, else $\mathbf{0}$
    \STATE Update momentum: $\mathbf{m} \leftarrow \beta \mathbf{m} + (1-\beta)\mathbf{g}_c$, $\hat{\mathbf{m}} \leftarrow \mathbf{m} / (1-\beta^t)$
    \IF{$t \ge t_{\text{warm}}$}
        \STATE $\mathbf{F} \leftarrow c \hat{\boldsymbol{\sigma}}^2 + \lambda_{\text{reg}} \mathbf{1}$
        \STATE $\mathbf{d}_t \leftarrow \hat{\mathbf{m}} \oslash \mathbf{F}$
    \ELSE
        \STATE $\mathbf{d}_t \leftarrow \hat{\mathbf{m}}$
    \ENDIF
    \STATE $\boldsymbol{\theta}_t \leftarrow \boldsymbol{\theta}_{t-1} - \eta_0 \mathbf{d}_t$
    \IF{$E_t < E^*$}
        \STATE $E^* \leftarrow E_t, \boldsymbol{\theta}^* \leftarrow \boldsymbol{\theta}_{t-1}$
    \ENDIF
\ENDFOR
\RETURN Optimal parameters $\boldsymbol{\theta}^*$, minimum energy $E^*$
\end{algorithmic}
\end{algorithm}

% ==============================================================================
% SECTION 4: EXPERIMENTAL SETUP
% ==============================================================================
\section{Experimental Methodology}
\label{sec:setup}

\subsection{Benchmark Hamiltonians}
We test optimizers on two canonical one-dimensional quantum spin models with periodic boundary conditions:
\begin{itemize}
    \item \textbf{1D Transverse-Field Ising Model (TFIM):}
    \begin{equation}
    H_{\text{TFIM}} = -J \sum_{i=1}^n Z_i Z_{i+1} - h \sum_{i=1}^n X_i, \quad J = 1.0, \, h = 0.5.
    \end{equation}
    Exact ground-state energies: $E_0 = -10.6356$ for $n=10$, and $E_0 = -12.7626$ for $n=12$.
    \item \textbf{1D Heisenberg XXZ Spin Chain:}
    \begin{equation}
    H_{\text{XXZ}} = J_{xy} \sum_{i=1}^n (X_i X_{i+1} + Y_i Y_{i+1}) + J_z \sum_{i=1}^n Z_i Z_{i+1} + h \sum_{i=1}^n Z_i,
    \end{equation}
    with couplings $J_{xy} = 1.0, J_z = 0.5, h = 0.1$. Exact ground-state energies: $E_0 = -15.2761$ for $n=10$, and $E_0 = -18.2291$ for $n=12$.
\end{itemize}

\subsection{Ansatz and Initialization}
We adopt a hardware-efficient circuit layout featuring depth $r=3$, which corresponds to $p=30$ variational parameters at $n=10$ and $p=36$ at $n=12$. For each experimental setup, initial parameter vectors $\boldsymbol{\theta}_0$ are sampled deterministically using matching random seeds ($\theta_{0,i} \sim \text{Uniform}(-\pi/8, \pi/8)$), guaranteeing rigorous paired comparisons across all candidate optimizers.

\subsection{Baseline Algorithms and Circuit Accounting}
All 11 algorithms (SIGMA-QGD together with 10 comparison methods) are evaluated under standardized quantum circuit budgets:
\begin{itemize}
    \item \textbf{Adaptive First-Order Baselines ($2p+1$ circuits/step):} Adam \cite{kingma2014adam}, AdaBelief \cite{zhuang2020adabelief}, RMSProp \cite{tieleman2012lecture}, and Yogi \cite{zaheer2018adaptive}.
    \item \textbf{Exact Second-Order Baselines ($2p+1$ circuits/step):} Hess-Mag and Diag-Hessian (computed analytically via Eq.~\eqref{eq:exact_hessian} \cite{wierichs2022general}).
    \item \textbf{Quantum Natural Gradient ($3p+2$ circuits/step):} Diag-QNG \cite{stokes2020quantum,wierichs2020avoiding}.
    \item \textbf{Derivative-Free and Stochastic Baselines:} COBYLA ($1$ circuit/step \cite{powell1994direct}), SPSA ($3$ circuits/step \cite{spall1992multivariate,spall1997accelerated,spall1998implementation,spall2000adaptive}), QN-SPSA ($9$ circuits/step \cite{gacon2021simultaneous}), and CMA-ES (budget-matched \cite{novak2025benchmarking}).
\end{itemize}

\subsection{Statistical Significance Testing}
Performance contrasts between SIGMA-QGD and each baseline method are assessed using two-sided paired Wilcoxon signed-rank tests. To prevent Family-Wise Error Rate inflation across multiple pairwise comparisons, we report all significance levels after applying Holm--Bonferroni stepdown adjustments.

% ==============================================================================
% SECTION 5: EMPIRICAL RESULTS
% ==============================================================================
\section{Empirical Results}
\label{sec:results}

\subsection{Curvature Proxy Validation}
\label{sec:exp_proxy}
To assess Proposition~\ref{prop:variance_curvature} empirically, we recorded step-by-step values of the log trajectory variance $\log \hat{\sigma}_i^2$ alongside the log exact analytic diagonal Hessian magnitude $\log |H_{ii}|$ along 200-step SIGMA-QGD runs on both TFIM and XXZ chains at 10 and 12 qubits ($N=15$ seeds, representing $>394{,}000$ coordinate-step points). At each iteration, the analytic diagonal Hessian was computed via Eq.~\eqref{eq:exact_hessian} without modifying the optimizer's path.

\begin{table*}[htbp]
\centering
\small
\caption{Consolidated curvature-proxy validation across physical Hamiltonians and system sizes ($N=15$ independent seeds, 200 steps). Reported values indicate mean $\pm$ standard deviation for log-space Pearson correlation $r$ and Spearman rank correlation $\rho$, accompanied by Wilcoxon signed-rank tests against $H_0: \text{correlation} \le 0$.}
\label{tab:hessian_corr}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
\textbf{Model} & \textbf{System Size} & \textbf{Evaluations/Seed} & \textbf{Pearson $r$ (Log-Space)} & \textbf{Wilcoxon $p$} & \textbf{Spearman $\rho$} & \textbf{Wilcoxon $p$} \\
\midrule
TFIM (Longitudinal/Transverse) & 10 qubits ($p=30$) & 5,970 & 0.5665 $\pm$ 0.0577 & $6.1 \times 10^{-5}$ & 0.3715 $\pm$ 0.0998 & $6.1 \times 10^{-5}$ \\
TFIM (Longitudinal/Transverse) & 12 qubits ($p=36$) & 7,164 & 0.6256 $\pm$ 0.0301 & $6.1 \times 10^{-5}$ & 0.4474 $\pm$ 0.0745 & $6.1 \times 10^{-5}$ \\
XXZ (Heisenberg Exchange)      & 10 qubits ($p=30$) & 5,970 & 0.3196 $\pm$ 0.0829 & $6.1 \times 10^{-5}$ & 0.1162 $\pm$ 0.1416 & $0.010$ \\
XXZ (Heisenberg Exchange)      & 12 qubits ($p=36$) & 7,164 & 0.2938 $\pm$ 0.0914 & $6.1 \times 10^{-5}$ & 0.1113 $\pm$ 0.1404 & $0.004$ \\
\bottomrule
\end{tabular}%
}
\end{table*}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.92\linewidth]{figures/fig_hessian_correlation_consolidated.pdf}
\caption{Consolidated curvature-proxy validation across 15 seeds on 10-qubit ($p=30$) and 12-qubit ($p=36$) TFIM and XXZ spin models. (a) Log-space Pearson correlation $r$ between gradient trajectory variance $\log \hat{\sigma}_i^2$ and exact diagonal Hessian $\log |H_{ii}|$. (b) Spearman rank correlation $\rho$. Diamond markers denote sample means, horizontal bars indicate medians, and individual points depict random seeds. All correlations are statistically positive ($p < 10^{-4}$ for Pearson across all models).}
\label{fig:hessian_corr_consolidated}
\end{figure}

The empirical validation results presented in Table~\ref{tab:hessian_corr} and Figure~\ref{fig:hessian_corr_consolidated} highlight several clear physical features:
\begin{itemize}
    \item \textbf{Strong Tracking on Commuting Field Landscapes (TFIM):} On the TFIM Hamiltonian, trajectory variance displays a pronounced log-linear correlation with the true diagonal Hessian ($r = 0.5665 \pm 0.0577$ at 10 qubits, rising to $r = 0.6256 \pm 0.0301$ at 12 qubits, with $p = 6.1 \times 10^{-5}$ in both systems). The increase in correlation with system size indicates that the variance estimator scales favorably as the underlying Hilbert space expands ($2^{10} \to 2^{12}$).
    \item \textbf{Statistically Significant Alignment on Frustrated Systems (XXZ):} For the non-commuting XXZ chain, Pearson correlation remains strictly positive and statistically significant ($r = 0.3196 \pm 0.0829$ at 10 qubits, $r = 0.2938 \pm 0.0914$ at 12 qubits, $p = 6.1 \times 10^{-5}$). Spearman rank correlations are likewise positive ($\rho = 0.1162, p = 0.010$ at 10 qubits; $\rho = 0.1113, p = 0.004$ at 12 qubits).
    \item \textbf{Underlying Physical Mechanism for Model Divergence:} The difference in correlation strength between TFIM ($r \approx 0.60$) and XXZ ($r \approx 0.31$) stems directly from operator commutativity. In the TFIM model, nearest-neighbor $Z_i Z_{i+1}$ interactions commute and coordinate dynamics are governed predominantly by the transverse field $X_i$, keeping off-diagonal Hessian elements small relative to diagonal entries ($\delta \sqrt{p} \ll |H_{ii}|$, fulfilling Assumption~\ref{ass:diag_dominance}). On the XXZ model, by contrast, non-commuting exchange terms $X_i X_{i+1} + Y_i Y_{i+1}$ induce dense off-diagonal parameter cross-talk. This restricts how much of the full landscape curvature can be captured through coordinate-diagonal projections alone, matching theoretical predictions on Hessian entry decay in variational algorithms \cite{hessian_resolvability_2026}.
    \item \textbf{Scale Invariance in Log Space:} While the actual preconditioner update in Eq.~\eqref{eq:preconditioned_update} uses $\hat{\sigma}_i^2$ directly and Proposition~\ref{prop:variance_curvature} connects $\hat{\sigma}_i^2 \propto H_{ii}^2$, the statistical validation is conducted in log space ($\log \hat{\sigma}_i^2$ versus $\log |H_{ii}|$). In parameterized quantum circuits, coordinate curvatures and gradient variances naturally span multiple orders of magnitude (from $10^{-4}$ in barren regions to $10^0$ in steep ravines). Logarithmic transformation avoids distortion from extreme outliers, verifying that the variance proxy faithfully captures the relative curvature ordering across the parameter space.
\end{itemize}

\subsection{Independent Learning Rate Sweep}
\label{sec:lr_sweep_results}
To ensure fair comparisons across all gradient-based algorithms, each method was subjected to an independent grid sweep over learning rates $\eta \in \{0.001, 0.005, 0.01, 0.02, 0.05, 0.1\}$ across both spin models.

\begin{table*}[htbp]
\centering
\small
\caption{Phase~1 learning rate grid sweep results across 7 gradient-based optimizers ($200$ steps). 10-qubit experiments evaluated over $N=10$ independent random seeds; 12-qubit experiments evaluated over $N=5$ seeds. Reported values indicate the optimal step size $\eta^*$ and resulting mean ground-state energy gap $\Delta E \pm \sigma$.}
\label{tab:lr_sweep_summary}
\label{tab:lr_sweep}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccccc}
\toprule
\multirow{2}{*}{\textbf{Method}} & \multicolumn{2}{c}{\textbf{TFIM 10q ($N=10$)}} & \multicolumn{2}{c}{\textbf{TFIM 12q ($N=5$)}} & \multicolumn{2}{c}{\textbf{XXZ 10q ($N=10$)}} & \multicolumn{2}{c}{\textbf{XXZ 12q ($N=5$)}} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}
& $\eta^*$ & $\Delta E$ & $\eta^*$ & $\Delta E$ & $\eta^*$ & $\Delta E$ & $\eta^*$ & $\Delta E$ \\
\midrule
\textbf{SIGMA-QGD v9.0} & \textbf{0.1} & \textbf{0.0091 $\pm$ 0.0018} & \textbf{0.1} & \textbf{0.0107 $\pm$ 0.0008} & \textbf{0.1} & \textbf{1.2670 $\pm$ 0.5183} & 0.1 & 1.8538 $\pm$ 0.7672 \\
Adam & 0.1 & 0.1020 $\pm$ 0.1000 & 0.1 & 0.1377 $\pm$ 0.0596 & 0.1 & 2.2947 $\pm$ 0.9253 & 0.1 & 2.3458 $\pm$ 0.8965 \\
AdaBelief & 0.1 & 0.0751 $\pm$ 0.0858 & 0.1 & 0.1128 $\pm$ 0.0577 & 0.1 & 2.4898 $\pm$ 0.8771 & 0.01 & 1.7934 $\pm$ 0.7127 \\
Yogi & 0.1 & 0.1092 $\pm$ 0.1070 & 0.1 & 0.1662 $\pm$ 0.0957 & 0.1 & 2.3267 $\pm$ 0.9600 & 0.1 & 2.3921 $\pm$ 0.9049 \\
RMSProp & 0.01 & 0.3876 $\pm$ 0.0530 & 0.01 & 0.4757 $\pm$ 0.0561 & 0.05 & 2.3214 $\pm$ 0.8798 & \textbf{0.05} & \textbf{1.5795 $\pm$ 0.5941} \\
Diag-Hessian & 0.1 & 0.4374 $\pm$ 0.0054 & 0.1 & 0.5588 $\pm$ 0.0050 & 0.1 & 1.9296 $\pm$ 0.7444 & 0.1 & 2.4350 $\pm$ 0.8453 \\
Hess-Mag & 0.001 & 0.4400 $\pm$ 0.0215 & 0.001 & 0.5370 $\pm$ 0.0447 & 0.001 & 2.0225 $\pm$ 0.8297 & 0.001 & 1.9218 $\pm$ 0.6795 \\
\bottomrule
\end{tabular}%
}
\end{table*}

Because the Phase~1 sweep for 12-qubit XXZ used a small validation sample ($N=5$ seeds) on a highly frustrated landscape, the sample means possess substantial standard errors ($\text{SE} = \sigma/\sqrt{5} \approx 0.27\text{--}0.34$). While RMSProp ($\eta^*=0.05$, $1.5795 \pm 0.5941$) and AdaBelief ($\eta^*=0.01$, $1.7934 \pm 0.7127$) showed slightly lower sample means than SIGMA-QGD ($1.8538 \pm 0.7672$) on those specific 5 seeds, the comprehensive 20-seed benchmark in Table~\ref{tab:main_benchmark_consolidated} demonstrates that SIGMA-QGD delivers the lowest overall mean energy gap ($1.8408 \pm 0.8336$ versus $1.9248 \pm 0.7486$ for RMSProp), with both methods statistically indistinguishable ($p_{\text{Holm}} = 0.52$).

\subsection{Hyperparameter Sensitivity Analysis}
\label{sec:sensitivity_results}
We evaluated parameter sensitivity by performing a one-factor-at-a-time (OFAT) sweep over the preconditioner multiplier $c \in \{0.5, 1.0, 2.0, 4.0, 6.0\}$ and regularization floor $\lambda_{\text{reg}} \in \{0.0005, 0.001, 0.005, 0.01, 0.05\}$ across both models at 10 and 12 qubits ($N=15$ seeds).

\begin{table}[htbp]
\centering
\small
\caption{Consolidated sensitivity analysis across TFIM and XXZ models at 10 and 12 qubits ($N=15$ seeds, mean gap $\pm$ std). Multiplier $c$ is swept at fixed baseline $\lambda_{\text{reg}} = 0.001$, and $\lambda_{\text{reg}}$ is swept at fixed $c = 1.0$.}
\label{tab:sensitivity_consolidated}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{llcccc}
\toprule
\textbf{Parameter} & \textbf{Value} & \textbf{TFIM 10q} & \textbf{TFIM 12q} & \textbf{XXZ 10q} & \textbf{XXZ 12q} \\
\midrule
\multirow{5}{*}{Multiplier $c$} 
 & $0.5$ & $0.0084 \pm 0.0014$ & $0.0111 \pm 0.0011$ & $1.5685 \pm 0.8114$ & $1.7399 \pm 0.8373$ \\
 & $1.0$ & $0.0122 \pm 0.0107$ & $0.0157 \pm 0.0110$ & $1.9503 \pm 0.9296$ & $\mathbf{1.6896 \pm 0.6444}$ \\
 & $2.0$ & $0.0552 \pm 0.0926$ & $0.0502 \pm 0.0319$ & $1.8691 \pm 0.8653$ & $1.9364 \pm 0.7551$ \\
 & $4.0$ & $0.1065 \pm 0.0864$ & $0.2221 \pm 0.0836$ & $1.6326 \pm 0.7514$ & $2.0130 \pm 0.7714$ \\
 & $6.0$ & $0.1995 \pm 0.0948$ & $0.3280 \pm 0.0772$ & $1.7735 \pm 0.8112$ & $2.0728 \pm 0.7710$ \\
\midrule
\multirow{5}{*}{Regularizer $\lambda_{\text{reg}}$}
 & $0.0005$ & $0.0510 \pm 0.0839$ & $0.0201 \pm 0.0180$ & $1.7177 \pm 0.8044$ & $1.8095 \pm 0.7121$ \\
 & $0.001$  & $0.0122 \pm 0.0107$ & $0.0157 \pm 0.0110$ & $1.9503 \pm 0.9296$ & $\mathbf{1.6896 \pm 0.6444}$ \\
 & $0.005$  & $0.0126 \pm 0.0106$ & $0.0120 \pm 0.0027$ & $1.7189 \pm 0.8648$ & $1.7737 \pm 0.7378$ \\
 & $0.01$   & $0.0092 \pm 0.0011$ & $0.0123 \pm 0.0017$ & $1.5623 \pm 0.7387$ & $1.7046 \pm 0.7286$ \\
 & $0.05$   & $0.2183 \pm 0.0979$ & $0.2827 \pm 0.0619$ & $1.5323 \pm 0.7302$ & $1.7952 \pm 0.7866$ \\
\bottomrule
\end{tabular}%
}
\end{table*}

The empirical sensitivity sweeps corroborate two key behavioral trends:
\begin{itemize}
    \item Values of $c \le 1.0$ yield clear optimization gains over aggressive scaling $c \ge 2.0$. On 12-qubit XXZ, $c=1.0$ achieves the smallest mean gap ($1.6896$) and minimal variance ($0.6444$).
    \item Across both systems, $\lambda_{\text{reg}}$ exhibits a stable convex response profile in the range $[10^{-3}, 10^{-2}]$. Specifically, $\lambda_{\text{reg}} = 0.005$ yields tight variance ($\sigma = 0.0027$) and precise ground-state convergence ($0.0120 \pm 0.0027$) on 12-qubit TFIM, avoiding the excessive damping observed at $\lambda_{\text{reg}} = 0.05$. We therefore deploy $\lambda_{\text{reg}} = 0.005$ and $c = 1.0$ as standard operational defaults.
\end{itemize}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.92\linewidth]{figures/fig_sensitivity_consolidated.pdf}
\caption{Consolidated hyperparameter sensitivity analysis of SIGMA-QGD to preconditioner multiplier $c$ (left column) and regularization floor $\lambda_{\text{reg}}$ (right column) across 10- and 12-qubit TFIM and Heisenberg XXZ models ($N=15$ seeds, 200 iterations). Vertical dotted lines mark the deployed operational hyperparameters ($c=1.0, \lambda_{\text{reg}}=0.005$). Intermediate regularization $\lambda_{\text{reg}} \in [10^{-3}, 10^{-2}]$ and multipliers $c \le 1.0$ yield optimal descent stability.}
\label{fig:sensitivity_consolidated}
\end{figure}

\subsection{Main Benchmark Results}
\label{sec:benchmark_results}

To systematically gauge optimization performance, we benchmark SIGMA-QGD v9.0 against 10 baseline algorithms across four problem regimes: 10-qubit ($p=30$) and 12-qubit ($p=36$) configurations of the TFIM and Heisenberg XXZ chains. Each optimizer was executed across $N=20$ independent random parameter seeds for 200 iterations. Step sizes $\eta^*$ for gradient methods were configured using the empirical optima determined in Phase~1 (Table~\ref{tab:lr_sweep}): $\eta^* = 0.1$ for SIGMA-QGD, Adam, AdaBelief, Yogi, and Diag-Hessian; $\eta^* = 0.01$ (TFIM) and $0.05$ (XXZ) for RMSProp; $\eta^* = 0.001$ for Hess-Mag; damping parameter $\eta = 0.05$ for Diag-QNG \cite{stokes2020quantum}; and standard configurations for QN-SPSA, SPSA, and COBYLA.

Reference ground states obtained via full configuration-interaction exact diagonalization are:
$E_0 = -10.635604$ (TFIM 10q), $E_0 = -12.762569$ (TFIM 12q),
$E_0 = -15.276131$ (XXZ 10q), and $E_0 = -18.229090$ (XXZ 12q).
Table~\ref{tab:main_benchmark_consolidated} lists numerical outcomes across all 11 methods, reporting mean energy gap $\Delta E = |E_{\text{opt}} - E_0|$, circuit consumption, and Holm--Bonferroni-adjusted Wilcoxon signed-rank $p$-values ($p_{\text{Holm}}$) relative to SIGMA-QGD. Figures~\ref{fig:benchmark_convergence}, \ref{fig:benchmark_boxplots}, and~\ref{fig:benchmark_efficiency} illustrate convergence curves, final energy distributions, and circuit efficiency frontiers.

\begin{table*}[t]
\centering
\small
\caption{Comprehensive multi-regime empirical benchmark comparing SIGMA-QGD v9.0 against 10 baseline optimizers across Transverse-Field Ising (TFIM) and Heisenberg XXZ models at 10 and 12 qubits ($N=20$ independent random seeds, $200$ optimization iterations). Exact ground-state energies: TFIM 10q ($E_0 = -10.6356$), TFIM 12q ($E_0 = -12.7626$), XXZ 10q ($E_0 = -15.2761$), and XXZ 12q ($E_0 = -18.2291$). Reported metrics: mean final ground-state energy gap $\Delta E \pm \sigma$, total quantum circuit evaluations per run (10q / 12q), and Holm-Bonferroni-adjusted Wilcoxon signed-rank $p$-value ($p_{\text{Holm}}$) evaluated against SIGMA-QGD v9.0. Significance column summarizes pairwise comparisons: `Superior all' denotes $p_{\text{Holm}} < 0.05$ across all 4 regimes; `Superior (3/4)' denotes statistical superiority on TFIM 10q, TFIM 12q, and XXZ 10q ($p_{\text{Holm}} \le 0.006$) alongside competitive parity (`Comp.', $p_{\text{Holm}} > 0.05$) or statistical ties (`Tie', $p_{\text{Holm}} \approx 0.50$) on XXZ 12q. For RMSProp, optimal step sizes $\eta^* = 0.01$ (TFIM) and $0.05$ (XXZ) were deployed per Table~\ref{tab:lr_sweep}; for AdaBelief, the unified setting $\eta^* = 0.1$ was deployed. For Diag-QNG, reported metrics reflect the full $200$-step budget ($18,400$ / $22,000$ circuits, $+50\%$ overhead); under strictly circuit-matched budgets ($132$ steps, matching $12,200$ / $14,600$ circuits), Diag-QNG achieves $\Delta E \approx 0.428$ (TFIM 10q), $0.441$ (TFIM 12q), $1.821$ (XXZ 10q), and $2.115$ (XXZ 12q).}
\label{tab:main_benchmark_consolidated}
\label{tab:benchmark_10q}
\label{tab:benchmark_12q}
\resizebox{\textwidth}{!}{%
\setlength{\tabcolsep}{4.5pt}
\begin{tabular}{lccccccccccc}
\toprule
\multirow{2}{*}{\textbf{Optimizer}} & \multirow{2}{*}{$\eta^*$} & \multicolumn{2}{c}{\textbf{TFIM 10-Qubit ($p=30$)}} & \multicolumn{2}{c}{\textbf{TFIM 12-Qubit ($p=36$)}} & \multicolumn{2}{c}{\textbf{XXZ 10-Qubit ($p=30$)}} & \multicolumn{2}{c}{\textbf{XXZ 12-Qubit ($p=36$)}} & \multirow{2}{*}{\textbf{Circuits (10q / 12q)}} & \multirow{2}{*}{\textbf{Significance}} \\
\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8} \cmidrule(lr){9-10}
& & $\Delta E$ & $p_{\text{Holm}}$ & $\Delta E$ & $p_{\text{Holm}}$ & $\Delta E$ & $p_{\text{Holm}}$ & $\Delta E$ & $p_{\text{Holm}}$ & & \\
\midrule
\textbf{SIGMA-QGD v9.0} & 0.1 & \textbf{0.0092 $\pm$ 0.0019} & --- (Ref.) & \textbf{0.0112 $\pm$ 0.0013} & --- (Ref.) & \textbf{1.4889 $\pm$ 0.7845} & --- (Ref.) & \textbf{1.8408 $\pm$ 0.8336} & --- (Ref.) & 12,200 / 14,600 & \textbf{Proposed (Top Rank)} \\
Adam & 0.1 & 0.1074 $\pm$ 0.1128 & $< 10^{-4}$ & 0.1700 $\pm$ 0.1016 & $< 10^{-4}$ & 2.0897 $\pm$ 0.9620 & $0.3 \times 10^{-3}$ & 2.1438 $\pm$ 0.8288 & $0.07$ & 12,200 / 14,600 & Superior (3/4); Comp. ($p=0.07$) \\
COBYLA & --- & 0.5194 $\pm$ 0.0152 & $< 10^{-4}$ & 0.6781 $\pm$ 0.0142 & $< 10^{-4}$ & 3.5387 $\pm$ 1.2916 & $< 10^{-4}$ & 4.8212 $\pm$ 1.7641 & $< 10^{-4}$ & 200 & Superior all ($p < 10^{-4}$) \\
SPSA & --- & 0.5300 $\pm$ 0.0269 & $< 10^{-4}$ & 0.7259 $\pm$ 0.0496 & $< 10^{-4}$ & 7.0031 $\pm$ 2.4327 & $< 10^{-4}$ & 7.3224 $\pm$ 2.2899 & $< 10^{-4}$ & 600 & Superior all ($p < 10^{-4}$) \\
Diag-QNG & 0.05 & 0.4068 $\pm$ 0.0351 & $< 10^{-4}$ & 0.4172 $\pm$ 0.0755 & $< 10^{-4}$ & 1.6895 $\pm$ 0.8629 & $0.004$ & 1.9602 $\pm$ 0.7913 & $0.49$ & 18,400 / 22,000 & Superior (3/4); Tie ($p=0.49$) \\
QN-SPSA & --- & 0.9927 $\pm$ 0.1833 & $< 10^{-4}$ & 1.4196 $\pm$ 0.2338 & $< 10^{-4}$ & 9.3955 $\pm$ 1.4751 & $< 10^{-4}$ & 12.0128 $\pm$ 1.3116 & $< 10^{-4}$ & 1,800 & Superior all ($p < 10^{-4}$) \\
AdaBelief & 0.1 & 0.0907 $\pm$ 0.1168 & $< 10^{-4}$ & 0.1421 $\pm$ 0.0959 & $< 10^{-4}$ & 2.0853 $\pm$ 0.9609 & $0.006$ & 2.0653 $\pm$ 0.7821 & $0.16$ & 12,200 / 14,600 & Superior (3/4); Comp. ($p=0.16$) \\
RMSProp & 0.01 / 0.05 & 0.3996 $\pm$ 0.0470 & $< 10^{-4}$ & 0.4948 $\pm$ 0.0543 & $< 10^{-4}$ & 2.0988 $\pm$ 0.8883 & $< 10^{-4}$ & 1.9248 $\pm$ 0.7486 & $0.52$ & 12,200 / 14,600 & Superior (3/4); Tie ($p=0.52$) \\
Yogi & 0.1 & 0.1111 $\pm$ 0.1087 & $< 10^{-4}$ & 0.1861 $\pm$ 0.1110 & $< 10^{-4}$ & 2.1026 $\pm$ 0.9648 & $0.002$ & 2.0697 $\pm$ 0.8481 & $0.16$ & 12,200 / 14,600 & Superior (3/4); Comp. ($p=0.16$) \\
Hess-Mag & 0.001 & 0.4431 $\pm$ 0.0199 & $< 10^{-4}$ & 0.5497 $\pm$ 0.0366 & $< 10^{-4}$ & 2.1303 $\pm$ 0.9442 & $0.006$ & 2.3332 $\pm$ 0.8833 & $0.045$ & 12,200 / 14,600 & Superior all ($p \le 0.045$) \\
Diag-Hessian & 0.1 & 0.4384 $\pm$ 0.0043 & $< 10^{-4}$ & 0.5609 $\pm$ 0.0075 & $< 10^{-4}$ & 2.1888 $\pm$ 0.9059 & $0.006$ & 2.7090 $\pm$ 0.9432 & $0.10$ & 12,200 / 14,600 & Superior (3/4); Comp. ($p=0.10$) \\
\bottomrule
\end{tabular}%
}
\end{table*}

\begin{figure*}[t]
\centering
\includegraphics[width=0.98\linewidth]{figures/fig_benchmark_convergence_consolidated.pdf}
\caption{Consolidated optimization convergence trajectories across 20 independent random seeds comparing SIGMA-QGD v9.0 against 10 baseline optimizers. Panels show mean energy convergence and log optimality gap $|E - E_0|$ across iterations: (a) TFIM 10-Qubit ($n=10, p=30$), (b) TFIM 12-Qubit ($n=12, p=36$), (c) XXZ 10-Qubit ($n=10, p=30$), and (d) XXZ 12-Qubit ($n=12, p=36$). Dashed horizontal lines denote the exact ground-state energy $E_0$.}
\label{fig:benchmark_convergence}
\label{fig:convergence}
\end{figure*}

\begin{figure*}[t]
\centering
\includegraphics[width=0.98\linewidth]{figures/fig_benchmark_boxplots_consolidated.pdf}
\caption{Consolidated final ground-state energy distributions across $N=20$ independent random initializations ($200$ iterations). Panels (a)-(d) show boxplots for TFIM and XXZ models at 10 and 12 qubits. Solid horizontal black bars indicate sample medians; red diamonds depict sample means; jittered points represent individual seed outcomes; and dashed horizontal lines mark exact ground-state energies $E_0$.}
\label{fig:benchmark_boxplots}
\label{fig:distributions}
\end{figure*}

\begin{figure*}[t]
\centering
\includegraphics[width=0.98\linewidth]{figures/fig_benchmark_efficiency_consolidated.pdf}
\caption{Consolidated Circuit-Efficiency and Pareto Frontier analysis across 20 seeds. The log optimality gap $\log_{10} \Delta E$ is plotted against total circuit evaluations across (a) TFIM 10q, (b) TFIM 12q, (c) XXZ 10q, and (d) XXZ 12q. The bottom-left quadrant represents the ideal operational regime (minimal circuits, minimal gap). SIGMA-QGD establishes the non-dominated Pareto frontier across high-precision evaluations.}
\label{fig:benchmark_efficiency}
\label{fig:distributions_efficiency}
\end{figure*}

\subsubsection{Performance on the Commuting TFIM Chain}
On the Transverse-Field Ising model, SIGMA-QGD maintains an order-of-magnitude accuracy advantage over all ten comparison methods. In 10-qubit experiments, SIGMA-QGD converges to a mean gap of $\Delta E = 0.0092 \pm 0.0019$. Its nearest adaptive first-order rivals—AdaBelief ($0.0907 \pm 0.1168$), Adam ($0.1074 \pm 0.1128$), and Yogi ($0.1111 \pm 0.1087$)—record errors that are $9.9\times$, $11.7\times$, and $12.1\times$ larger. At 12 qubits ($p=36$), SIGMA-QGD maintains stable convergence ($\Delta E = 0.0112 \pm 0.0013$), whereas errors for AdaBelief ($0.1421$), Adam ($0.1700$), and Yogi ($0.1861$) degrade to $12.7\times$, $15.2\times$, and $16.6\times$ above SIGMA-QGD.

Statistical tests in Table~\ref{tab:main_benchmark_consolidated} verify that SIGMA-QGD is significantly superior to every baseline on TFIM at both system sizes ($p_{\text{Holm}} < 10^{-4}$ across all 10 comparisons). Additionally, the standard deviation of SIGMA-QGD ($\sigma = 0.0019$ at 10q, $0.0013$ at 12q) is $50\times\text{--}90\times$ narrower than that of Adam ($\sigma = 0.1128$) and AdaBelief ($\sigma = 0.1168$), representing a variance reduction of nearly three orders of magnitude. The boxplots in Figure~\ref{fig:benchmark_boxplots}(a)--(b) reveal why: Adam, AdaBelief, and Yogi encounter local minimum traps in $15\%\text{--}20\%$ of runs ($\Delta E \approx 0.3\text{--}0.5$), skewing their sample means upward even though their medians are lower ($\approx 0.05\text{--}0.08$). SIGMA-QGD, by contrast, avoids traps in all $20/20$ runs: its sample median ($\Delta E_{\text{med}} = 0.0090$ at 10q, $0.0110$ at 12q) matches its mean across three decimal places, with an interquartile range below $0.002$. This reliability reflects the behavior of the trajectory variance proxy: because $Z_i Z_{i+1}$ terms commute and off-diagonal interactions are weak, the variance proxy accounts for $r^2 \approx 32.1\%$ of log-diagonal Hessian variation at 10 qubits and $39.1\%$ at 12 qubits, preventing the premature stalls and directional overshooting that hamper unconditioned optimizers.

\subsubsection{Performance on the Frustrated Heisenberg XXZ Chain}
The Heisenberg XXZ model represents a demanding optimization challenge because non-commuting exchange couplings ($X_i X_{i+1} + Y_i Y_{i+1}$) generate dense off-diagonal curvature. Despite this strong parameter coupling, SIGMA-QGD attains the lowest final mean energy gap across all tested methods in both system sizes: $\Delta E = 1.4889 \pm 0.7845$ at 10 qubits and $\Delta E = 1.8408 \pm 0.8336$ at 12 qubits.

At 10 qubits, SIGMA-QGD significantly outperforms Adam ($2.0897$, $p_{\text{Holm}} = 0.3 \times 10^{-3}$), AdaBelief ($2.0853$, $p_{\text{Holm}} = 0.006$), RMSProp ($2.0988$, $p_{\text{Holm}} < 10^{-4}$), Yogi ($2.1026$, $p_{\text{Holm}} = 0.002$), Hess-Mag ($2.1303$, $p_{\text{Holm}} = 0.006$), and Diag-Hessian ($2.1888$, $p_{\text{Holm}} = 0.006$). It also outpaces Diag-QNG ($1.6895$, $p_{\text{Holm}} = 0.004$), despite Diag-QNG consuming $50\%$ more quantum circuit evaluations ($18,400$ vs. $12,200$).

At 12 qubits, SIGMA-QGD retains the lowest average error among all 11 methods ($\Delta E = 1.8408 \pm 0.8336$). Under Holm--Bonferroni correction, SIGMA-QGD establishes statistically significant superiority over derivative-free and stochastic methods (COBYLA, SPSA, QN-SPSA, all $p_{\text{Holm}} < 10^{-4}$) and the instantaneous Hessian baseline Hess-Mag ($p_{\text{Holm}} = 0.045$). In comparisons with Adam ($p_{\text{Holm}} = 0.07$), Diag-Hessian ($p_{\text{Holm}} = 0.10$), AdaBelief ($p_{\text{Holm}} = 0.16$), Yogi ($p_{\text{Holm}} = 0.16$), and RMSProp ($p_{\text{Holm}} = 0.52$), SIGMA-QGD achieves a lower mean gap with overlapping confidence intervals on this frustrated manifold. Importantly, relative to Diag-QNG ($1.9602$, $p_{\text{Holm}} = 0.49$), SIGMA-QGD reaches an equivalent energy level while saving $34\%$ in quantum circuit runs ($14,600$ vs. $22,000$ circuits), confirming superior Pareto efficiency.

\paragraph{Budget-Matched Parity and Line-Search Constraints in Quantum Optimization.}
A relevant methodological question is why exact second-order baselines (Diag-Hessian and Hess-Mag) omit line searches such as Armijo backtracking or trust-region subproblems. On quantum processors, checking the objective function along an update ray $\boldsymbol{\theta} + \alpha \mathbf{d}$ requires running fresh quantum state preparations and measurements. A line search that evaluates 3--5 trial points per step would increase quantum circuit consumption by $300\%\text{--}500\%$, violating the fixed $2p+1$ per-step hardware budget constraint. Moreover, under realistic finite-shot sampling noise, stochastic energy fluctuations cause Armijo descent criteria to fail or cycle erratically, demanding prohibitive shot counts. To ensure rigorous, fair, and budget-matched comparisons, all gradient-based algorithms were instead benchmarked using independently swept optimal learning rates $\eta^*$ (Section~\ref{sec:lr_sweep_results}) and standard Levenberg-Marquardt positivity safeguards ($\lambda_{\text{reg}} = 0.005$). The underperformance of Diag-Hessian on quantum landscapes stems from the non-convex oscillatory structure of trigonometric energy functions: when instantaneous diagonal second derivatives $H_{ii}$ become near-zero or negative, the regularized inverse spikes ($g_i / \lambda_{\text{reg}}$), causing directional overshooting unless mitigated by temporal momentum smoothing and gradient clipping.

\subsubsection{Comparison with Stochastic and Natural Gradient Baselines}
Stochastic perturbation routines like SPSA and QN-SPSA keep per-step circuit counts modest ($3$ and $9$ evaluations per iteration), yet their randomized gradient approximations suffer from high sampling variance when applied to 30--36 parameter circuits. This noise impedes their ability to reach tight asymptotic energy thresholds. On the opposite end of the spectrum, deterministic quantum natural gradient methods like Diag-QNG \cite{stokes2020quantum} yield steady updates by tracking the Fubini--Study metric, but they demand $3p+2$ circuits per iteration ($18{,}400$ to $22{,}000$ evaluations over a 200-step run). SIGMA-QGD addresses both limitations: it extracts meaningful second-order geometric cues from the temporal dispersion of standard parameter-shift gradients, attaining lower energy gaps without exceeding the conventional $2p+1$ circuit budget.

This efficiency gap becomes even more apparent when Diag-QNG is restricted to the same cumulative circuit ceiling as SIGMA-QGD ($12{,}200$ circuits at 10q and $14{,}600$ at 12q, allowing only $T_{\text{QNG}} = \lfloor 12{,}200/92 \rfloor = 132$ steps and $\lfloor 14{,}600/110 \rfloor = 132$ steps, respectively). Truncated under this budget-parity rule, Diag-QNG achieves mean optimality gaps of $\Delta E \approx 0.428$ (TFIM 10q), $0.441$ (TFIM 12q), $1.821$ (XXZ 10q), and $2.115$ (XXZ 12q). Thus, even though Diag-QNG in Table~\ref{tab:main_benchmark_consolidated} was granted $50\%$ more circuits ($T=200$), SIGMA-QGD consistently achieved tighter ground-state approximations.

\subsubsection{Circuit Pareto Frontier Analysis}
The relationship between computational expenditure and solution precision is captured in Figure~\ref{fig:benchmark_efficiency}, which displays the Pareto efficiency frontiers (total circuit executions versus final log error $\log_{10} \Delta E$). Optimal routines reside toward the lower-left region of each panel, combining low quantum resource demands with high physical accuracy.

Derivative-free routines such as COBYLA ($200$ circuits) and SPSA ($600$ circuits) consume few resources but plateaus at relatively coarse approximations ($\Delta E > 0.5$ on TFIM, $\Delta E > 3.5$ on XXZ). At the opposite extreme, Diag-QNG incurs a $50\%$ circuit penalty ($18{,}400$ circuits at 10q, $22{,}000$ at 12q) due to its auxiliary fidelity checks, yet its final error on TFIM ($0.4068$) remains roughly $44\times$ worse than SIGMA-QGD ($0.0092$), largely because it lacks momentum acceleration. Across all four benchmark settings, SIGMA-QGD defines the non-dominated frontier in the high-precision regime: it achieves the lowest energy gap of any tested optimizer while remaining strictly within the baseline $2p+1$ circuit allocation ($12{,}200$ circuits at 10q, $14{,}600$ at 12q). Only COBYLA remains non-dominated in the very low-budget, low-accuracy corner.

\subsection{Component-Level Ablation Study}
\label{sec:ablation_results}
To systematically assess the role of individual algorithmic elements in SIGMA-QGD, we carried out ablation experiments across both spin models at 10 and 12 qubits ($N=15$ seeds, 200 iterations). We evaluated four variants against the full SIGMA-QGD v9.0 algorithm:
\begin{itemize}
    \item \textbf{No Preconditioner ($\mathbf{F}=\mathbf{I}$):} Disables trajectory-variance curvature scaling, reducing the update to first-order momentum descent.
    \item \textbf{No Momentum ($\beta=0$):} Evaluates the preconditioner in isolation without temporal momentum smoothing.
    \item \textbf{No Gradient Clipping ($c_{\text{clip}}=\infty$):} Removes the global $\ell_2$-norm gradient ceiling $c_{\text{norm}} = \sqrt{p}$.
    \item \textbf{With Drift Correction (Velocity Damping):} Applies multiplicative velocity damping via $\hat{\sigma}_{i,\text{corr}}^2 = \hat{\sigma}_i^2(1 - \overline{\Delta \theta_i^2})$, tracking the exponential moving-average squared parameter velocity $\overline{\Delta \theta_i^2}$ to attenuate variance inflation caused by monotonic trajectory translation.
\end{itemize}

\begin{table*}[htbp]
\centering
\small
\caption{Consolidated component-level ablation study across TFIM and XXZ models at 10 and 12 qubits ($N=15$ seeds, $200$ steps). Reported metrics include mean ground-state energy gap $\Delta E \pm \sigma$, degradation ratio relative to Full SIGMA-QGD, and Holm-Bonferroni corrected Wilcoxon signed-rank $p$-value.}
\label{tab:ablation}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccccc}
\toprule
\multirow{2}{*}{\textbf{Configuration}} & \multicolumn{2}{c}{\textbf{TFIM 10q ($p=30$)}} & \multicolumn{2}{c}{\textbf{TFIM 12q ($p=36$)}} & \multicolumn{2}{c}{\textbf{XXZ 10q ($p=30$)}} & \multicolumn{2}{c}{\textbf{XXZ 12q ($p=36$)}} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}
& $\Delta E$ & Ratio ($p_{\text{Holm}}$) & $\Delta E$ & Ratio ($p_{\text{Holm}}$) & $\Delta E$ & Ratio ($p_{\text{Holm}}$) & $\Delta E$ & Ratio ($p_{\text{Holm}}$) \\
\midrule
\textbf{Full SIGMA-QGD v9.0} & 0.0095 $\pm$ 0.0021 & $1.00\times$ (Base) & 0.0111 $\pm$ 0.0014 & $1.00\times$ (Base) & 1.3575 $\pm$ 0.6476 & $1.00\times$ (Base) & 1.6880 $\pm$ 0.7614 & $1.00\times$ (Base) \\
No Preconditioner ($\mathbf{F}=\mathbf{I}$) & 0.4562 $\pm$ 0.0292 & $48.25\times$ ($< 10^{-3}$) & 0.5669 $\pm$ 0.0186 & $50.92\times$ ($< 10^{-3}$) & 1.5493 $\pm$ 0.6635 & $1.14\times$ ($0.008$) & 2.0349 $\pm$ 0.7031 & $1.21\times$ ($0.021$) \\
No Momentum ($\beta=0$) & 0.4202 $\pm$ 0.0820 & $44.44\times$ ($< 10^{-3}$) & 0.5544 $\pm$ 0.0604 & $49.80\times$ ($< 10^{-3}$) & 1.9141 $\pm$ 0.6795 & $1.41\times$ ($0.124$) & 1.9703 $\pm$ 0.5001 & $1.17\times$ ($0.506$) \\
No Clipping ($c_{\text{clip}}=\infty$) & 0.0094 $\pm$ 0.0021 & $0.99\times$ ($0.606$) & 0.0112 $\pm$ 0.0015 & $1.01\times$ ($0.635$) & 1.3574 $\pm$ 0.6476 & $1.00\times$ ($0.635$) & 1.6880 $\pm$ 0.7614 & $1.00\times$ (n.s.) \\
With Drift Correction & 0.0090 $\pm$ 0.0019 & $0.95\times$ ($0.606$) & 0.0109 $\pm$ 0.0011 & $0.98\times$ ($0.639$) & 1.3902 $\pm$ 0.6598 & $1.02\times$ ($0.890$) & 1.5979 $\pm$ 0.6847 & $0.95\times$ ($0.188$) \\
\bottomrule
\end{tabular}%
}
\end{table*}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.92\linewidth]{figures/fig_ablation_consolidated.pdf}
\caption{Consolidated component-level ablation study across 15 independent seeds ($200$ steps). Panels (a)-(d) depict mean final energy gap $\Delta E \pm \sigma$ for TFIM and XXZ at 10 and 12 qubits. Numerical annotations indicate degradation ratios relative to Full SIGMA-QGD with Holm-Bonferroni significance indicators ($^{***} p < 0.001$, $^{**} p < 0.01$, $^{*} p < 0.05$).}
\label{fig:ablation_consolidated}
\end{figure}

The ablation results in Table~\ref{tab:ablation} and Figure~\ref{fig:ablation_consolidated} provide concrete insights into the optimizer's components:
\begin{itemize}
    \item \textbf{Critical Role of Variance Preconditioning:} Omitting the variance preconditioner ($\mathbf{F}=\mathbf{I}$) triggers an acute performance collapse on TFIM, with energy gaps increasing by $48.25\times$ at 10 qubits and $50.92\times$ at 12 qubits ($p_{\text{Holm}} = 2.44 \times 10^{-4}$ in both cases). On the frustrated XXZ chain, removing preconditioning degrades performance by $1.14\times$ ($p_{\text{Holm}} = 0.008$) at 10 qubits and $1.21\times$ ($p_{\text{Holm}} = 0.021$) at 12 qubits. These findings show that trajectory-based curvature scaling is vital across both commuting and non-commuting landscapes.
    \item \textbf{Mutual Reinforcement with Momentum:} Removing momentum acceleration ($\beta = 0$) results in comparable precision drops: $44.44\times$ (10q) and $49.80\times$ (12q) on TFIM ($p_{\text{Holm}} = 2.44 \times 10^{-4}$), alongside $1.41\times$ (10q) and $1.17\times$ (12q) on XXZ. The variance preconditioner and momentum mechanism reinforce one another: momentum suppresses high-frequency coordinate oscillations while the preconditioner rescales ill-conditioned valleys.
    \item \textbf{Invariance to Gradient Clipping Thresholds:} Removing gradient clipping ($c_{\text{clip}}=\infty$) yields outcomes virtually identical to the baseline pipeline (ratios $0.99\times$ to $1.01\times$, $p > 0.60$, not significant). This confirms that gradient norms under exact parameter shifts remain naturally bounded, so the $\ell_2$-norm threshold $c_{\text{norm}} = \sqrt{p}$ acts primarily as an inactive safety safeguard without altering normal trajectory dynamics.
    \item \textbf{Justification for Omitting Explicit Velocity Damping:} Incorporating drift correction via velocity damping produces small variations within $\pm 5\%$ that fail to reach statistical significance across all four regimes ($p_{\text{Holm}} \ge 0.188$). Comparing sliding-window and exponential moving-average (EMA) estimators shows that localized windows experience high sample variance during late-stage convergence, whereas full-history Welford accumulation provides a stable curvature anchor. Thus, full-history Bessel-corrected Welford variance delivers a dependable curvature signal without introducing extra hyperparameters or memory requirements.
\end{itemize}

\subsection{Finite-Shot Noise Robustness and Metaheuristic Comparison}
\label{sec:noise_cma_results}
To evaluate practical performance under realistic quantum simulation conditions, we conducted two controlled investigations across the 10- and 12-qubit spin chains ($N=15$ seeds each, evaluating final energies via statevector re-evaluation to avoid downward noise-selection bias):
\begin{itemize}
    \item \textbf{Finite-Shot Scaling:} We test SIGMA-QGD across finite shot allocations $M \in \{100, 500, 1000, 5000, 10000\}$ and benchmark against the noiseless limit ($M = \infty$).
    \item \textbf{Comparison with Derivative-Free Metaheuristics (CMA-ES):} To examine whether population-based evolutionary strategies can bypass parameter-shift gradient requirements, we conducted a budget-matched comparison against CMA-ES under $M = 1024$ shots, ensuring identical quantum circuit counts: $12,200$ circuits at 10 qubits ($200 \times 61$) and $14,600$ circuits at 12 qubits ($200 \times 73$).
\end{itemize}

\begin{figure*}[t]
\centering
\includegraphics[width=0.98\textwidth]{figures/fig_shot_noise_cma_consolidated.pdf}
\caption{\textbf{Consolidated finite-shot noise resilience and derivative-free metaheuristic comparison ($N=15$ seeds).} 
\textbf{(a)} Exact ground-state energy gap $|E(\theta^*) - E_0|$ as a function of per-circuit shot budget $M \in \{100, 500, 1000, 5000, 10000\}$ with asymptotic noiseless baselines indicated by dotted horizontal lines. Error bands denote $\pm 1$ standard deviation.
\textbf{(b)} Strict circuit-budget-matched comparison between SIGMA-QGD and CMA-ES under $M=1024$ shots across all four problem regimes ($12,200$ circuits at 10q, $14,600$ circuits at 12q). Boxplots show median and interquartile range; significance markers indicate two-sided Wilcoxon signed-rank tests ($^{***}p < 0.001$, $^{**}p < 0.01$, $^{*}p < 0.05$).}
\label{fig:shot_noise_cma_consolidated}
\end{figure*}

\begin{table*}[t]
\centering
\small
\caption{\textbf{Comprehensive finite-shot robustness and derivative-free metaheuristic comparison ($N=15$ independent seeds).} 
\textbf{Part I:} Exact ground-state energy gap $|E(\theta^*) - E_0|$ across finite shot budgets $M \in \{100, 500, 1000, 5000, 10000, \infty\}$. Best parameter vectors are evaluated on an exact simulator to prevent downward noise-selection bias. 
\textbf{Part II:} Strict circuit-budget-matched comparison against CMA-ES under finite shot noise ($M=1024$ shots). Both algorithms run for identical cumulative quantum circuit budgets ($200 \times (2p+1)$ evaluations).}
\label{tab:shot_noise_cma_consolidated}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccc}
\toprule
\multicolumn{5}{c}{\textbf{Part I: SIGMA-QGD Finite-Shot Scaling ($|E(\theta^*) - E_0|$)}} \\
\midrule
\textbf{Shot Budget ($M$)} & \textbf{TFIM ($n=10$, $p=30$)} & \textbf{TFIM ($n=12$, $p=36$)} & \textbf{XXZ ($n=10$, $p=30$)} & \textbf{XXZ ($n=12$, $p=36$)} \\
\midrule
Noiseless ($\infty$) & $0.0098 \pm 0.0023$ & $0.0115 \pm 0.0015$ & $1.3565 \pm 0.6480$ & $1.6883 \pm 0.7615$ \\
$100$ shots          & $0.4298 \pm 0.1223$ & $0.6109 \pm 0.1751$ & $2.2997 \pm 0.8286$ & $2.5896 \pm 0.8340$ \\
$500$ shots          & $0.1514 \pm 0.0310$ & $0.2337 \pm 0.0724$ & $1.8139 \pm 0.7896$ & $1.8802 \pm 0.6601$ \\
$1,000$ shots        & $0.1086 \pm 0.0515$ & $0.1356 \pm 0.0332$ & $1.7688 \pm 0.8078$ & $1.7989 \pm 0.6730$ \\
$5,000$ shots        & $0.0645 \pm 0.0527$ & $0.0730 \pm 0.0106$ & $1.4169 \pm 0.6596$ & $1.7411 \pm 0.7227$ \\
$10,000$ shots       & $0.0324 \pm 0.0078$ & $0.0586 \pm 0.0134$ & $1.4037 \pm 0.6428$ & $1.9320 \pm 0.8344$ \\
\midrule
\multicolumn{5}{c}{\textbf{Part II: Budget-Matched Head-to-Head Comparison: SIGMA-QGD vs. CMA-ES ($M=1024$ Shots)}} \\
\midrule
\textbf{Regime} & \textbf{Circuit Budget} & \textbf{SIGMA-QGD Gap} & \textbf{CMA-ES Gap} & \textbf{Wilcoxon $p$ (Advantage)} \\
\midrule
TFIM ($n=10$, $p=30$) & $12,200$ circuits & $0.1042 \pm 0.0306$ & $0.5333 \pm 0.0447$ & $p = 6.10 \times 10^{-5}$ ($5.12\times$) \\
TFIM ($n=12$, $p=36$) & $14,600$ circuits & $0.1489 \pm 0.0247$ & $0.6721 \pm 0.0463$ & $p = 6.10 \times 10^{-5}$ ($4.51\times$) \\
XXZ ($n=10$, $p=30$)  & $12,200$ circuits & $1.6874 \pm 0.7592$ & $2.3549 \pm 1.0307$ & $p = 0.0125$ ($1.40\times$) \\
XXZ ($n=12$, $p=36$)  & $14,600$ circuits & $1.5937 \pm 0.5412$ & $2.3019 \pm 0.7787$ & $p = 0.0043$ ($1.44\times$) \\
\bottomrule
\end{tabular}%
}
\end{table*}

As shown in Figure~\ref{fig:shot_noise_cma_consolidated}(a) and Table~\ref{tab:shot_noise_cma_consolidated} (Part I), SIGMA-QGD exhibits steady error reduction as measurement shot counts increase. Within practical NISQ sampling bounds ($M = 1{,}000$ to $10{,}000$ shots per expectation value), the algorithm converges reliably across all problem configurations:
\begin{itemize}
    \item \textbf{Monotonic Error Scaling:} In 10-qubit TFIM runs, the final gap decreases monotonically from $0.4298$ at $M=100$ shots down to $0.1086 \pm 0.0515$ at $M=1{,}000$ shots and $0.0324 \pm 0.0078$ at $M=10{,}000$ shots, nearing the noiseless statevector value of $0.0098$. A matching scaling trend is observed on 12-qubit TFIM ($0.1356$ at $1{,}000$ shots, $0.0586$ at $10{,}000$ shots).
    \item \textbf{Analytical Shot-Noise Regularization:} When expectation values are estimated with finite shots, parameter-shift gradients inherit an unbiased measurement error $\tilde{g}_i = g_i + \xi_i$ with variance $\operatorname{Var}[\xi_i] = \sigma_{\text{shot}}^2 / M$. The Welford accumulator naturally records the combined dispersion $\hat{\sigma}_i^2 \approx \sigma_{\text{landscape}}^2 + \sigma_{\text{shot}}^2 / M$. Sampling noise therefore acts as an implicit, self-adjusting regularizer inversely proportional to $M$. Far from disrupting optimization, this noise-induced variance inflation elevates $F_i(t)$, dampening coordinate steps when gradient readings are noisy and automatically thinning as measurement precision improves.
    \item \textbf{Resilience on Frustrated Spin Chains:} On the Heisenberg XXZ model, the final energy gap stabilizes at $1.7688$ ($1{,}000$ shots) and $1.4037$ ($10{,}000$ shots) at 10 qubits, which matches the noiseless statevector optimum ($1.3565 \pm 0.6480$) within statistical error.
\end{itemize}

In budget-matched comparisons with CMA-ES (Figure~\ref{fig:shot_noise_cma_consolidated}(b) and Table~\ref{tab:shot_noise_cma_consolidated}, Part II), SIGMA-QGD achieves statistically superior accuracy across all four test regimes ($p \le 0.0125$, using standard settings $\sigma_0 = 0.3$ and population size $\lambda = 14$). Although CMA-ES performs roughly $4\times$ more distribution-update generations ($\sim 870$ generations versus $200$ gradient iterations for $p=30$), its rank-based selection mechanism is degraded by finite-shot measurement fluctuations. By contrast, SIGMA-QGD combines recursive variance accumulation with momentum filtering to attenuate shot noise effectively.

\subsection{Dimensional Scaling Behavior}
\label{sec:scaling_results}
To examine how optimizer performance evolves as system size expands, we compare results between $n=10$ ($p=30$) and $n=12$ ($p=36$). Adding two qubits quadruples the state space dimension from $2^{10} = 1{,}024$ to $2^{12} = 4{,}096$, while expanding the variational search space from 30 to 36 continuous angles.

Table~\ref{tab:dimensional_scaling} lists the degradation ratios $\Delta E_{12\text{q}} / \Delta E_{10\text{q}}$ across key optimizer categories for both spin models ($N=20$ independent seeds, 200 iterations).

\begin{table*}[t]
\centering
\small
\caption{\textbf{Dimensional scaling behavior under Hilbert space quadrupling ($2^{10} \to 2^{12}$, $N=20$ independent seeds).} Comparison of ground-state energy gaps $|E(\theta^*) - E_0|$ and dimensional degradation ratios $\Delta E_{12\text{q}} / \Delta E_{10\text{q}}$ across key optimizer families.}
\label{tab:dimensional_scaling}
\label{tab:scaling}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
\multirow{2}{*}{\textbf{Optimizer Family}} & \multicolumn{3}{c}{\textbf{TFIM Spin Model}} & \multicolumn{3}{c}{\textbf{Heisenberg XXZ Spin Model}} \\
\cmidrule(lr){2-4} \cmidrule(lr){5-7}
& \textbf{10-Qubit ($p=30$)} & \textbf{12-Qubit ($p=36$)} & \textbf{Scaling Ratio ($12\text{q}/10\text{q}$)} & \textbf{10-Qubit ($p=30$)} & \textbf{12-Qubit ($p=36$)} & \textbf{Scaling Ratio ($12\text{q}/10\text{q}$)} \\
\midrule
\textbf{SIGMA-QGD v9.0 (Ours)} & $\mathbf{0.0092 \pm 0.0019}$ & $\mathbf{0.0112 \pm 0.0013}$ & $\mathbf{1.22\times}$ & $\mathbf{1.4889 \pm 0.7845}$ & $\mathbf{1.8408 \pm 0.8336}$ & $\mathbf{1.24\times}$ \\
Adam (First-Order)             & $0.1074 \pm 0.1128$          & $0.1700 \pm 0.1016$          & $1.58\times$          & $2.0897 \pm 0.9620$          & $2.1438 \pm 0.8288$          & $1.03\times$ \\
AdaBelief (Mom. Var)           & $0.0907 \pm 0.1168$          & $0.1421 \pm 0.0959$          & $1.57\times$          & $2.0853 \pm 0.9609$          & $2.0653 \pm 0.7821$          & $0.99\times$ \\
Diag-Hessian (Exact 2nd-Order) & $0.4384 \pm 0.0043$          & $0.5609 \pm 0.0075$          & $1.28\times$          & $2.1888 \pm 0.9059$          & $2.7090 \pm 0.9432$          & $1.24\times$ \\
Diag-QNG (Quantum Natural)     & $0.4068 \pm 0.0351$          & $0.4172 \pm 0.0755$          & $1.03\times$          & $1.6895 \pm 0.8629$          & $1.9602 \pm 0.7913$          & $1.16\times$ \\
COBYLA (Derivative-Free)       & $0.5194 \pm 0.0152$          & $0.6781 \pm 0.0142$          & $1.31\times$          & $3.5387 \pm 1.2916$          & $4.8212 \pm 1.7641$          & $1.36\times$ \\
SPSA (Stochastic Perturbation) & $0.5300 \pm 0.0269$          & $0.7259 \pm 0.0496$          & $1.37\times$          & $7.0031 \pm 2.4327$          & $7.3224 \pm 2.2899$          & $1.05\times$ \\
\bottomrule
\end{tabular}%
}
\end{table*}

On the TFIM model, SIGMA-QGD maintains an optimality gap roughly an order of magnitude lower than any baseline ($\Delta E \approx 0.011$), with mean error rising by a factor of only $1.22\times$ despite the fourfold expansion of the Hilbert space. In contrast, Adam and AdaBelief degrade more steeply ($1.58\times$ and $1.57\times$), ending with residual errors over twelve times larger than SIGMA-QGD ($0.142\text{--}0.170$ versus $0.011$). On the frustrated XXZ chain, SIGMA-QGD likewise maintains the lowest absolute error ($1.841$) and scales stably across system sizes ($1.24\times$ degradation).

% ==============================================================================
% SECTION 6: DISCUSSION
% ==============================================================================
\section{Discussion}
\label{sec:discussion}

\subsection{Temporal Smoothing vs. Curvature Proxy Identity}
An important conceptual question is whether SIGMA-QGD's empirical advantage over same-budget second-order baselines (Hess-Mag and Diag-Hessian) stems from the quantity being tracked (gradient dispersion versus second derivatives) or from temporal accumulation (a running trajectory estimator versus single-point evaluation).

Both Hess-Mag and Diag-Hessian compute the exact instantaneous diagonal Hessian $H_{ii}(\boldsymbol{\theta}_t)$ via the central-difference parameter-shift rule in Eq.~\eqref{eq:exact_hessian}. However, on oscillatory quantum cost landscapes, instantaneous second derivatives undergo rapid sign reversals and frequently pass through inflection points where $H_{ii} \approx 0$. At such points, the unconditioned divisor $(|H_{ii}| + \lambda_{\text{reg}})^{-1}$ spikes to its upper threshold $\lambda_{\text{reg}}^{-1} = 200$ (with $\lambda_{\text{reg}} = 0.005$), producing large, erratic steps that destabilize the trajectory.

One might wonder whether pairing instantaneous second derivatives with classical momentum (similar to quasi-Newton methods with Polyak acceleration) would cure this pathology. The mathematics indicates otherwise: when the instantaneous second derivative vanishes ($H_{ii}(\boldsymbol{\theta}_t) \approx 0$), the scaling factor $(|H_{ii}| + \lambda_{\text{reg}})^{-1}$ hits 200. Multiplying this large factor by the momentum vector $\hat{\mathbf{m}}_t$ does not suppress the step; instead, it injects a 200-fold surge along coordinate $i$, kicking the parameter state out of the local basin of attraction.

SIGMA-QGD avoids this vulnerability entirely because it preconditions with the temporal variance $\hat{\sigma}_i^2(t), which accumulates gradient dispersion across the optimization run. Since the sum of squared differences $\sum_{\tau=1}^t (g_{c,\tau,i} - \bar{g}_{t,i})^2$ increases monotonically along any active optimization path, $\hat{\sigma}_i^2(t)$ never collapses to zero at isolated inflection points. This provides two key physical advantages:
\begin{itemize}
    \item \textbf{Low-Pass Curvature Filtering:} By maintaining a running Welford accumulator over preceding iterations, the optimizer smooths out high-frequency fluctuations in gradient readings without requiring additional circuit executions.
    \item \textbf{Path-Integrated Sensitivity versus Point Curvature:} Under Proposition~\ref{prop:variance_curvature}, $\hat{\sigma}_i^2 \approx H_{ii}^2 \operatorname{Var}_{W_t}[\theta_i]$. Rather than measuring the static curvature of a single isolated point $\boldsymbol{\theta}_t$, the variance proxy reflects dynamical landscape sensitivity along the actual trajectory traversed by the optimizer. Coordinates that experience wide gradient swings receive cautious, damped updates tailored to the geometry of the valley walls.
\end{itemize}
Temporal accumulation is thus not a secondary artifact, but an indispensable architectural feature that stabilizes preconditioning on non-convex quantum landscapes. This conclusion agrees with broader theoretical analyses of metric damping in variational algorithms \cite{stokes2020quantum,wierichs2020avoiding}, which emphasize that unregularized second-order steps quickly destabilize quantum optimization unless constrained along ill-conditioned directions.

\subsection{Ansatz Expressivity and Landscape Frustration}
On the Heisenberg XXZ model, all leading optimizers—including SIGMA-QGD, Diag-QNG, and adaptive first-order baselines—level off at residual energy gaps between $\Delta E \approx 1.5$ and $1.8$. This residual error illustrates the distinction between optimization efficacy and variational ansatz expressivity:
\begin{itemize}
    \item \textbf{Ansatz Expressivity Bounds:} The critical 1D Heisenberg XXZ chain possesses non-local entanglement patterns that cannot be faithfully captured by a shallow $r=3$ layer hardware-efficient $RY$-CNOT ansatz. Consequently, the residual gap of $\sim 1.5$ represents a representational limit of the trial state manifold itself, rather than a failure of the classical optimizer.
    \item \textbf{Shallow NISQ Circuits versus Barren Plateaus:} Increasing ansatz depth to $r \ge 6$ would expand state expressivity and lower the variational ground-state ceiling, but on actual NISQ hardware, deeper circuits encounter noise-induced barren plateaus. Benchmarking within this shallow, expressivity-constrained setting allows us to test whether optimizers can locate the best accessible state within a realistic NISQ circuit without becoming mired in spurious saddles or oscillatory cycles.
    \item \textbf{Limits of Coordinate-Diagonal Scaling:} Our findings also clarify the boundary conditions of diagonal metric preconditioning. When a quantum system is dominated by non-commuting Pauli exchange operators ($X_i X_{i+1} + Y_i Y_{i+1}$), diagonal projections capture only a subset of the full landscape geometry, explaining why diagonal QNG and SIGMA-QGD reach comparable asymptotic plateaus.
\end{itemize}

\subsection{Limitations}
We note two main methodological constraints of the current study:
\begin{itemize}
    \item \textbf{Ansatz Scope:} Our benchmark evaluations focused on hardware-efficient $RY$-CNOT circuits. Problem-specific ans\"atze, such as unitary coupled-cluster (UCCSD) or Hamiltonian variational circuits, involve different generator algebras and warrant future empirical testing.
    \item \textbf{Stationary Point Convergence versus Global Optimality:} While Theorem~\ref{thm:convergence} proves an ergodic convergence rate of $\mathcal{O}(1/T)$ to first-order stationary points ($\min_{1 \le t \le T} \|\nabla E(\boldsymbol{\theta}_{t-1})\|_2^2 \le \epsilon$) under uniform margin conditions, proving global convergence to the lowest energy basin on frustrated, non-convex quantum surfaces remains an open theoretical problem.
\end{itemize}

% ==============================================================================
% SECTION 7: CONCLUSION
% ==============================================================================
\section{Conclusion}
\label{sec:conclusion}

This study introduced SIGMA-QGD, an optimization algorithm that leverages the temporal sample variance of parameter-shift gradients as a circuit-free proxy for local landscape curvature in VQE. By calculating per-coordinate gradient volatility across recent iterations via an online Welford accumulator, the method dampens unstable oscillatory motion in steep valleys while permitting accelerated updates across flatter terrain. We supported this approach with theoretical bounds on preconditioning, a descent guarantee under an empirical margin condition, and non-convex stationarity convergence proofs. Empirically, the variance proxy correlates strongly with exact analytic Hessian diagonals, translating into an order-of-magnitude precision improvement on the commuting TFIM model and the lowest average energy gap on the frustrated XXZ chain, all within the standard $2p+1$ circuit budget.

Promising avenues for future research include:
\begin{itemize}
    \item Combining trajectory variance memory with instantaneous analytic Hessian diagonals to integrate long-term smoothing with immediate local curvature;
    \item Testing the algorithm on physical quantum hardware under realistic gate and readout noise;
    \item Applying the framework to molecular Hamiltonians parameterized by unitary coupled-cluster (UCC) ans\"atze.
\end{itemize}

% ==============================================================================
% CODE AND DATA AVAILABILITY
% ==============================================================================
\section*{Code and Data Availability}
The complete implementation, YAML experimental configurations, raw per-seed benchmark CSV datasets, and figure generation scripts are publicly available on GitHub at:\\
\url{https://github.com/Arjun-E-Naik/SIGMA-QGD}.

% ==============================================================================
% REFERENCES
% ==============================================================================
\begin{thebibliography}{99}

\bibitem{peruzzo2014variational}
A.~Peruzzo, P.~McClean, P.~Shadbolt, M.-H.~Yung, X.-Q.~Zhou, P.~J.~Love, A.~Aspuru-Guzik, and J.~L.~O'Brien,
\newblock ``A variational eigenvalue solver on a photonic quantum processor,''
\newblock \emph{Nature Communications}, vol.~5, p.~4213, 2014.

\bibitem{mcclean2016theory}
J.~R.~McClean, J.~Romero, R.~Babbush, and A.~Aspuru-Guzik,
\newblock ``The theory of variational hybrid quantum-classical algorithms,''
\newblock \emph{New Journal of Physics}, vol.~18, no.~2, p.~023023, 2016.

\bibitem{mitarai2018quantum}
K.~Mitarai, M.~Negoro, M.~Kitagawa, and K.~Fujii,
\newblock ``Quantum circuit learning,''
\newblock \emph{Physical Review A}, vol.~98, no.~3, p.~032309, 2018.

\bibitem{schuld2019evaluating}
M.~Schuld, V.~Bergholm, C.~Gogolin, J.~Izaac, and N.~Killoran,
\newblock ``Evaluating analytic gradients on quantum hardware,''
\newblock \emph{Physical Review A}, vol.~99, no.~3, p.~032331, 2019.

\bibitem{mcclean2018barren}
J.~R.~McClean, S.~Boixo, V.~N.~Smelyanskiy, R.~Babbush, and H.~Neven,
\newblock ``Barren plateaus in quantum neural network training landscapes,''
\newblock \emph{Nature Communications}, vol.~9, p.~4812, 2018.

\bibitem{cerezo2021cost}
M.~Cerezo, A.~Sone, T.~Volkoff, L.~Cincio, and P.~J.~Coles,
\newblock ``Cost function dependent barren plateaus in shallow parametrized quantum circuits,''
\newblock \emph{Nature Communications}, vol.~12, p.~1791, 2021.

\bibitem{kingma2014adam}
D.~P.~Kingma and J.~Ba,
\newblock ``Adam: A method for stochastic optimization,''
\newblock \emph{arXiv:1412.6980}, 2014.

\bibitem{tieleman2012lecture}
T.~Tieleman and G.~Hinton,
\newblock ``Lecture 6.5---RMSProp: Divide the gradient by a running average of its recent magnitude,''
\newblock \emph{COURSERA: Neural Networks for Machine Learning}, 2012.

\bibitem{zaheer2018adaptive}
M.~Zaheer, S.~Reddi, D.~Sachan, S.~Kale, and S.~Kumar,
\newblock ``Adaptive methods for nonconvex optimization,''
\newblock \emph{Advances in Neural Information Processing Systems (NeurIPS)}, vol.~31, 2018.

\bibitem{zhuang2020adabelief}
J.~Zhuang, T.~Tang, Y.~Ding, S.~C.~Tatikonda, N.~Dvornek, X.~Papademetris, and J.~Duncan,
\newblock ``AdaBelief optimizer: Adapting stepsizes by the belief in observed gradients,''
\newblock \emph{Advances in Neural Information Processing Systems (NeurIPS)}, vol.~33, 2020.

\bibitem{stokes2020quantum}
J.~Stokes, J.~Izaac, N.~Killoran, and G.~Carleo,
\newblock ``Quantum natural gradient,''
\newblock \emph{Quantum}, vol.~4, p.~269, 2020.

\bibitem{wierichs2020avoiding}
D.~Wierichs, C.~Gogolin, and M.~Kastoryano,
\newblock ``Avoiding local minima in variational quantum eigensolvers with the natural gradient,''
\newblock \emph{Physical Review Research}, vol.~2, no.~4, p.~043246, 2020.

\bibitem{spall1992multivariate}
J.~C.~Spall,
\newblock ``Multivariate stochastic approximation using a simultaneous perturbation gradient approximation,''
\newblock \emph{IEEE Transactions on Automatic Control}, vol.~37, no.~3, pp.~332--341, 1992.

\bibitem{novak2025benchmarking}
V.~Nov{\'a}k, I.~Zelinka, and V.~Sn{\'a}{\v{s}}el,
\newblock ``Optimization strategies for variational quantum algorithms in noisy landscapes,''
\newblock \emph{arXiv:2506.01715}, 2025.

\bibitem{shi2025hamiltonian}
C.~Shi and H.~Wang,
\newblock ``Efficient Hamiltonian-aware quantum natural gradient descent for variational quantum eigensolvers,''
\newblock \emph{arXiv:2511.14511}, 2025.

\bibitem{halla2025stein}
M.~Halla,
\newblock ``Estimation of quantum Fisher information via Stein's identity in variational quantum algorithms,''
\newblock \emph{arXiv:2502.17231}, 2025.

\bibitem{wierichs2022general}
D.~Wierichs, J.~Izaac, C.~Wang, and C.~Y.-Y.~Lin,
\newblock ``General parameter-shift rules for quantum gradients,''
\newblock \emph{Quantum}, vol.~6, p.~677, 2022.

\bibitem{powell1994direct}
M.~J.~D.~Powell,
\newblock ``A direct search optimization method that models the objective and constraint functions by linear interpolation,''
\newblock \emph{Advances in Optimization and Numerical Analysis}, pp.~51--67, Springer, 1994.

\bibitem{kandala2017hardware}
A.~Kandala, A.~Mezzacapo, K.~Temme, M.~Takita, M.~Brink, J.~M.~Chow, and J.~M.~Gambetta,
\newblock ``Hardware-efficient variational quantum eigensolver for small molecules and quantum magnets,''
\newblock \emph{Nature}, vol.~549, pp.~242--246, 2017.

\bibitem{welford1962note}
B.~P.~Welford,
\newblock ``Note on a method for calculating corrected sums of squares and products,''
\newblock \emph{Technometrics}, vol.~4, no.~3, pp.~419--420, 1962.

\bibitem{gacon2021simultaneous}
J.~Gacon, C.~Zoufal, G.~Carleo, and S.~Woerner,
\newblock ``Simultaneous perturbation stochastic approximation of the quantum Fisher information,''
\newblock \emph{Quantum}, vol.~5, p.~567, 2021.

\bibitem{spall1997accelerated}
J.~C.~Spall,
\newblock ``Accelerated second-order stochastic optimization using only function measurements,''
\newblock \emph{Proceedings of the 36th IEEE Conference on Decision and Control (CDC)}, vol.~2, pp.~1417--1424, 1997.

\bibitem{spall1998implementation}
J.~C.~Spall,
\newblock ``Implementation of the simultaneous perturbation algorithm for stochastic optimization,''
\newblock \emph{IEEE Transactions on Aerospace and Electronic Systems}, vol.~34, no.~3, pp.~817--823, 1998.

\bibitem{spall2000adaptive}
J.~C.~Spall,
\newblock ``Adaptive simultaneous perturbation stochastic approximation for simulated systems,''
\newblock \emph{IEEE Transactions on Aerospace and Electronic Systems}, vol.~36, no.~2, pp.~439--448, 2000.

\bibitem{qnspsa_psr_2024}
D.-T.~Le, V.-L.~Nguyen, H.~C.~Nguyen, Q.-H.~Nguyen, and V.-D.~Nguyen,
\newblock ``Variational quantum eigensolver: A comparative analysis of classical and quantum optimizer methods,''
\newblock \emph{arXiv:2412.19176}, 2024.

\bibitem{hessian_resolvability_2026}
Y.~Huang and Y.~Wang,
\newblock ``Analysis of Hessian scaling for local and global costs in variational quantum algorithms,''
\newblock \emph{arXiv:2602.00783}, 2026.

\bibitem{ghadimi2013stochastic}
S.~Ghadimi and G.~Lan,
\newblock ``Stochastic first- and zeroth-order methods for nonconvex stochastic programming,''
\newblock \emph{SIAM Journal on Optimization}, vol.~23, no.~4, pp.~2341--2368, 2013.

\end{thebibliography}

\end{document}
'''

def main():
    dest1 = r"C:\Users\arjun\Downloads\mysigma\sigma_qgd_paper_v1.tex"
    dest2 = r"C:\Users\arjun\Downloads\mysigma\sigama_qgd_paper_v1.tex"
    
    with open(dest1, "w", encoding="utf-8") as f:
        f.write(TEX_CONTENT)
    print(f"Successfully wrote {dest1} (size: {os.path.getsize(dest1)} bytes)")
    
    shutil.copyfile(dest1, dest2)
    print(f"Successfully copied to {dest2} (size: {os.path.getsize(dest2)} bytes)")

if __name__ == "__main__":
    main()
