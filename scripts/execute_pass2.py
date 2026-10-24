# -*- coding: utf-8 -*-
"""
Robust execution script for Pass 2:
Extracts exact old text slices from sigma_qgd_paper_v1.tex,
applies the new humanized prose blocks,
and synchronizes both sigma_qgd_paper_v1.tex and sigama_qgd_paper_v1.tex.
"""

import os
import re

V1_PATH = r"C:\Users\arjun\Downloads\mysigma\sigma_qgd_paper_v1.tex"
V1_ALT_PATH = r"C:\Users\arjun\Downloads\mysigma\sigama_qgd_paper_v1.tex"

with open(V1_PATH, "r", encoding="utf-8") as f:
    text = f.read()

def get_block(src, start_marker, end_marker):
    p1 = src.find(start_marker)
    if p1 == -1:
        raise ValueError(f"Start marker not found: {start_marker[:60]}")
    p2 = src.find(end_marker, p1)
    if p2 == -1:
        raise ValueError(f"End marker not found: {end_marker[:60]}")
    p2 += len(end_marker)
    return src[p1:p2]

# Define the 18 new replacement blocks
b1_new = r"""\begin{remark}[Aggregate Alignment vs. Coordinate-wise Margin]
A positive overall inner product $\langle \hat{\mathbf{m}}_t, -\mathbf{g}_t \rangle > 0$ does not guarantee that a diagonal preconditioner will descend. We can see this in a simple two-dimensional example. Take $\mathbf{F} = (0.1, 0.005)^\top$ and $\hat{\mathbf{m}}_t \odot (-\mathbf{g}_t) = (1.0, -0.5)^\top$. The unscaled vectors align:
\[
\langle \hat{\mathbf{m}}_t, -\mathbf{g}_t \rangle = 1.0 - 0.5 = 0.5 > 0.
\]
Yet the preconditioned inner product turns negative:
\[
\langle \mathbf{d}_t, -\mathbf{g}_t \rangle = \frac{1.0}{0.1} + \frac{-0.5}{0.005} = 10 - 100 = -90 < 0.
\]
The second coordinate has an opposing sign. Because its preconditioner $F_i$ is tiny, that coordinate blows up and drives the update uphill. While coordinate-wise alignment (Assumption~\ref{ass:alignment}) offers a simple sufficient test, the true necessary and sufficient criterion is the margin condition in Proposition~\ref{prop:descent}.
\end{remark}"""

b2_new = r"""\begin{remark}[Dynamical Variance Scaling, Energy Rescaling, and Window Formulation]
\label{rem:variance_scaling}
Three practical points clarify how this mechanism behaves:
\begin{itemize}
    \item \textbf{Curvature vs. Trajectory Dampening and Hamiltonian Energy Scale:} Standard Newton-Raphson methods scale updates inversely with the first power of curvature ($H_{ii}^{-1}$). In SIGMA-QGD, the variance proxy instead links to $H_{ii}^2 \operatorname{Var}[\theta_i]$. Updates therefore scale approximately as $\hat{m}_{t,i} / (c H_{ii}^2 \operatorname{Var}[\theta_i] + \lambda_{\text{reg}})$. Along coordinates with steep curvature, rapid oscillations inflate the denominator quadratically. This curbs runaway steps and stabilizes descent. Along flat coordinates ($H_{ii} \approx 0$), the step size stays safely bounded by $\lambda_{\text{reg}}^{-1}$. In the variance-dominated regime where $d_i \sim \hat{m}_i / \hat{\sigma}_i^2 \propto 1/g_i$, the update vector is not invariant under an arbitrary scalar rescaling of the Hamiltonian ($H \to \alpha H$). This behavior matches unnormalized second-order methods. It ties good values of $c$ and $\lambda_{\text{reg}}$ to the characteristic energy scale $\sum_k |c_k|$ of the Hamiltonian. When working with arbitrary spectral norm $\Lambda = \sum_k |c_k|$, one can restore scale invariance by dividing the energy by $\Lambda$ before optimizing ($\tilde{E}(\boldsymbol{\theta}) = E(\boldsymbol{\theta}) / \Lambda$) or by setting $\lambda_{\text{reg}} \propto \Lambda^2$.
    \item \textbf{Trajectory-Dependent Displacement:} The term $H_{ii}^2 \operatorname{Var}_{W_t}[\theta_i]$ depends on the coordinate variance $\operatorname{Var}_{W_t}[\theta_i]$ traversed by the optimizer. Because of this, $\hat{\sigma}_i^2$ reflects curvature along the actual descent path rather than at a single isolated point.
    \item \textbf{Sliding Window vs. Online Accumulating Welford Estimator:} Our proofs in Assumptions~\ref{ass:quadraticity}--\ref{ass:displacement} and Proposition~\ref{prop:variance_curvature} use a compact sliding window $W_t$ of width $w$. This models the active descent phase ($t \in [1, t_0]$) where parameter displacement is large ($r = \Theta(1)$). In practice, Algorithm~\ref{alg:sigma_qgd} runs an online Bessel-corrected Welford accumulator over all prior steps ($t \ge 2$). It requires only $\mathcal{O}(1)$ time and memory per step. When the optimizer enters a basin near a local minimum ($t > t_0$), parameter movements shrink ($\|\boldsymbol{\theta}_\tau - \boldsymbol{\theta}^*\| \to 0$). In a narrow sliding window of width $w$, displacement variance $\operatorname{Var}_{W_t}[\theta_i]$ would vanish, pushing $\hat{\sigma}_i^2 \to 0$ and dropping the preconditioner to the bare floor $\lambda_{\text{reg}}$. In contrast, the full-history Welford accumulator $\mathbf{S}_t / (t-1)$ keeps memory of the curvature directions explored earlier in the trajectory. This memory prevents the preconditioner from collapsing late in the run, preserving directional scaling even as parameters slow down.
\end{itemize}
\end{remark}"""

b3_new = r"""\section{Experimental Methodology}
\label{sec:setup}

\subsection{Benchmark Hamiltonians}
We test the algorithms on two standard spin chains with periodic boundary conditions:
\begin{itemize}
    \item \textbf{1D Transverse-Field Ising Model (TFIM):}
    \begin{equation}
    H_{\text{TFIM}} = -J \sum_{i=1}^n Z_i Z_{i+1} - h \sum_{i=1}^n X_i, \quad J = 1.0, \, h = 0.5.
    \end{equation}
    Exact ground states: $E_0 = -10.6356$ ($n=10$), $E_0 = -12.7626$ ($n=12$).
    \item \textbf{1D Heisenberg XXZ Spin Chain:}
    \begin{equation}
    H_{\text{XXZ}} = J_{xy} \sum_{i=1}^n (X_i X_{i+1} + Y_i Y_{i+1}) + J_z \sum_{i=1}^n Z_i Z_{i+1} + h \sum_{i=1}^n Z_i,
    \end{equation}
    with $J_{xy} = 1.0, J_z = 0.5, h = 0.1$. Exact ground states: $E_0 = -15.2761$ ($n=10$), $E_0 = -18.2291$ ($n=12$).
\end{itemize}

\subsection{Ansatz and Initialization}
We set up a hardware-efficient ansatz of depth $r=3$. This layout gives $p=30$ trainable rotation angles for $n=10$ qubits and $p=36$ angles for $n=12$ qubits. In every trial, we draw the initial parameter vector $\boldsymbol{\theta}_0$ from a uniform distribution ($\theta_{0,i} \sim \text{Uniform}(-\pi/8, \pi/8)$). All competing optimizers start from identical random seeds to keep comparisons direct and paired.

\subsection{Baseline Algorithms and Circuit Accounting}
We test 11 optimizers in total (SIGMA-QGD alongside 10 baselines) while tracking every quantum circuit call:
\begin{itemize}
    \item \textbf{First-order baselines ($2p+1$ circuits/step):} Adam \cite{kingma2014adam}, AdaBelief \cite{zhuang2020adabelief}, RMSProp \cite{tieleman2012lecture}, and Yogi \cite{zaheer2018adaptive}.
    \item \textbf{Exact second-order baselines ($2p+1$ circuits/step):} Hess-Mag and Diag-Hessian (via Eq.~\eqref{eq:exact_hessian} \cite{wierichs2022general}).
    \item \textbf{Quantum Natural Gradient ($3p+2$ circuits/step):} Diag-QNG \cite{stokes2020quantum,wierichs2020avoiding}.
    \item \textbf{Derivative-free baselines:} COBYLA ($1$ call/step \cite{powell1994direct}), SPSA ($3$ calls/step \cite{spall1992multivariate,spall1997accelerated,spall1998implementation,spall2000adaptive}), QN-SPSA ($9$ calls/step \cite{gacon2021simultaneous}), and CMA-ES (budget-matched \cite{novak2025benchmarking}).
\end{itemize}

\subsection{Statistical Significance Testing}
To measure differences between SIGMA-QGD and each baseline, we run two-sided paired Wilcoxon signed-rank tests. We adjust all reported $p$-values with the Holm--Bonferroni step-down correction to control family-wise error across the 10 pairwise comparisons."""

b4_new = r"""The Heisenberg XXZ model provides a tough optimization testbed. Its non-commuting terms ($X_i X_{i+1} + Y_i Y_{i+1}$) generate strong qubit entanglement and dense off-diagonal curvature. Even with this frustration, SIGMA-QGD reaches the lowest mean energy gap across all 11 methods in both system sizes: $\Delta E = 1.4889 \pm 0.7845$ at 10 qubits and $\Delta E = 1.8408 \pm 0.8336$ at 12 qubits."""

b5_new = r"""At 12 qubits, SIGMA-QGD maintains the lowest mean energy gap among all 11 evaluated algorithms ($\Delta E = 1.8408 \pm 0.8336$). Under Holm--Bonferroni corrections, its advantage is statistically significant over derivative-free and stochastic techniques (COBYLA, SPSA, QN-SPSA, all $p_{\text{Holm}} < 10^{-4}$) and over the instantaneous Hessian baseline Hess-Mag ($p_{\text{Holm}} = 0.045$). When compared with Adam ($p_{\text{Holm}} = 0.07$), Diag-Hessian ($p_{\text{Holm}} = 0.10$), AdaBelief ($p_{\text{Holm}} = 0.16$), Yogi ($p_{\text{Holm}} = 0.16$), and RMSProp ($p_{\text{Holm}} = 0.52$), SIGMA-QGD delivers lower mean error with overlapping error bars on this frustrated surface. It also matches Diag-QNG ($1.9602$, $p_{\text{Holm}} = 0.49$) in energy while using $34\%$ fewer circuits ($14,600$ vs. $22,000$), demonstrating superior practical efficiency."""

b6_new = r"""\paragraph{Budget-Matched Parity and Line-Search Constraints in Quantum Optimization.}
One might ask why second-order baselines like Diag-Hessian and Hess-Mag omit adaptive line searches (such as Armijo backtracking) or trust-region steps. The reason is circuit cost. On quantum hardware, checking the objective at trial points along an update ray $\boldsymbol{\theta} + \alpha \mathbf{d}$ requires fresh state preparation and measurement runs. Testing 3--5 trial points per iteration would increase total circuit usage by $300\%\text{--}500\%$. That would break our $2p+1$ per-step budget constraint. Furthermore, measurement shot noise makes Armijo tests unreliable; random energy fluctuations cause the condition to accept bad steps or loop endlessly. To keep comparisons fair and equal in budget, we tested each gradient method using its tuned optimal learning rate $\eta^*$ (Section~\ref{sec:lr_sweep_results}) alongside standard Levenberg-Marquardt damping ($\lambda_{\text{reg}} = 0.005$). Diag-Hessian struggles because trigonometric quantum landscapes oscillate rapidly. When instantaneous second derivatives $H_{ii}$ approach zero or go negative, the inverse multiplier surges ($g_i / \lambda_{\text{reg}}$), throwing the parameters out of the basin unless momentum and clipping rein it in."""

b7_new = r"""To test Diag-QNG under an identical circuit limit, we also examined its performance when stopped at the exact circuit budget given to SIGMA-QGD ($12,200$ circuits at 10q and $14,600$ circuits at 12q). That budget caps Diag-QNG at $T_{\text{QNG}} = \lfloor 12,200/92 \rfloor = 132$ iterations for 10 qubits and $\lfloor 14,600/110 \rfloor = 132$ iterations for 12 qubits. With equal circuit counts, Diag-QNG yields mean energy gaps of $\Delta E \approx 0.428$ (TFIM 10q), $0.441$ (TFIM 12q), $1.821$ (XXZ 10q), and $2.115$ (XXZ 12q). In Table~\ref{tab:main_benchmark_consolidated}, Diag-QNG had a $50\%$ circuit advantage ($T=200$) and still converged to higher energies. Enforcing strict circuit parity makes SIGMA-QGD's lead wider across all four benchmarks."""

b8_new = r"""\subsubsection{Circuit Pareto Frontier Analysis}
Figure~\ref{fig:benchmark_efficiency} shows the empirical Pareto frontier, plotting total quantum circuits against final log energy gap ($\log_{10} \Delta E$). Optimizers toward the lower-left corner deliver the best balance between execution time and solution accuracy.

Zero-order schemes (COBYLA at $200$ circuits, SPSA at $600$ circuits) finish rapidly but stall far from the ground state ($\Delta E > 0.5$ on TFIM, $\Delta E > 3.5$ on XXZ). At the other extreme, Diag-QNG consumes $50\%$ more circuits ($18,400$ circuits at 10q, $22,000$ at 12q) because of its extra metric evaluations. Even with that surplus, its TFIM gap ($0.4068$) is $44\times$ worse than SIGMA-QGD ($0.0092$) because it lacks momentum damping. SIGMA-QGD anchors the high-precision Pareto front in all four test cases. It finds the lowest energy while staying strictly within the standard gradient budget ($12,200$ circuits at 10q, $14,600$ circuits at 12q). Only COBYLA shares the frontier, occupying the low-cost 200-circuit corner."""

b9_new = r"""The ablation tests in Table~\ref{tab:ablation} and Figure~\ref{fig:ablation_consolidated} highlight several clear findings:
\begin{itemize}
    \item \textbf{Decisive Impact of the Variance Preconditioner:} Setting the preconditioner to the identity matrix ($\mathbf{F}=\mathbf{I}$) ruins performance on TFIM. The ground-state gap grows by $48.25\times$ at 10 qubits and $50.92\times$ at 12 qubits ($p_{\text{Holm}} = 2.44 \times 10^{-4}$ in both cases). On the XXZ model, turning off the preconditioner also hurts accuracy ($1.14\times$ larger gap at 10 qubits, $p_{\text{Holm}} = 0.008$; $1.21\times$ larger at 12 qubits, $p_{\text{Holm}} = 0.021$). Scaling steps by trajectory variance is necessary on both commuting and frustrated landscapes.
    \item \textbf{Synergy with Momentum:} Dropping momentum ($\beta=0$) similarly increases error by $44.44\times$ (10q) and $49.80\times$ (12q) on TFIM ($p_{\text{Holm}} = 2.44 \times 10^{-4}$), and by $1.41\times$ (10q) and $1.17\times$ (12q) on XXZ. The two mechanisms complement each other. Momentum damps quick parameter oscillations, while the preconditioner widens steps along narrow ravines.
    \item \textbf{Empirical Invariance to Gradient Clipping:} Removing gradient clipping ($c_{\text{clip}}=\infty$) leaves results unchanged from the baseline (ratios $0.99\times$ to $1.01\times$, $p > 0.60$). Parameter-shift gradients stay well within bounded limits on these circuits. The global norm cap $c_{\text{norm}} = \sqrt{p}$ serves only as a safety net that never interferes during normal optimization.
    \item \textbf{Drift Correction Omission Justified:} Adding velocity damping to the variance estimator changes final energies by less than $\pm 5\%$, none of which are statistically significant ($p_{\text{Holm}} \ge 0.188$). Tests with sliding windows or exponential moving averages also showed that short windows fluctuate heavily on late plateaus. In contrast, full-history Welford accumulation maintains a steady curvature baseline. Bessel-corrected Welford tracking gives a reliable signal across the entire trajectory without tuning extra velocity parameters or spending extra memory.
\end{itemize}"""

b10_new = r"""As shown in Figure~\ref{fig:shot_noise_cma_consolidated}(a) and Table~\ref{tab:shot_noise_cma_consolidated} (Part I), SIGMA-QGD shows a steady drop in error as shot budgets increase. Under practical measurement limits ($M=1{,}000$ and $10{,}000$ shots per circuit), SIGMA-QGD solves all four benchmark systems reliably:
\begin{itemize}
    \item \textbf{Monotonic Error Scaling:} On the 10-qubit TFIM, the final energy gap falls steadily from $0.4298$ at $M=100$ shots to $0.1086 \pm 0.0515$ at $M=1{,}000$ shots and $0.0324 \pm 0.0078$ at $M=10{,}000$ shots, nearing the noiseless limit ($0.0098$). The 12-qubit TFIM displays the same downward trend ($0.1356$ at $1{,}000$ shots, $0.0586$ at $10{,}000$ shots).
    \item \textbf{Analytical Shot-Noise Interaction:} With finite shots, each gradient evaluation carries measurement noise $\tilde{g}_i = g_i + \xi_i$, where $\operatorname{Var}[\xi_i] = \sigma_{\text{shot}}^2 / M$. The Welford accumulator tracks total variance $\hat{\sigma}_i^2 \approx \sigma_{\text{landscape}}^2 + \sigma_{\text{shot}}^2 / M$. Finite-shot sampling adds an automatic noise floor that scales as $1/M$. Instead of throwing descent off track, this extra variance increases $F_i(t)$. It scales down step sizes when gradient noise is loud and fades away as shot counts rise.
    \item \textbf{Resilience on Frustrated Spin Chains:} On the Heisenberg XXZ chain, the final gap settles at $1.7688$ ($1{,}000$ shots) and $1.4037$ ($10{,}000$ shots) for 10 qubits. Both values match the noiseless statevector baseline ($1.3565 \pm 0.6480$) within statistical error.
\end{itemize}"""

b11_new = r"""In our direct comparison with CMA-ES under matched circuit budgets (Figure~\ref{fig:shot_noise_cma_consolidated}(b) and Table~\ref{tab:shot_noise_cma_consolidated}, Part II), SIGMA-QGD beats CMA-ES across all tested settings ($p \le 0.0125$, using step size $\sigma_0 = 0.3$ and population size $\lambda = 14$). Even though CMA-ES updates its distribution over $4\times$ more generations ($\sim 870$ generations compared to $200$ gradient iterations at $p=30$), shot noise misleads its candidate rankings. In contrast, accumulating variance over time and smoothing updates with momentum keeps SIGMA-QGD steady against shot noise."""

b12_new = r"""\subsection{Dimensional Scaling Behavior}
\label{sec:scaling_results}
We next examine how optimization holds up as system size grows from $n=10$ ($p=30$) to $n=12$ ($p=36$). Adding two qubits quadruples the state space from $2^{10} = 1,024$ to $2^{12} = 4,096$ basis states, while expanding the parameter vector from $30$ to $36$ rotation angles. 

Table~\ref{tab:dimensional_scaling} reports how performance changes across key optimizer families on both TFIM and XXZ spin chains ($N=20$ independent seeds, $200$ steps)."""

b13_new = r"""On the TFIM model, SIGMA-QGD yields an energy gap an order of magnitude lower than all baselines ($\Delta E \approx 0.011$). Its gap increases by only $1.22\times$ despite the $4\times$ larger Hilbert space. By comparison, Adam and AdaBelief show noticeably worse scaling ($1.58\times$), leaving final gaps roughly $12\times$ larger than SIGMA-QGD ($0.142\text{--}0.170$ vs. $0.011$). On the XXZ model, SIGMA-QGD again delivers the lowest final gap ($1.841$) and scales stably across the transition ($1.24\times$)."""

b14_new = r"""Does SIGMA-QGD outperform same-budget second-order baselines because of \emph{what} it measures (gradient variance rather than instantaneous curvature) or because of \emph{how} it gathers data over time (a temporal running average instead of a single-point snapshot)? Both factors matter, but their roles differ."""

b15_new = r"""Hess-Mag and Diag-Hessian compute the exact diagonal second derivatives $H_{ii}(\boldsymbol{\theta}_t)$ at each step using Eq.~\eqref{eq:exact_hessian}. On non-convex quantum surfaces, especially with non-commuting Pauli operators or shot noise, these instantaneous derivatives change wildly between iterations. Near inflection points where $H_{ii}$ drops close to zero, the unconditioned inverse factor $1/(|H_{ii}| + \lambda_{\text{reg}})$ spikes violently. This surge triggers wild parameter jumps and derails optimization."""

b16_new = r"""SIGMA-QGD takes a different route by computing the running variance $\hat{\sigma}_i^2 = \operatorname{Var}_{W_t}[g_i]$ with Welford's algorithm along the path taken. This design provides two key benefits:
\begin{itemize}
    \item \textbf{Low-Pass Curvature Regularization:} Gathering variance over the parameter path filters out high-frequency gradient jitter without running additional quantum circuits.
    \item \textbf{Path-Integrated Sensitivity vs. Static Point Curvature:} As shown in Proposition~\ref{prop:variance_curvature}, $\hat{\sigma}_i^2 \approx H_{ii}^2 \operatorname{Var}_{W_t}[\theta_i]$. Instead of sampling point curvature at an isolated coordinate $\boldsymbol{\theta}_t$, trajectory variance captures parameter sensitivity along the entire descent path. Parameters that swung through large gradient shifts receive smaller, safer step sizes, fitting the update directly to the traversed terrain.
\end{itemize}
Temporal smoothing is not an accidental byproduct. It is a core structural feature that stabilizes optimization across quantum landscapes without extra circuit executions. This agrees with earlier theoretical studies of curvature regularization in variational algorithms \cite{stokes2020quantum,wierichs2020avoiding}, which showed that raw second-order updates easily diverge on non-convex surfaces unless damped along poorly conditioned coordinates."""

b17_new = r"""\subsection{Ansatz Expressivity and Landscape Frustration}
On the Heisenberg XXZ Hamiltonian, diagonal QNG and adaptive first-order baselines reach final energy gaps close to SIGMA-QGD ($\Delta E \approx 1.5\text{--}1.8$). This remaining gap highlights the boundary between what an optimizer can find and what a circuit can express:
\begin{itemize}
    \item \textbf{Ansatz Expressivity vs. Optimization Limits:} The 1D Heisenberg XXZ chain has gapless critical ground states and entangled structures that a shallow $r=3$ hardware-efficient $RY$-CNOT circuit cannot capture exactly. The residual gap of $\sim 1.5$ stems from the expressivity limit of the circuit family itself, not from early convergence failure.
    \item \textbf{Shallow NISQ Regimes vs. Barren Plateaus:} Deeper circuits ($r \ge 6$) could express the state more accurately, but they run into noise-induced barren plateaus on present-day quantum processors. By testing on this shallow, frustrated model, we test whether optimizers can locate the best parameter set possible for a realistic circuit without stalling in false local minima or oscillating out of control.
    \item \textbf{Boundary of Diagonal Approximations:} Our results also mark where diagonal curvature methods reach their limit. In models dominated by non-local Pauli exchanges, diagonal matrices capture only part of the full geometric curvature. This explains why diagonal QNG and SIGMA-QGD eventually settle onto similar energy plateaus.
\end{itemize}"""

b18_new = r"""\subsection{Limitations}
We note two main boundaries of our study:
\begin{itemize}
    \item \textbf{Ansatz Specificity:} Our benchmarks focused on hardware-efficient $RY$-CNOT circuits. Problem-specific designs, such as unitary coupled-cluster operators for quantum chemistry, warrant dedicated follow-up experiments.
    \item \textbf{Stationary Point vs. Global Basin Guarantees:} Theorem~\ref{thm:convergence} proves an ergodic $\mathcal{O}(1/T)$ rate to first-order stationary points ($\min_{1 \le t \le T} \|\nabla E(\boldsymbol{\theta}_{t-1})\|_2^2 \le \epsilon$) under uniform margin conditions. Proving convergence to the global ground-state basin on non-convex, frustrated quantum manifolds remains an open theoretical challenge.
\end{itemize}"""

# Extract the corresponding old blocks from text
blocks = [
    ("Block 1 (Remark 1)", get_block(text, r"\begin{remark}[Aggregate Alignment vs. Coordinate-wise Margin]", r"\end{remark}"), b1_new),
    ("Block 2 (Remark 2)", get_block(text, r"\begin{remark}[Dynamical Variance Scaling", r"\end{remark}"), b2_new),
    ("Block 3 (Sec 4.1-4.4)", get_block(text, r"\section{Experimental Methodology}", r"Holm--Bonferroni stepdown corrections to all reported $p$-values."), b3_new),
    ("Block 4 (Sec 5.4.2 p1)", get_block(text, r"The Heisenberg XXZ model represents an exceptionally challenging optimization landscape", r"$\Delta E = 1.8408 \pm 0.8336$ at 12 qubits."), b4_new),
    ("Block 5 (Sec 5.4.2 p3)", get_block(text, r"At 12 qubits, SIGMA-QGD preserves the lowest mean energy gap across all 11 evaluated optimizers", r"confirming superior Pareto efficiency."), b5_new),
    ("Block 6 (Sec 5.4.2 budget-matched)", get_block(text, r"\paragraph{Budget-Matched Parity and Line-Search Constraints in Quantum Optimization.}", r"mitigated by temporal momentum smoothing and gradient clipping."), b6_new),
    ("Block 7 (Sec 5.4.3 p2)", get_block(text, r"Moreover, to evaluate Diag-QNG under a strictly circuit-matched constraint", r"widens SIGMA-QGD's performance advantage across all four system benchmarks."), b7_new),
    ("Block 8 (Sec 5.4.4 Pareto)", get_block(text, r"\subsubsection{Circuit Pareto Frontier Analysis}", r"only COBYLA remaining non-dominated at the minimal 200-circuit allocation."), b8_new),
    ("Block 9 (Sec 5.5 Ablation bullets)", get_block(text, r"The consolidated results in Table~\ref{tab:ablation} and Figure~\ref{fig:ablation_consolidated} establish several clear empirical conclusions regarding the algorithmic architecture:", r"explicit velocity detrending.") + "\n\\end{itemize}", b9_new),
    ("Block 10 (Sec 5.6 Shot noise bullets)", get_block(text, r"As shown in Figure~\ref{fig:shot_noise_cma_consolidated}(a) and Table~\ref{tab:shot_noise_cma_consolidated} (Part I)", r"within statistical error.") + "\n\\end{itemize}", b10_new),
    ("Block 11 (Sec 5.6 CMA-ES)", get_block(text, r"In the budget-matched head-to-head comparison against CMA-ES", r"inherent robustness against shot noise."), b11_new),
    ("Block 12 (Sec 5.7 Intro)", get_block(text, r"\subsection{Dimensional Scaling Behavior}", r"$N=20$ independent random seeds, $200$ steps)."), b12_new),
    ("Block 13 (Sec 5.7 Conclusion)", get_block(text, r"On the TFIM model, SIGMA-QGD maintains an optimality gap an order of magnitude lower", r"scaling stably across the transition ($1.24\times$)."), b13_new),
    ("Block 14 (Sec 6.1 p1)", get_block(text, r"A key question is whether SIGMA-QGD's advantage over same-budget second-order baselines", r"temporal accumulator vs. single-point evaluation)."), b14_new),
    ("Block 15 (Sec 6.1 p2)", get_block(text, r"Both Hess-Mag and Diag-Hessian compute the exact instantaneous parameter-shift Hessian diagonal", r"destabilize convergence."), b15_new),
    ("Block 16 (Sec 6.1 p4 & bullets)", get_block(text, r"In contrast, the Welford accumulator in SIGMA-QGD computes running sample variance", r"dampened along ill-conditioned directions."), b16_new),
    ("Block 17 (Sec 6.2 Ansatz expressivity)", get_block(text, r"\subsection{Ansatz Expressivity and Landscape Frustration}", r"similar asymptotic energy plateaus.") + "\n\\end{itemize}", b17_new),
    ("Block 18 (Sec 6.3 Limitations)", get_block(text, r"\subsection{Limitations}", r"non-convex quantum manifolds.") + "\n\\end{itemize}", b18_new),
]

modified_text = text
for name, old_b, new_b in blocks:
    count = modified_text.count(old_b)
    if count != 1:
        print(f"ERROR: {name} count is {count} (expected 1)!")
        exit(1)
    modified_text = modified_text.replace(old_b, new_b)
    print(f"SUCCESS: Replaced {name} cleanly (len {len(old_b)} -> {len(new_b)}).")

print(f"Original len: {len(text)}, New len: {len(modified_text)}")

with open(V1_PATH, "w", encoding="utf-8") as f:
    f.write(modified_text)

with open(V1_ALT_PATH, "w", encoding="utf-8") as f:
    f.write(modified_text)

print(f"Successfully updated {V1_PATH} and {V1_ALT_PATH}!")
