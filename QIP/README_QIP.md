# Quantum Information Processing (QIP) Submission Package

This directory (`QIP/`) contains the complete, self-contained manuscript and asset package formatted strictly according to the author guidelines for **Quantum Information Processing** (Springer Nature, Journal no. `11128`, ISSN: `1570-0755` / `1573-1332`).

Official Journal Portal: [https://link.springer.com/journal/11128](https://link.springer.com/journal/11128)  
Public Code & Data Repository: [https://github.com/Arjun-E-Naik/SIGMA-QGD](https://github.com/Arjun-E-Naik/SIGMA-QGD)

---

## 1. Directory Structure

```text
QIP/
├── main.tex                    # Official Springer Nature class (sn-jnl) manuscript
├── main_standard.tex           # Standard LaTeX article fallback (identical content & declarations)
├── sn-jnl.cls                  # Official Springer Nature document class
├── sn-mathphys.bst             # Springer Nature Physical Sciences BibTeX style
├── references.bib              # Complete BibTeX bibliography with all permanent DOIs & URLs
├── figures/                    # Vector PDF and 300+ DPI PNG figures referenced in manuscript
│   ├── fig_ablation_consolidated.pdf / .png
│   ├── fig_benchmark_boxplots_consolidated.pdf / .png
│   ├── fig_benchmark_convergence_consolidated.pdf / .png
│   ├── fig_benchmark_efficiency_consolidated.pdf / .png
│   ├── fig_hessian_correlation_consolidated.pdf / .png
│   ├── fig_lr_sweep_consolidated.pdf / .png
│   ├── fig_sensitivity_consolidated.pdf / .png
│   └── fig_shot_noise_cma_consolidated.pdf / .png
└── README_QIP.md               # This submission guide
```

---

## 2. Theoretical & Mathematical Integrity Audit

| Item | Canonical Paper Label | Mathematical Statement & Proof Summary |
| :--- | :--- | :--- |
| **Assumption 1** | Local Quadraticity | $E$ satisfies 2nd-order Taylor expansion on window $W_t$ with $L$-Lipschitz Hessian remainder $\|\mathbf{R}(\boldsymbol{\theta}_\tau)\| \le \frac{L}{2}\|\boldsymbol{\theta}_\tau - \boldsymbol{\theta}^*\|^2$. |
| **Assumption 2** | Aggregate Diagonal Dominance | Off-diagonal Hessian elements $|H_{ij}| \le \delta$ satisfy $\delta\sqrt{p} \ll |H_{ii}|$ for active coordinates. |
| **Assumption 3** | Bounded Window Displacement | Trajectory displacement on window $W_t$ satisfies $\|\boldsymbol{\theta}_\tau - \boldsymbol{\theta}^*\| \le r$. |
| **Assumption 4** | Bounded Gradient and Curvature | Coordinate gradients and diagonal Hessian entries satisfy $|g_i| \le M$ and $|H_{ii}| \le M$ with $M \le \sum_k |c_k|$. |
| **Lemma 1** | Local Secant Relation | Gradient difference $g_i(\boldsymbol{\theta}_\tau) - g_i(\boldsymbol{\theta}^*) = H_{ii}(\boldsymbol{\theta}^*)(\theta_{\tau,i} - \theta_i^*) + \varepsilon_{i,\tau}$ with $|\varepsilon_{i,\tau}| \le \delta r\sqrt{p} + \frac{L}{2}r^2$. |
| **Proposition 1** | Variance--Curvature Proportionality | $\hat{\sigma}_i^2(t) = H_{ii}(\boldsymbol{\theta}^*)^2 \operatorname{Var}_{W_t}[\theta_i] + \mathcal{R}_i$, where $|\mathcal{R}_i| = \mathcal{O}((M\delta\sqrt{p} + \delta^2 p)r^2 + MLr^3 + L^2 r^4)$. |
| **Remark 1** | Scaling & Welford Decomposition | Path-averaged squared diagonal curvature plus macroscopic drift via Law of Total Variance. |
| **Proposition 2** | Uniform Preconditioner Boundedness | $\lambda_{\text{reg}} \le F_i(t) \le F_{\max} \equiv 2c G_{\max}^2 + \lambda_{\text{reg}}$ and $\frac{|\hat{m}_{t,i}|}{F_{\max}} \le |d_{t,i}| \le \frac{|\hat{m}_{t,i}|}{\lambda_{\text{reg}}}$. |
| **Proposition 3** | Descent Direction Guarantee | $\langle \mathbf{d}_t, -\mathbf{g}_t \rangle > 0$ under coordinate margin condition $\sum_{i \in \mathcal{S}_+} \frac{\hat{m}_{t,i}(-g_{t,i})}{F_i(t)} > \sum_{j \in \mathcal{S}_-} \frac{|\hat{m}_{t,j} g_{t,j}|}{F_j(t)}$. |
| **Assumption 5** | Coordinate-wise Momentum Alignment | $\hat{m}_{t,i}(-g_{t,i}) \ge 0$ for all $i$ ($\mathcal{S}_- = \emptyset$), yielding unconditional descent. |
| **Remark 2** | Alignment vs. Coordinate Margin | 2D counterexample demonstrating why unscaled inner product alignment $\langle \hat{\mathbf{m}}_t, -\mathbf{g}_t \rangle > 0$ fails with ill-conditioned preconditioners without the coordinate-wise margin. |
| **Theorem 1** | Ergodic Stationarity Convergence Rate | $\min_{1 \le t \le T} \|\nabla E(\boldsymbol{\theta}_{t-1})\|_2^2 = \mathcal{O}(1/T)$ in noiseless setting, and $\mathcal{O}(1/\sqrt{T})$ under finite-shot quantum measurement noise. |

---


