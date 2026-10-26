# Quantum Information Processing (QIP) Submission Package

This folder (`QIP/`) contains the complete, self-contained manuscript and asset package formatted strictly according to the author guidelines for **Quantum Information Processing** (Springer Nature, Journal no. `11128`, ISSN: `1570-0755` / `1573-1332`).

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
├── references.bib              # Complete BibTeX bibliography with all permanent DOIs
├── figures/                    # Vector PDF and PNG figures referenced in manuscript
│   ├── fig_ablation_consolidated.pdf
│   ├── fig_ablation_consolidated.png
│   ├── fig_benchmark_boxplots_consolidated.pdf
│   ├── fig_benchmark_boxplots_consolidated.png
│   ├── fig_benchmark_convergence_consolidated.pdf
│   ├── fig_benchmark_convergence_consolidated.png
│   ├── fig_benchmark_efficiency_consolidated.pdf
│   ├── fig_benchmark_efficiency_consolidated.png
│   ├── fig_hessian_correlation_consolidated.pdf
│   ├── fig_hessian_correlation_consolidated.png
│   ├── fig_lr_sweep_consolidated.pdf
│   ├── fig_lr_sweep_consolidated.png
│   ├── fig_sensitivity_consolidated.pdf
│   ├── fig_sensitivity_consolidated.png
│   ├── fig_shot_noise_cma_consolidated.pdf
│   └── fig_shot_noise_cma_consolidated.png
└── README_QIP.md               # This submission guide
```

---

## 2. Compliance Checklist for Springer Nature / QIP

| Springer Requirement | Status | Implementation Details |
| :--- | :---: | :--- |
| **Document Class** | ✅ Satisfied | Uses `\documentclass[sn-mathphys,Numbered]{sn-jnl}` with standard Springer frontmatter (`\fnm`, `\sur`, `\affil*`, `\abstract`, `\keywords`). |
| **Author & Affiliation** | ✅ Satisfied | Structured metadata with corresponding author email (`bt24csd005@iiitn.ac.in`) and IIIT Nagpur institutional address. |
| **Declarations Section** | ✅ Satisfied | Standalone unnumbered section preceding references with all mandatory headings: Funding, Competing Interests, Author Contributions (CRediT), Data Availability, Code Availability, Ethics, AI Disclosure. |
| **AI Policy Compliance** | ✅ Satisfied | Fully compliant with Springer Nature generative AI policy. All figures are programmatic vector plots from Matplotlib (no generative AI images). Text and math derived directly by authors; AI used solely for syntax linting and grammar editing. |
| **Table Formatting** | ✅ Satisfied | Uses clean `booktabs` rules (`\toprule`, `\midrule`, `\bottomrule`). All wide tables use `table*` without shrinking via `\resizebox`, preventing unreadable small font sizes. |
| **Figures** | ✅ Satisfied | High-resolution vector graphics (`.pdf`) with 300+ DPI raster backups (`.png`). Vector fonts embedded. |
| **References & Citations** | ✅ Satisfied | All 28 cited literature entries are verified and furnished with permanent DOIs or arXiv identifiers. Numbered sequential citations matching QIP standard. |

---

## 3. How to Compile or Upload to Overleaf

### Option A: Upload to Overleaf (Recommended)
1. Compress the contents of `QIP/` into a single `.zip` file (include `main.tex`, `sn-jnl.cls`, `sn-mathphys.bst`, `references.bib`, and the `figures/` folder).
2. Go to [Overleaf](https://www.overleaf.com/) $\to$ **New Project** $\to$ **Upload Project**.
3. Select the `.zip` file.
4. Set the compiler to **pdfLaTeX** or **TeX Live 2023 / 2024**.
5. Compile `main.tex`.

### Option B: Local Compilation
If you have a local TeX distribution (TeX Live / MacTeX / MiKTeX):
```bash
cd QIP
pdflatex main.tex
pdflatex main.tex
```
*(If preferred, you may also compile `main_standard.tex` using any standard LaTeX environment without needing `sn-jnl.cls`)*.

---

## 4. Submission Steps for Editorial Manager (QIP)

When submitting through the Springer Editorial Manager portal:
1. **Article Type:** Select *Original Research Paper*.
2. **Title & Abstract:** Copy and paste the text from `main.tex`.
3. **Authors:**
   - Author 1 (Corresponding): Arjun E. Naik (`bt24csd005@iiitn.ac.in`), IIIT Nagpur.
   - Author 2: Jitendra Tembhurne (`jtembhurne@iiitn.ac.in`), IIIT Nagpur.
4. **Item Types in File Upload:**
   - `main.tex`: Choose item type **Manuscript**.
   - `sn-jnl.cls` and `sn-mathphys.bst`: Choose item type **LaTeX Style / Class File** (or Supplementary Material / Manuscript source).
   - Figures (`fig_*.pdf`): Upload as item type **Figure**.
   - `references.bib`: Upload as item type **LaTeX Source / BibTeX**.
5. **Declarations:** When prompted in the submission portal checkboxes, confirm:
   - Data is available at GitHub: `https://github.com/Arjun-E-Naik/SIGMA-QGD`.
   - Code is available at GitHub: `https://github.com/Arjun-E-Naik/SIGMA-QGD`.
   - No conflicting financial or non-financial interests.
