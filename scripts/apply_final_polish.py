# -*- coding: utf-8 -*-
"""
Apply final polish to sigma_qgd_paper_v1.tex and sigama_qgd_paper_v1.tex:
1. Section 5.5 Bullet 2: change 'widens steps along narrow ravines' to 'rescales steps across narrow ravines'
2. Section 6.1 Paragraph 3: humanize the remaining formal prose
3. Replace raw unicode em-dashes (\u2014) with LaTeX '---' for 100% pure ASCII Overleaf/pdflatex compatibility
4. Verify 0 non-breaking spaces (U+00A0)
"""

V1_PATH = r"C:\Users\arjun\Downloads\mysigma\sigma_qgd_paper_v1.tex"
V1_ALT_PATH = r"C:\Users\arjun\Downloads\mysigma\sigama_qgd_paper_v1.tex"

with open(V1_PATH, "r", encoding="utf-8") as f:
    text = f.read()

# 1. Polish Section 5.5 Bullet 2
old_p55 = "Momentum damps quick parameter oscillations, while the preconditioner widens steps along narrow ravines."
new_p55 = "Momentum damps quick parameter oscillations, while the preconditioner rescales steps across narrow ravines."

assert text.count(old_p55) == 1, f"Expected 1 occurrence of old_p55, found {text.count(old_p55)}"
text = text.replace(old_p55, new_p55)
print("SUCCESS: Polished Section 5.5 Bullet 2 ('widens' -> 'rescales').")

# 2. Polish Section 6.1 Paragraph 3
old_sec61_p3 = r"""A natural question is whether augmenting instantaneous Hessian baselines with first-order momentum (analogous to quasi-Newton methods with Polyak or EMA momentum) would resolve this instability. In practice, it cannot. If the instantaneous second derivative vanishes ($H_{ii}(\boldsymbol{\theta}_t) \approx 0$), the diagonal preconditioning factor $(|H_{ii}(\boldsymbol{\theta}_t)| + \lambda_{\text{reg}})^{-1}$ inflates to its upper regularization bound $\lambda_{\text{reg}}^{-1} = 200$ (under operational $\lambda_{\text{reg}} = 0.005$). Multiplying this step multiplier by an accumulated momentum vector $\hat{\mathbf{m}}_t$ does not stabilize the step. Instead, it injects a $200\times$ step-size spike along coordinate $i$, pushing parameters out of the local basin. In contrast, SIGMA-QGD preconditions using temporal variance $\hat{\sigma}_i^2(t)$, which aggregates gradient dispersion over the entire descent history. Because the cumulative sum of squared deviations $\sum_{\tau=1}^t (g_{c,\tau,i} - \bar{g}_{t,i})^2$ monotonically accumulates along any non-trivial trajectory, $\hat{\sigma}_i^2(t)$ never collapses to near-zero at isolated inflection points, completely avoiding the singularity spikes inherent to instantaneous Hessian inversion."""

new_sec61_p3 = r"""One might ask whether adding first-order momentum to instantaneous Hessian baselines (similar to quasi-Newton methods with Polyak or EMA momentum) would fix this instability. In practice, it cannot. When the instantaneous second derivative drops near zero ($H_{ii}(\boldsymbol{\theta}_t) \approx 0$), the diagonal scaling factor $(|H_{ii}(\boldsymbol{\theta}_t)| + \lambda_{\text{reg}})^{-1}$ hits its upper ceiling $\lambda_{\text{reg}}^{-1} = 200$ (for $\lambda_{\text{reg}} = 0.005$). Multiplying this large factor by a momentum vector $\hat{\mathbf{m}}_t$ does not calm the update; instead, it creates a $200\times$ step-size jump along coordinate $i$, pushing parameters out of the local basin. By comparison, SIGMA-QGD scales updates using temporal variance $\hat{\sigma}_i^2(t)$, which gathers gradient spread over the full descent history. Because the sum of squared deviations $\sum_{\tau=1}^t (g_{c,\tau,i} - \bar{g}_{t,i})^2$ grows steadily along any active trajectory, $\hat{\sigma}_i^2(t)$ never drops to zero at isolated inflection points. This completely avoids the singular spikes seen in instantaneous Hessian inversion."""

assert text.count(old_sec61_p3) == 1, f"Expected 1 occurrence of old_sec61_p3, found {text.count(old_sec61_p3)}"
text = text.replace(old_sec61_p3, new_sec61_p3)
print("SUCCESS: Polished Section 6.1 Paragraph 3.")

# 3. Unicode em-dash replacement to standard LaTeX '---'
em_dash_count = text.count("\u2014")
print(f"Found {em_dash_count} unicode em-dashes.")
if em_dash_count > 0:
    text = text.replace("\u2014", "---")
    print(f"SUCCESS: Replaced {em_dash_count} unicode em-dashes with '---'.")

# 4. Check for any non-breaking spaces (U+00A0)
nbsp_count = text.count("\u00a0")
print(f"Non-breaking spaces found: {nbsp_count}")
if nbsp_count > 0:
    text = text.replace("\u00a0", " ")
    print(f"Replaced {nbsp_count} non-breaking spaces with standard spaces.")

# 5. Check if any remaining non-ASCII characters exist
non_ascii = [c for c in text if ord(c) > 127]
print(f"Remaining non-ASCII characters count: {len(non_ascii)}")

# Save to both files
with open(V1_PATH, "w", encoding="utf-8") as f:
    f.write(text)

with open(V1_ALT_PATH, "w", encoding="utf-8") as f:
    f.write(text)

print(f"Successfully updated {V1_PATH} and {V1_ALT_PATH} (Length: {len(text)} chars).")
