import os
import glob
import csv
import numpy as np
from scipy import stats

def analyze_hessian():
    print("=" * 60)
    print("HESSIAN CORRELATION SUMMARY")
    print("=" * 60)
    files = [
        ("TFIM 10q", "sigma_data/hessian_results/hessian_correlation_results_hesscorr_tfim_10q.csv"),
        ("TFIM 12q", "sigma_data/hessian_results/hessian_correlation_results_hesscorr_tfim_12q.csv"),
        ("XXZ 10q",  "sigma_data/hessian_results/hessian_correlation_results_hesscorr_xxz_10q.csv"),
        ("XXZ 12q",  "sigma_data/hessian_results/hessian_correlation_results_hesscorr_xxz_12q.csv"),
    ]
    for label, path in files:
        pearsons, spearmans = [], []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                pearsons.append(float(r["pearson_log"]))
                spearmans.append(float(r["spearman"]))
        p_arr = np.array(pearsons)
        s_arr = np.array(spearmans)
        w_p, p_p = stats.wilcoxon(p_arr)
        w_s, p_s = stats.wilcoxon(s_arr)
        print(f"\n[{label}] N={len(p_arr)} seeds:")
        print(f"  Pearson r (log-space): {p_arr.mean():.4f} +/- {p_arr.std(ddof=1):.4f} (min={p_arr.min():.4f}, max={p_arr.max():.4f}) | W={w_p:.1f}, p={p_p:.4e}")
        print(f"  Spearman rho:          {s_arr.mean():.4f} +/- {s_arr.std(ddof=1):.4f} (min={s_arr.min():.4f}, max={s_arr.max():.4f}) | W={w_s:.1f}, p={p_s:.4e}")

def analyze_ablation():
    print("\n" + "=" * 60)
    print("ABLATION SUMMARY")
    print("=" * 60)
    cases = [
        ("TFIM 10q", "sigma_data/ablation_results/ablation_results_ablation_tfim_10q.csv", "sigma_data/ablation_results/ablation_significance_ablation_tfim_10q.csv"),
        ("TFIM 12q", "sigma_data/ablation_results/ablation_results_ablation_tfim_12q.csv", "sigma_data/ablation_results/ablation_significance_ablation_tfim_12q.csv"),
        ("XXZ 10q",  "sigma_data/ablation_results/ablation_results_ablation_xxz_10q.csv",  "sigma_data/ablation_results/ablation_significance_ablation_xxz_10q.csv"),
        ("XXZ 12q",  "sigma_data/ablation_results/ablation_results_ablation_xxz_12q.csv",  "sigma_data/ablation_results/ablation_significance_ablation_xxz_12q.csv"),
    ]
    for label, res_path, sig_path in cases:
        print(f"\n--- {label} ---")
        sig_map = {}
        if os.path.exists(sig_path):
            with open(sig_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    sig_map[r["config"]] = r
        with open(res_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                cfg = r["config"]
                sig = sig_map.get(cfg, {})
                p_raw = sig.get("p_value", "---")
                p_holm = sig.get("p_value_holm", "---")
                try:
                    p_r_str = f"{float(p_raw):.2e}"
                except:
                    p_r_str = str(p_raw)
                try:
                    p_h_str = f"{float(p_holm):.2e}"
                except:
                    p_h_str = str(p_holm)
                print(f"  {cfg:22s} | Gap: {float(r['gap_mean']):.6f} +/- {float(r['gap_std']):.6f} | Ratio: {float(r['ratio']):6.2f}x | p_raw: {p_r_str:8s} | p_holm: {p_h_str:8s} | {r['interpretation']}")

if __name__ == "__main__":
    analyze_hessian()
    analyze_ablation()
