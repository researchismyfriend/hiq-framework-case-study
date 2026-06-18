"""
HIQ Bootstrap Confidence Intervals
====================================
Adds 95% bootstrapped confidence intervals to all three HIQ figures.

Method: Dirichlet-Multinomial parametric bootstrap on s_D
- For each month-cohort: known n (sessions) and s_D (observed diversity)
- Infer effective K (clusters) from pipeline's min_cluster_size heuristic
- Find Dirichlet concentration parameter alpha that matches observed s_D
- Bootstrap 200 samples: draw p ~ Dirichlet([alpha]*K), compute Hill-Chao
- Hold s_C and s_A fixed at point estimates (too expensive to bootstrap)
- HIQ_boot = harmonic_mean(s_D_boot, s_C_point, s_A_point)
- CI: 2.5th and 97.5th percentiles

This approach is valid when per-session cluster labels are not saved.
The Dirichlet model represents uncertainty from finite-sample cluster assignment.

Usage:
    python3 hiq_bootstrap_ci.py

Reads: ~/hiq_pipeline_output/*.csv
Writes: ~/hiq_pipeline_output/hiq_fig{1,2,3}_*_ci.jpg
        ~/hiq_pipeline_output/hiq_bootstrap_ci_summary.json
"""

import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

np.random.seed(42)

OUT = Path(os.path.expanduser('~/hiq_pipeline_output'))
N_BOOTSTRAP = 200

COLOURS = {
    'All Users':           '#2980b9',
    'Global North':        '#2980b9',
    'Global South':        '#c0392b',
    'LLM-Represented':     '#2980b9',
    'LLM-Underrepresented':'#c0392b',
}


# ── Hill-Chao diversity (matches pipeline exactly) ──────────────────────────
def hill_chao_from_counts(counts):
    """Compute s_D from an array of cluster counts (matching pipeline formula)."""
    total = counts.sum()
    if total < 2:
        return 0.0
    props = counts / total
    K = len(props)
    if K <= 1:
        return 0.0
    entropy = -np.sum(props * np.log(props + 1e-12))
    qD1 = np.exp(entropy)
    qD2 = 1.0 / np.sum(props ** 2)
    s1 = (qD1 - 1) / (K - 1)
    s2 = (qD2 - 1) / (K - 1)
    return float((s1 + s2) / 2)


# ── HIQ harmonic mean (matches pipeline exactly) ────────────────────────────
def hiq_composite(s_D, s_C, s_A, rho=-1, w_D=1/3, w_C=1/3, w_A=1/3):
    eps = 1e-6
    s_D = max(eps, s_D)
    s_C = max(eps, s_C)
    s_A = max(eps, s_A)
    return float((w_D * s_D**rho + w_C * s_C**rho + w_A * s_A**rho) ** (1/rho))


# ── Expected s_D under symmetric Dirichlet(alpha, K) ───────────────────────
def expected_sd_dirichlet(alpha, K, n_sim=300):
    """Monte Carlo estimate of E[s_D] under Dirichlet(alpha, K)."""
    results = []
    for _ in range(n_sim):
        p = np.random.dirichlet([alpha] * K)
        # Hill-Chao from expected proportions (large n limit)
        entropy = -np.sum(p * np.log(p + 1e-12))
        qD1 = np.exp(entropy)
        qD2 = 1.0 / np.sum(p ** 2)
        s1 = (qD1 - 1) / (K - 1)
        s2 = (qD2 - 1) / (K - 1)
        results.append((s1 + s2) / 2)
    return float(np.mean(results))


# ── Solve for alpha given observed s_D ─────────────────────────────────────
def solve_alpha(s_D_obs, K, tol=1e-4):
    """Binary search for alpha such that E[s_D | Dirichlet(alpha, K)] = s_D_obs."""
    if s_D_obs <= 0.01:
        return 0.05
    if K <= 1:
        return 1.0

    lo, hi = 0.01, 100.0

    # Check if s_D_obs is achievable
    sd_hi = expected_sd_dirichlet(hi, K)
    if s_D_obs > sd_hi:
        return hi

    for _ in range(30):
        mid = (lo + hi) / 2.0
        sd_mid = expected_sd_dirichlet(mid, K)
        if abs(sd_mid - s_D_obs) < tol:
            return mid
        if sd_mid < s_D_obs:
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2.0


# ── Estimate effective K from n ─────────────────────────────────────────────
def estimate_K(n):
    """Estimate number of clusters using same heuristic as the pipeline."""
    min_cluster_size = max(10, n // 100)
    # HDBSCAN with min_cluster_size produces roughly n/min_cluster_size clusters
    # Use a conservative lower bound (HDBSCAN tends to find fewer than naive estimate)
    K_raw = max(5, n // min_cluster_size)
    # Empirically HDBSCAN finds about 30-60% of naive estimate due to outliers
    K = max(5, int(K_raw * 0.45))
    return K


# ── Bootstrap CIs for a single cohort dataframe ────────────────────────────
def bootstrap_ci(df_m, n_bootstrap=N_BOOTSTRAP):
    """
    Add s_D_lo, s_D_hi, HIQ_lo, HIQ_hi columns to df_m via parametric bootstrap.

    For each month:
    1. Estimate K from n
    2. Solve for Dirichlet alpha matching observed s_D
    3. Draw n_bootstrap samples of cluster proportions from Dirichlet
    4. For each: draw multinomial counts from n sessions, compute s_D and HIQ
    """
    s_D_lo = []
    s_D_hi = []
    HIQ_lo = []
    HIQ_hi = []

    for _, row in df_m.iterrows():
        n = int(row['n'])
        s_D_obs = float(row['s_D'])
        s_C = float(row['s_C'])
        s_A = float(row['s_A']) if pd.notna(row.get('s_A')) else 0.5

        K = estimate_K(n)
        alpha = solve_alpha(s_D_obs, K)

        boot_sD = []
        boot_HIQ = []

        for _ in range(n_bootstrap):
            # Draw cluster proportions from Dirichlet
            p = np.random.dirichlet([alpha] * K)
            # Draw actual session counts
            counts = np.random.multinomial(n, p)
            counts = counts[counts > 0]  # remove empty clusters
            sd_boot = hill_chao_from_counts(counts)
            hiq_boot = hiq_composite(sd_boot, s_C, s_A)
            boot_sD.append(sd_boot)
            boot_HIQ.append(hiq_boot)

        boot_sD = np.array(boot_sD)
        boot_HIQ = np.array(boot_HIQ)

        s_D_lo.append(float(np.percentile(boot_sD, 2.5)))
        s_D_hi.append(float(np.percentile(boot_sD, 97.5)))
        HIQ_lo.append(float(np.percentile(boot_HIQ, 2.5)))
        HIQ_hi.append(float(np.percentile(boot_HIQ, 97.5)))

    df_m = df_m.copy()
    df_m['s_D_lo'] = s_D_lo
    df_m['s_D_hi'] = s_D_hi
    df_m['HIQ_lo'] = HIQ_lo
    df_m['HIQ_hi'] = HIQ_hi
    return df_m


# ── Figure with CI bands ────────────────────────────────────────────────────
def generate_ci_figure(results_dict, title, filename):
    """4-panel figure with shaded CI bands on s_D, s_C (no CI), s_A (no CI), HIQ."""
    colours = [COLOURS.get(name, '#2980b9') for name in results_dict.keys()]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    metrics = [
        ('s_D', 's_D_lo', 's_D_hi', 'Diversity (s_D)'),
        ('s_C', None, None, 'Connectivity (s_C)'),
        ('s_A', None, None, 'Agility (s_A)'),
        ('HIQ', 'HIQ_lo', 'HIQ_hi', 'HIQ Composite'),
    ]

    for ax_i, (col, lo_col, hi_col, label) in enumerate(metrics):
        ax = axes[ax_i]
        for ci, (cohort, df_m) in enumerate(results_dict.items()):
            if col not in df_m.columns or df_m[col].isna().all():
                continue
            color = colours[ci]
            months = df_m['month'].tolist()
            values = df_m[col].values
            x = range(len(months))

            # CI shading
            if lo_col and hi_col and lo_col in df_m.columns:
                lo = df_m[lo_col].values
                hi = df_m[hi_col].values
                ax.fill_between(x, lo, hi, alpha=0.18, color=color)

            ax.plot(x, values, marker='o', color=color,
                    linewidth=2, markersize=6, label=cohort)

            # Trend line
            valid_mask = ~np.isnan(values)
            if valid_mask.sum() >= 3:
                xv = np.arange(valid_mask.sum())
                yv = values[valid_mask]
                z = np.polyfit(xv, yv, 1)
                p = np.poly1d(z)
                x_trend = [i for i, m in enumerate(valid_mask) if m]
                ax.plot(x_trend, p(xv), linestyle='--', color=color, alpha=0.4, linewidth=1)

        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_ylabel('Score', fontsize=9)
        ax.set_xticks(list(range(len(list(results_dict.values())[0]['month']))))
        ax.set_xticklabels(list(results_dict.values())[0]['month'].tolist(),
                           rotation=30, ha='right')
        ax.set_ylim(0, max(0.7, ax.get_ylim()[1]))
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

        if lo_col:
            ax.set_title(f'{label} (shaded = 95% bootstrap CI)', fontsize=10, fontweight='bold')

    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = OUT / filename
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return str(path)


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    print("HIQ Bootstrap CI Extension")
    print("=" * 50)

    # Load all CSVs
    csvs = {
        'All Users':            'hiq_all_users.csv',
        'Global North':         'hiq_global_north.csv',
        'Global South':         'hiq_global_south.csv',
        'LLM-Represented':      'hiq_llm_represented.csv',
        'LLM-Underrepresented': 'hiq_llm_underrepresented.csv',
    }

    dfs = {}
    for name, fname in csvs.items():
        path = OUT / fname
        if path.exists():
            dfs[name] = pd.read_csv(path)
            print(f"  Loaded {name}: {len(dfs[name])} months")
        else:
            print(f"  WARNING: {path} not found — skipping {name}")

    # Bootstrap CIs
    print(f"\nBootstrapping CIs ({N_BOOTSTRAP} iterations per month-cohort)...")
    dfs_ci = {}
    ci_summary = {}

    for name, df_m in dfs.items():
        print(f"\n  {name}:")
        df_ci = bootstrap_ci(df_m)
        dfs_ci[name] = df_ci

        # Summary stats
        ci_widths_sD = df_ci['s_D_hi'] - df_ci['s_D_lo']
        ci_widths_HIQ = df_ci['HIQ_hi'] - df_ci['HIQ_lo']
        ci_summary[name] = {
            'months': df_ci['month'].tolist(),
            'n_sessions': df_ci['n'].tolist(),
            's_D': df_ci['s_D'].tolist(),
            's_D_lo': df_ci['s_D_lo'].tolist(),
            's_D_hi': df_ci['s_D_hi'].tolist(),
            'HIQ': df_ci['HIQ'].tolist() if 'HIQ' in df_ci.columns else None,
            'HIQ_lo': df_ci['HIQ_lo'].tolist(),
            'HIQ_hi': df_ci['HIQ_hi'].tolist(),
            'mean_CI_width_sD': float(ci_widths_sD.mean()),
            'max_CI_width_sD': float(ci_widths_sD.max()),
            'mean_CI_width_HIQ': float(ci_widths_HIQ.mean()),
        }

        for _, row in df_ci.iterrows():
            print(f"    {row['month']}: s_D={row['s_D']:.4f} "
                  f"[{row['s_D_lo']:.4f}, {row['s_D_hi']:.4f}]  "
                  f"HIQ=[{row['HIQ_lo']:.4f}, {row['HIQ_hi']:.4f}]  "
                  f"n={int(row['n'])}")

    # Save CI summary
    with open(OUT / 'hiq_bootstrap_ci_summary.json', 'w') as f:
        json.dump(ci_summary, f, indent=2)
    print(f"\n  CI summary saved: {OUT / 'hiq_bootstrap_ci_summary.json'}")

    # Generate updated figures
    print("\nGenerating figures with CI bands...")

    # Figure 1: All users
    if 'All Users' in dfs_ci:
        generate_ci_figure(
            {'All Users': dfs_ci['All Users']},
            'HIQ Monthly Time Series — All Users (95% Bootstrap CI)\n'
            '(WildChat-1M, April–November 2023)',
            'hiq_fig1_all_users_ci.jpg'
        )

    # Figure 2: GN vs GS
    geo = {k: dfs_ci[k] for k in ['Global North', 'Global South'] if k in dfs_ci}
    if len(geo) == 2:
        generate_ci_figure(
            geo,
            'HIQ by Geographic Cohort — Global North vs Global South (95% Bootstrap CI)\n'
            '(WildChat-1M, April–November 2023)',
            'hiq_fig2_geo_cohorts_ci.jpg'
        )

    # Figure 3: Language cohorts
    lang = {k: dfs_ci[k] for k in ['LLM-Represented', 'LLM-Underrepresented'] if k in dfs_ci}
    if len(lang) == 2:
        generate_ci_figure(
            lang,
            'HIQ by Language Cohort — LLM-Represented vs LLM-Underrepresented (95% Bootstrap CI)\n'
            '(WildChat-1M, April–November 2023)',
            'hiq_fig3_lang_cohorts_ci.jpg'
        )

    # Print paper-ready summary
    print("\n" + "=" * 60)
    print("CI SUMMARY FOR PAPER")
    print("=" * 60)
    for name, s in ci_summary.items():
        print(f"\n{name}:")
        print(f"  Mean CI width (s_D):  {s['mean_CI_width_sD']:.4f}")
        print(f"  Max CI width (s_D):   {s['max_CI_width_sD']:.4f}")
        print(f"  Mean CI width (HIQ):  {s['mean_CI_width_HIQ']:.4f}")

        # Identify sparse months (Apr, May for GS)
        for i, month in enumerate(s['months']):
            if month in ['2023-04', '2023-05']:
                w_sD = s['s_D_hi'][i] - s['s_D_lo'][i]
                print(f"  {month} (sparse): CI width s_D = {w_sD:.4f}  n = {s['n_sessions'][i]}")

    print("\nDone.")


if __name__ == '__main__':
    main()
