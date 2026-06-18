# HIQ Framework — Operational Demonstration

This repository contains the pipeline code, results, and figures for the operational demonstration accompanying the paper:

**"Measuring Equitable AI Value Delivery Across Language Communities: A Specification for the Humane Intelligence Quotient (HIQ)"**

> Submitted for peer review — author details withheld for blind review.

---

## What This Repository Contains

| File | Description |
|------|-------------|
| `hiq_bootstrap_ci.py` | Full pipeline: ingests WildChat interaction data, computes D/F/A scores per cohort, generates bootstrap confidence intervals (B=1000), and outputs ranked model comparisons |
| `hiq_bootstrap_ci_summary.json` | Pre-computed results from the operational demonstration run on the WildChat corpus |
| `HI - All Code and Results - Cleaned.ipynb` | Jupyter notebook with full code and annotated results — the primary reproducibility artefact |
| `hiq_fig1_all_users_ci.jpg` | Figure 1: HIQ composite scores with 95% bootstrap CIs, all-users cohort |
| `hiq_fig2_geo_cohorts_ci.jpg` | Figure 2: HIQ scores disaggregated by geographic cohort |
| `hiq_fig3_lang_cohorts_ci.jpg` | Figure 3: HIQ scores disaggregated by language cohort |

---

## Purpose

This code demonstrates **operational feasibility**: that real-world interaction data can be ingested, processed through the HIQ D/F/A pipeline, and produce interpretable scores. It is not an empirical study and makes no causal claims about model quality.

The outputs illustrate the kind of signal that becomes visible when HIQ is applied to real data at scale. They are examples of what the framework makes visible, not findings about the models evaluated.

---

## Framework Overview

HIQ measures equitable AI value delivery across three structurally independent dimensions:

- **D — Diversity:** Breadth of language communities served
- **F — Fairness:** Consistency of value delivery across cohorts (MAD-based)
- **A — Agility:** Responsiveness signal derived from user behavioural indicators (return rate, non-return rate, session depth ratio)

Scores are aggregated via geometric mean, which penalises dimensional imbalance. HIQ scores are meaningful only comparatively — they rank models relative to each other, not against an absolute standard.

---

## Data Source

Operational demonstration uses the [WildChat dataset](https://huggingface.co/datasets/allenai/WildChat) (Zhao et al., 2024). Users consented to research use of their interactions at the point of collection, under the dataset's published terms.

---

## Reproducing the Results

```bash
pip install pandas numpy scipy
python hiq_bootstrap_ci.py
```

The script expects WildChat data in the format described in the notebook. Pre-computed results are in `hiq_bootstrap_ci_summary.json`.

---

## Citation

Citation details will be provided upon publication. In the interim, please cite the paper title and this repository URL.

---

## Contact

For questions about the framework specification, refer to the accompanying paper. For questions about this code, open an issue in this repository.
