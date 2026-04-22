# The Wrong Kind of Momentum: Why Break Points Kill It and Tiebreaks Create It

Live blog: https://flynnpresence.github.io/BEE2041-tennis-momentum/blog.html

**Is Momentum a Statistical Illusion in Professional Tennis?**

## Overview

This project tests whether point-by-point momentum exists in professional tennis using all four 2023 Grand Slams across both ATP and WTA tours. It applies logistic regression and a Causal Forest (econml) to estimate whether winning a high-leverage point causally increases the probability of winning the next.

---

## Repository Structure# BEE2041-tennis-momentum
BEE2041 Data Science in Economics — Tennis Momentum Project
---

## Replication

### Option A — Full pipeline (downloads raw data)

```bash
pip install -r requirements.txt
make all
```

This runs download → clean → features → model in sequence.

### Option B — From checkpoint (skips download and cleaning)

The processed feature checkpoint is included in the repository. To run the model directly:

```bash
pip install -r requirements.txt
/usr/local/bin/python3 scripts/model.py
```

---

## Dependencies

See `requirements.txt`. Key packages:

- pandas, numpy, scipy
- statsmodels
- scikit-learn
- econml (Causal Forest)
- matplotlib

---

## Data Sources

- **Jeff Sackmann Match Charting Project** — point-by-point data
  https://github.com/JeffSackmann/tennis_MatchChartingProject
- **Jeff Sackmann ATP/WTA results** — official rankings
  https://github.com/JeffSackmann/tennis_atp
  https://github.com/JeffSackmann/tennis_wta

---

## Outputs

| # | File | Description |
|---|------|-------------|
| 1 | output1_chi2_table.png | Per-player Chi-squared and runs test results |
| 2 | output2_cusum_wta.png | CUSUM momentum tracker, Alexandrova vs Brengle |
| 3 | output3_tboe_scatter.png | Tiebreak over-expectation by player |
| 4 | output4_cate_plot.png | Heterogeneous causal effects across player rankings |
| 5 | output5_model_table.png | Logistic regression and Causal Forest ATE estimates |
| 6 | blog.js (D3) | Interactive reveal chart — BP vs tiebreak ATE split |
| 7 | output6_feature_importance.png | Causal Forest feature importance |
| — | ate_results.csv | Python-generated ATE values backing the D3 chart |

---

*Module: BEE2041 Data Science in Economics — University of Exeter*