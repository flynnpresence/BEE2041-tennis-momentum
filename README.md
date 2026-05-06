# The Wrong Kind of Momentum

Live blog: https://flynnpresence.github.io/BEE2041-tennis-momentum/blog.html

BEE2041 Data Science in Economics, University of Exeter

---

## Overview

This project tests whether point-by-point momentum exists in professional tennis using all four 2023 Grand Slams across both ATP and WTA tours. It applies logistic regression and a Causal Forest (econml) to estimate whether winning a high-leverage point causally increases the probability of winning the next point.

**Finding:** Break points generate a hangover effect (ATP: -0.0601, WTA: -0.0291). Tiebreaks generate genuine positive momentum (ATP: +0.0564, WTA: +0.0209). The two effects partially offset in aggregate, explaining why prior studies reached conflicting conclusions.

---

## Methodological Rigor

To address conflicting conclusions in existing literature (Gilovich et al., 1985 vs. Miller and Sanjurjo, 2018), this project implements several safeguards:

- **Random Focal Player Mask:** Each match is randomly viewed from the perspective of one player to prevent perfectly correlated duplicate observations and satisfy independence assumptions.
- **Forward-Rolling Priors:** Luck proxies use neutral tour-wide Bayesian priors (0.38 for return points; 0.5 for tiebreaks) to avoid cold-start bias at match beginnings.
- **Deconfounding Baseline Skill:** Consistent with Kovalchik (2016), official rankings are attached via a two-step validated merge (tournament-specific then season-median fallback) to isolate momentum from player quality.
- **Clustered Standard Errors:** Logistic models use match-level clustering to account for intra-match point dependency.
- **Causal Forest (Double Machine Learning, DML):** Employs econml to estimate the Average Treatment Effect while controlling for player ranking, rolling win percentage, CUSUM (cumulative momentum score), and winning streak length, isolating the "success breeds success" mechanic.

---

## Replication

```bash
pip install -r requirements.txt
make all
```

This runs the full pipeline in sequence: download → clean → features → model → build_blog_data → render.

Requires Python 3.10+ (tested on 3.13.0), pip, and [Quarto](https://quarto.org) 1.9+ (tested on 1.9.37) installed on your system. Note: pandas 3.0.1 requires Python ≥ 3.10. The download step requires an internet connection to fetch raw data from GitHub.

Run `make reset` to wipe all generated files and start fresh.

Note: Raw data is read from `data/raw/` and never modified. All cleaned and feature-engineered data is written to `data/processed/`, following the principle that raw data is sacred.

Note: The Makefile uses `outputs/ate_results.csv` as the sentinel for the model step. If individual chart files are deleted while this CSV exists, run `make reset && make all` to force a full rebuild.

---

## Directory Structure

```
BEE2041-tennis-momentum/
├── data/
│   ├── raw/            # Raw data: never modified
│   └── processed/      # Cleaned and feature-engineered data
├── scripts/
│   ├── download.py     # Downloads raw data from Jeff Sackmann's GitHub
│   ├── clean.py        # Filters, merges, and validates data
│   ├── features.py     # Engineers momentum and control features
│   ├── model.py        # Logistic regression and Causal Forest
│   └── build_blog_data.py  # Generates blog_data.js for D3 chart
├── outputs/            # HTML charts and CSV results from model.py
├── blog.qmd            # Quarto blog source
├── blog.html           # Rendered blog (auto-generated)
├── blog.js             # D3 reveal chart and interactive elements
├── blog_data.js        # Python-generated data constants (auto-generated)
├── styles.css          # Blog styling
├── Makefile            # Pipeline orchestration
├── README.md           # Project documentation
└── requirements.txt    # Python dependencies
```

---

## Dependencies

See `requirements.txt`. Key packages:

- pandas, numpy, scipy
- statsmodels
- scikit-learn
- econml (Causal Forest)
- matplotlib, plotly
- requests

---

## Data Sources

- **Jeff Sackmann Match Charting Project**: point-by-point data
  https://github.com/JeffSackmann/tennis_MatchChartingProject
- **Jeff Sackmann ATP/WTA results**: official rankings
  https://github.com/JeffSackmann/tennis_atp
  https://github.com/JeffSackmann/tennis_wta

Each points file contains one row per point with columns for score state (Pts), server (Svr), set and game counters, and point outcome. Each results file contains one row per match with player names, tournament, round, surface, and ranking data.

The Match Charting Project is licensed under CC BY 4.0. The ATP and WTA results repositories are licensed under CC BY-NC-SA 4.0 (non-commercial use only). Both licences permit academic use with attribution.

---

## Outputs

| # | File | Description |
|---|------|-------------|
| 1 | output1_chi2_table.html | Per-player Chi-squared and runs test results |
| 2 | output2_cusum_wta.html | CUSUM momentum tracker, Alexandrova vs Brengle |
| 3 | output3_tboe_scatter.html | Tiebreak over-expectation by player |
| 4 | output4_cate_plot.html | Heterogeneous causal effects across player rankings |
| 5 | output5_model_table.html | Logistic regression and Causal Forest ATE estimates |
| 6 | output6_reveal_chart.html | Standalone Plotly reveal chart (the interactive version embedded in the blog is rendered by D3 via blog.js) |
| 7 | output7_feature_importance.html | Causal Forest feature importance, ATP and WTA |
| - | ate_results.csv | Python-generated ATE values |
| - | feature_importance.csv | Python-generated feature importance values |

---

## Key References

- Du, C., Zhang, C. and Zhou, L. (2025). A novel methodological framework for analyzing the momentum effect in tennis singles. [online] Available at: https://arxiv.org/abs/2509.01243 [Accessed 20 Apr. 2026].
- Gilovich, T., Vallone, R. and Tversky, A. (1985). The hot hand in basketball: On the misperception of random sequences. *Cognitive Psychology*, [online] 17(3), pp.295–314. doi:https://doi.org/10.1016/0010-0285(85)90010-6.
- Kovalchik, S.A. (2016). Searching for the GOAT of tennis win prediction. *Journal of Quantitative Analysis in Sports*, [online] 12(3). doi:https://doi.org/10.1515/jqas-2015-0059.
- Miller, J.B. and Sanjurjo, A. (2018). Surprised by the Hot Hand Fallacy? A Truth in the Law of Small Numbers. *OSF Preprints*. doi:https://doi.org/10.31219/osf.io/sv9x2.
- Sackmann, J. (2023). The Match Charting Project. [online] GitHub. Available at: https://github.com/JeffSackmann/tennis_MatchChartingProject [Accessed 20 Apr. 2026].

---

*Module: BEE2041 Data Science in Economics, University of Exeter*
