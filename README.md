# The Wrong Kind of Momentum

> **Live blog:** https://flynnpresence.github.io/BEE2041-tennis-momentum/blog.html

BEE2041 Data Science in Economics, University of Exeter

---

## Overview

This project tests whether point-by-point momentum exists in professional tennis using volunteer-charted matches across all four 2023 Grand Slams (169 ATP matches, ~33% of men's draws; 128 WTA matches, ~25% of women's draws). It applies logistic regression and a Causal Forest (econml) to estimate the effect of winning a high-leverage point on the probability of winning the next point.

**Finding:** Break points produce a large positive effect on the next point in both tours (ATP: +0.1470, WTA: +0.0863), driven by the serving transition that follows a break point win rather than psychological momentum. Tiebreaks are null in both tours (ATP: +0.0267, 95% CI [-0.018, 0.058]; WTA: -0.0253, 95% CI [-0.092, 0.051], based on a smaller 209 treated observations). The combined ATEs (ATP: +0.1042, WTA: +0.0640) are dominated by the more frequent break point effect, explaining why pooled analyses reach conflicting conclusions.

---

## Methodological Rigour

To address conflicting conclusions in existing literature (Gilovich et al., 1985 vs. Miller and Sanjurjo, 2018), this project implements several safeguards:

- **Random Focal Player Mask:** Each match is randomly viewed from the perspective of one player to prevent perfectly correlated duplicate observations and satisfy independence assumptions.
- **Forward-Rolling Priors:** Luck proxies use a neutral Bayesian prior (0.5 for tiebreaks) to avoid cold-start bias at match beginnings.
- **Deconfounding Baseline Skill:** Official rankings are attached via a two-step validated merge (tournament-specific then season-median fallback) to isolate momentum from player quality. Elo was considered but rejected: the two-stage fallback process required to achieve full coverage across all 56,253 points is not readily supported by available Elo data.
- **Retirement Matches:** Retirement matches could not be filtered at point level because the Match Charting Project data contains no match outcome flag; their points remain in the dataset. Rankings from retirement matches are excluded from the ranking lookup via the W/O|RET filter applied in both ranking steps. These matches are a small fraction of the total and affect both tours equally, so they do not materially skew the findings.
- **Clustered Standard Errors:** Logistic models use match-level clustering (`cov_type='cluster'`) to account for intra-match point dependency. Causal forest and LinearDML standard errors come from a match-clustered bootstrap (resampling whole matches, not points, since points within a match are not independent observations) rather than the dispersion of per-unit CATE predictions; see `scripts/bootstrap_ate.py`. Primary fits also use grouped cross-fitting (`groups=match_id`) so DML cross-fitting keeps each match within a single fold.
- **Estimator Selection:** CausalForestDML is used where treatment is well-powered (the combined and break-point specs). LinearDML is used for the two tiebreak cells (sparse treatment: 729 ATP / 209 WTA) and the four rank-only robustness checks (single-control propensity fragility): a cv-fold sweep showed the forest's tiebreak point estimate flipping sign under grouped cross-fitting, while LinearDML reproduces the forest's own outlier-trimmed median to within ~5% with a stable bootstrap distribution in every case, confirming the instability was estimator noise rather than a genuine disagreement.
- **Stratified Subsampling:** The causal forest subsamples to 15,000 observations per run, preserving all treated observations and randomly sampling control observations to reach 15,000 total; the ATP is subsampled from 38,488 points and the WTA from 17,765 points.
- **VIF Check:** A Variance Inflation Factor check confirmed no problematic multicollinearity among control variables (maximum score 1.22; VIF above 10 indicates problematic overlap).
- **Causal Forest (Double Machine Learning, DML):** Employs econml to estimate the Average Treatment Effect while controlling for player ranking, rolling win percentage, CUSUM (cumulative momentum score), and winning streak length.

---

## Replication

```bash
pip install -r requirements.txt
make all
```

This runs the full pipeline in sequence: download → clean → features → model → build_blog_data → render.

Requires Python 3.10+ (tested on 3.12.7), pip, and [Quarto](https://quarto.org) 1.9+ (tested on 1.9.37) installed on your system. Note: Quarto is an external binary not managed by pip and must be installed separately. Note: pandas 3.0.1 requires Python ≥ 3.9. The download step requires an internet connection to fetch raw data from GitHub.

Run `make pipeline` to run the data pipeline without rendering the blog (no Quarto required). Run `make reset` to wipe all generated files and start fresh.

Note: Raw data is read from `data/raw/` and never modified. All cleaned and feature-engineered data is written to `data/processed/`, following the principle that raw data is sacred.

Note: The Makefile uses `outputs/ate_results.csv` as the sentinel for the model step. If individual chart files are deleted while this CSV exists, run `make reset && make all` to force a full rebuild.

---

## Directory Structure

```
BEE2041-tennis-momentum/
├── data/
│   ├── raw/                # Raw data: never modified
│   └── processed/          # Cleaned and feature-engineered data
├── outputs/                # HTML charts and CSV results from model.py
├── scripts/
│   ├── build_blog_data.py  # Generates blog_data.js for D3 chart
│   ├── clean.py            # Filters, merges, and validates data
│   ├── download.py         # Downloads raw data from Jeff Sackmann's GitHub
│   ├── features.py         # Engineers momentum and control features
│   └── model.py            # Logistic regression and Causal Forest
├── .gitignore              # Git ignore patterns (excludes raw/processed data, build artefacts)
├── Makefile                # Pipeline orchestration
├── README.md               # Project documentation
├── blog.html               # Rendered blog (auto-generated)
├── blog.js                 # D3 reveal chart and interactive elements
├── blog.qmd                # Quarto blog source
├── blog_data.js            # Python-generated data constants (auto-generated)
├── requirements.txt        # Python dependencies
└── styles.css              # Blog styling
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

## Data Dictionary

All processed data files are emitted to `data/processed/`. Features are engineered strictly forward-rolling within match groups: no point's value depends on points that come after it.

### processed_features.csv

Model checkpoint. One row per point, focal-player perspective. Last point of each match excluded (no successor for `Next_Point_Won`). This is the file consumed by `model.py`.

| Column | Type | Description |
|--------|------|-------------|
| `match_id` | str | Match identifier from the Match Charting Project. Format: `YYYYMMDD-{M\|W}-Tournament-Round-Player1-Player2`. |
| `Pt` | int64 | Point number within the match (1-indexed, sequential). |
| `Tournament` | str | Grand Slam: Australian Open, Roland Garros, Wimbledon, or US Open. |
| `Round` | str | Round code (R128, R64, R32, R16, QF, SF, F). Qualifying rounds excluded upstream. |
| `Surface` | str | Court surface (Hard, Clay, Grass). |
| `Focal_Player` | str | Name of the randomly assigned focal player for this match. |
| `Opponent_Player` | str | Focal player's opponent. |
| `Focal_Ranking` | int64 | Official ATP/WTA ranking of focal player. Primary source: tournament-specific ranking from the GS results table. Fallback: season-median ranking. |
| `Opponent_Ranking` | int64 | Same convention as `Focal_Ranking`, applied to opponent. |
| `Ranking_Diff` | int64 | `Focal_Ranking - Opponent_Ranking`. Positive = focal is the lower-ranked (worse) player. |
| `Svr` | int64 | Server on this point. 1 or 2, refers to Player_1 / Player_2 in the original match record, not focal / opponent. |
| `TbSet` | bool | Set-format flag (True for all 2023 GS rows). Does not identify individual tiebreak points; use `High_Leverage_TB` for that. |
| `High_Leverage` | int64 | 1 if break point or tiebreak point; 0 otherwise. Treatment indicator for the causal forest. |
| `High_Leverage_BP` | int64 | 1 if break point (`Pts` in {0-40, 15-40, 30-40, 40-AD}); 0 otherwise. |
| `High_Leverage_TB` | int64 | 1 if point played within a tiebreak (`Gm1 == 6 & Gm2 == 6`); 0 otherwise. |
| `Streak_k4` | int64 | 1 if focal player won all of the previous 4 points in the match; 0 otherwise. Uses only past points. 0 by construction for the first 4 points of every match. |
| `Rolling_Win_Pct` | float64 | Focal player's win rate over up to the previous 10 points in the match (current point excluded). Window grows from 1 point at the start of the match to a maximum of 10. Range [0, 1]. Initialised at 0.5 (neutral prior) for the first point of each match. |
| `CUSUM` | float64 | Cumulative deviation of focal player from their expanding within-match win rate. Positive = focal is performing above their match-to-date average. Computed using shifted (past-only) values; CUSUM = 0 at the first point of each match by construction. |
| `TBOE` | float64 | Tiebreak Over-Expectation. Within-match historical tiebreak win rate minus current `Rolling_Win_Pct`. Positive = focal overperforms in tiebreaks relative to general form. The tiebreak-rate component is initialised at 0.5 before any tiebreak points have been played; combined with `Rolling_Win_Pct` also at 0.5, TBOE starts at 0. |
| `Point_Won` | int64 | 1 if focal player won the current point; 0 otherwise. |
| `Next_Point_Won` | int64 | 1 if focal player wins the immediately following point; 0 otherwise. Outcome variable for all causal models. |
| `Tour` | str | `'ATP'` or `'WTA'`. |

### atp_cleaned_points.csv / wta_cleaned_points.csv

Per-tour intermediate output of `clean.py`. One row per point. Contains all engineered columns plus passthrough fields from the Match Charting Project source files.

**Structural and score-state columns** (from `charting-{m,w}-points-2020s.csv`):

| Column | Type | Description |
|--------|------|-------------|
| `match_id` | str | Match identifier (see format above). |
| `Pt` | int64 | Point number within the match (1-indexed). |
| `Set1`, `Set2` | int64 | Sets won by Player_1 / Player_2 so far in the match. |
| `Gm1`, `Gm2` | int64 | Games won by Player_1 / Player_2 so far in the current set. |
| `Gm#` | int64 | Game number within the current set, beginning at 1. (See Sackmann's `data_dictionary.txt`.) |
| `Pts` | str | Score at the start of the point in server-receiver format (e.g. `30-40`). |
| `TbSet` | bool | True if the current set is being played to a tiebreak format (True for all 2023 GS rows). |
| `Svr` | int64 | Server on this point: 1 (Player_1) or 2 (Player_2). |
| `PtWinner` | int64 | Winner of the point: 1 (Player_1) or 2 (Player_2). |
| `1st` | str | Charting code for the outcome of the first serve and rally. User-submitted free-text per Sackmann's notation. Empty for points where no first serve was charted. |
| `2nd` | str | Charting code for the second serve and rally, where applicable. Empty if no second serve. |
| `Notes` | str | Charter's free-text notes (e.g. challenge outcomes). Mostly empty. |

**Match metadata columns** (from `charting-{m,w}-matches.csv`):

| Column | Type | Description |
|--------|------|-------------|
| `Player_1`, `Player_2` | str | Original player labels from the matches file (Player_1 always served first). Independent of focal / opponent assignment. |
| `Pl_1_hand`, `Pl_2_hand` | str | Handedness: `'R'` (right) or `'L'` (left). |
| `Date` | str | Match date in `YYYYMMDD` format. |
| `Tournament` | str | Grand Slam name with underscores converted to spaces. |
| `Round` | str | Round code. |
| `Time` | str | Match start time where recorded; otherwise empty. |
| `Court` | str | Court name where recorded (e.g. `'Centre Court'`, `'Court Philippe Chatrier'`); otherwise empty. |
| `Surface` | str | Hard, Clay, or Grass. |
| `Umpire` | str | Chair umpire name where recorded; otherwise empty. |
| `Best_of` | int64 | Match format: 5 (ATP Grand Slam) or 3 (WTA Grand Slam). |
| `Final_TB?` | str | Final-set tiebreak format flag from the source file (e.g. `'1'`, `'A'`, `'N'`). Encoding follows Sackmann's matches schema. |
| `Charted_by` | str | Volunteer charter handle. |

**Engineered columns** (added by `clean.py`):

| Column | Type | Description |
|--------|------|-------------|
| `High_Leverage` | int64 | 1 if break point or tiebreak point; 0 otherwise. |
| `High_Leverage_BP` | int64 | 1 if break point (`Pts` in {0-40, 15-40, 30-40, 40-AD}); 0 otherwise. |
| `High_Leverage_TB` | int64 | 1 if point played within a tiebreak (`Gm1 == 6 & Gm2 == 6`); 0 otherwise. |
| `focal_is_p1` | bool | True if focal player is Player_1, False if Player_2. Random per-match assignment, seeded with `np.random.seed(42)`. |
| `Focal_Player` | str | Name of focal player (= Player_1 if `focal_is_p1`, else Player_2). |
| `Opponent_Player` | str | Name of opponent (= Player_2 if `focal_is_p1`, else Player_1). |
| `Point_Won` | int64 | 1 if focal player won the current point; 0 otherwise. |
| `Focal_Is_Server` | int64 | 1 if focal player is serving on this point; 0 otherwise. |
| `Next_Point_Won` | int64 | 1 if focal player wins the next point; 0 otherwise. Last point of each match dropped (no successor). |
| `Tournament_Key` | str | Normalised tournament name used as a merge key for ranking attachment. Identical to `Tournament` after `Us Open` to `US Open` substitution. |
| `Focal_Ranking` | int64 | Official ranking of focal player. Primary: GS-specific lookup. Fallback: season-median ranking. |
| `Opponent_Ranking` | int64 | Same convention applied to opponent. |
| `Ranking_Diff` | int64 | `Focal_Ranking - Opponent_Ranking`. Positive = focal is lower-ranked. |

For full Sackmann-source column semantics (notation codes inside `1st`, `2nd`, etc.), see `tennis_MatchChartingProject/data_dictionary.txt`.

### per_player_tests.csv

Output of the per-player Chi-squared and Wald-Wolfowitz runs tests run in `features.py`. Used to populate Output 1 (Chi-squared summary table) in the blog. One row per player; players with fewer than 20 focal points excluded.

| Column | Type | Description |
|--------|------|-------------|
| `Tour` | str | `'ATP'` or `'WTA'`. |
| `Player` | str | Player name. |
| `N_Points` | int64 | Number of points contributed by this player as focal. Minimum 20 for inclusion. |
| `Win_Rate` | float64 | Proportion of focal points won. Range [0, 1]. |
| `Chi2_Stat` | float64 | Pearson chi-squared statistic from 2x2 contingency table of `Point_Won` x `Next_Point_Won`. NaN if not computable. |
| `Chi2_P` | float64 | Uncorrected two-sided p-value for the chi-squared independence test. |
| `Chi2_Sig` | int64 | 1 if `Chi2_P < 0.05` (uncorrected); 0 otherwise. |
| `Runs_Z` | float64 | Standardised z-statistic from the Wald-Wolfowitz runs test on the sequence of `Point_Won`. |
| `Runs_P` | float64 | Uncorrected two-sided p-value for the runs test. |
| `Runs_Sig` | int64 | 1 if `Runs_P < 0.05` (uncorrected); 0 otherwise. |
| `Chi2_P_BH` | float64 | Benjamini-Hochberg FDR-corrected p-value for the chi-squared test. |
| `Chi2_Sig_BH` | int64 | 1 if `Chi2_P_BH` crosses the BH threshold (FDR = 0.05); 0 otherwise. |
| `Runs_P_BH` | float64 | BH-corrected p-value for the runs test. |
| `Runs_Sig_BH` | int64 | 1 if `Runs_P_BH` crosses the BH threshold; 0 otherwise. |

**Notes**

- **Forward-rolling principle.** All within-match features (`Streak_k4`, `Rolling_Win_Pct`, `CUSUM`, `TBOE`) are computed using `shift(1)` before any rolling or expanding window, guaranteeing that no feature value depends on the point it predicts. This is enforced in `features.py::engineer_features`.
- **Neutral priors.** `Rolling_Win_Pct` and the tiebreak-win-rate component of `TBOE` are initialised at 0.5 to avoid cold-start bias at match beginnings.
- **Focal-player assignment.** One player per match is randomly designated as focal (`np.random.seed(42)`). This prevents the perfectly correlated duplicate observations that would arise from tracking both players, which would violate the independence assumption of the causal forest.
- **Constants** (set in `features.py`): `STREAK_K = 4`, `ROLLING_WIN = 10`, `ALPHA = 0.05`.

---

## Outputs

| # | File | Description |
|---|------|-------------|
| 1 | output1_chi2_table.html | Summary of per-player Chi-squared and runs test results by tour |
| 2 | output2_cusum_wta.html | CUSUM momentum tracker, Alexandrova vs Brengle |
| 3 | output3_tboe_scatter.html | Tiebreak over-expectation by player |
| 4 | output4_cate_plot.html | Heterogeneous causal effects across player rankings |
| 5 | output5_model_table.html | Logistic regression and Causal Forest ATE estimates |
| 6 | output6_reveal_chart.html | Standalone Plotly reveal chart (the interactive version embedded in the blog is rendered by D3 via blog.js) |
| 7 | output7_feature_importance.html | Causal Forest feature importance, ATP and WTA |
| - | ate_results.csv | Python-generated ATE values |
| - | ate_results_robustness.csv | Robustness check ATEs using ranking-only controls |
| - | feature_importance.csv | Python-generated feature importance values |

---

## Key References

- Du, C., Zhang, C. and Zhou, L. (2025). A novel methodological framework for analyzing the momentum effect in tennis singles. [online] Available at: https://arxiv.org/abs/2509.01243 [Accessed 20 Apr. 2026].
- Gilovich, T., Vallone, R. and Tversky, A. (1985). The hot hand in basketball: On the misperception of random sequences. *Cognitive Psychology*, [online] 17(3), pp.295–314. doi:https://doi.org/10.1016/0010-0285(85)90010-6.
- Miller, J.B. and Sanjurjo, A. (2018). Surprised by the Hot Hand Fallacy? A Truth in the Law of Small Numbers. *OSF Preprints*. doi:https://doi.org/10.31219/osf.io/sv9x2.
- Sackmann, J. (2023). The Match Charting Project. [online] GitHub. Available at: https://github.com/JeffSackmann/tennis_MatchChartingProject [Accessed 20 Apr. 2026].

---

*Module: BEE2041 Data Science in Economics, University of Exeter*
