"""
leverage_matched_test.py
-------------------------
The load-bearing robustness test that resolves the leverage mismatch
apply_morris_importance.py surfaces (methodology.md Β§7a.3): restricts each
tour's won-BP and won-SGP populations to a common-support Morris-importance
band, then re-estimates both ATEs within it via the same match-clustered
bootstrap (CausalForestDML, B=199) used everywhere else in this project.

Answers: does the BP/SGP sign flip survive when the two groups are matched
on formal point importance, not just the crude score-state/game-margin/
set-number proxy? If yes, the leverage gap the Morris measure found is not
what's producing the sign difference, closing the discouragement rival
properly rather than by disclaiming a known confound.

Requires data/processed/{atp,wta}_morris_importance.csv (run
apply_morris_importance.py first).
"""

import os
import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from sklearn.ensemble import GradientBoostingRegressor
from joblib import Parallel, delayed

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, 'data', 'processed')
OUT_DIR = os.path.join(BASE_DIR, 'diagnostics', 'leverage_matched_test')
os.makedirs(OUT_DIR, exist_ok=True)

CONTROLS = ['Focal_Ranking', 'Rolling_Win_Pct', 'Streak_k4', 'CUSUM']
SEED = 42
SUBSAMPLE_CAP = 15000
B = 199
N_JOBS = 6

# Common-support band per tour: overlap of each group's [p5, p95] Morris
# importance range (see apply_morris_importance.py's output for the
# percentiles this is derived from). ATP's band is thinner than WTA's --
# reported honestly in methodology.md Β§7a.3 rather than smoothed over.
BANDS = {'ATP': (0.0180, 0.0455), 'WTA': (0.0235, 0.0913)}


def build_matched(proc: pd.DataFrame, morris: pd.DataFrame, tour: str,
                  flag: str, lo: float, hi: float) -> pd.DataFrame:
    """Full (un-subsampled) spec frame with Treatment restricted to the
    target flag's won-points falling inside [lo, hi] Morris importance.
    Control pool construction (exclude other treatments' won-points) is
    identical to bootstrap_ate.py's build_spec."""
    d = proc[proc['Tour'] == tour].copy().reset_index(drop=True)
    won_bp = (d['High_Leverage_BP'] == 1) & (d['Point_Won'] == 1)
    won_tb = (d['High_Leverage_TB'] == 1) & (d['Point_Won'] == 1)
    won_sgp = (d['High_Leverage_SGP'] == 1) & (d['Point_Won'] == 1)
    is_target = (d[flag] == 1) & (d['Point_Won'] == 1)

    # Morris_Importance is only defined for the won-BP/SGP subset (that's
    # all apply_morris_importance.py scores); align by position since
    # `morris` was built by iterating this exact same subset in this exact
    # row order.
    combined_flag = ((d['High_Leverage_BP'] == 1) | (d['High_Leverage_SGP'] == 1)) & (d['Point_Won'] == 1)
    imp_col = pd.Series(np.nan, index=d.index)
    imp_col.loc[combined_flag[combined_flag].index] = morris['Morris_Importance'].values
    d['Morris_Importance'] = imp_col

    in_band = is_target & (d['Morris_Importance'] >= lo) & (d['Morris_Importance'] <= hi)
    d['Treatment'] = in_band.astype(float)

    other_won = (won_tb | won_sgp) if flag == 'High_Leverage_BP' else (won_bp | won_tb)
    d = d[(d['Treatment'] == 1) | ~other_won]
    keep = CONTROLS + ['Next_Point_Won', 'match_id', 'Treatment']
    return d[keep].dropna()


def fit_ate(data: pd.DataFrame, subsample_seed: int, forest_seed: int = SEED,
           n_jobs: int = 1) -> float:
    d = data
    if len(d) > SUBSAMPLE_CAP:
        treated = d[d['Treatment'] == 1]
        n_control = min(len(d) - len(treated), max(0, SUBSAMPLE_CAP - len(treated)))
        control = d[d['Treatment'] == 0].sample(n_control, random_state=subsample_seed)
        d = pd.concat([treated, control])
    T = d['Treatment'].values
    Y = d['Next_Point_Won'].astype(float).values
    X = d[CONTROLS].astype(float).values
    groups = d['match_id'].values
    cf = CausalForestDML(
        model_y=GradientBoostingRegressor(n_estimators=200, random_state=forest_seed),
        model_t=GradientBoostingRegressor(n_estimators=200, random_state=forest_seed),
        n_estimators=200, cv=5, n_jobs=n_jobs, random_state=forest_seed, verbose=0)
    cf.fit(Y, T, X=X, groups=groups)
    return float(cf.ate(X))


def cluster_resample(data: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    matches = data['match_id'].unique()
    drawn = rng.choice(matches, size=len(matches), replace=True)
    parts = []
    for k, m in enumerate(drawn):
        rows = data[data['match_id'] == m].copy()
        rows['match_id'] = f'boot_{k}'
        parts.append(rows)
    return pd.concat(parts, ignore_index=True)


def bootstrap(proc, morris, tour, flag, lo, hi) -> dict:
    data = build_matched(proc, morris, tour, flag, lo, hi)
    n_treated = int(data['Treatment'].sum())
    n_matches = data['match_id'].nunique()
    ate_point = fit_ate(data, subsample_seed=SEED, n_jobs=1)

    def one_rep(b):
        rng = np.random.default_rng(SEED + 1 + b)
        boot = cluster_resample(data, rng)
        try:
            return fit_ate(boot, subsample_seed=SEED + 1 + b, n_jobs=1)
        except Exception:
            return np.nan

    reps = Parallel(n_jobs=N_JOBS)(delayed(one_rep)(b) for b in range(B))
    reps = np.array(reps, dtype=float)
    ok = reps[~np.isnan(reps)]
    return dict(
        tour=tour, flag=flag, n_matches=n_matches, n_treated=n_treated,
        ate_point=round(ate_point, 4), se=round(float(ok.std(ddof=1)), 4),
        ci_lo=round(float(np.percentile(ok, 2.5)), 4),
        ci_hi=round(float(np.percentile(ok, 97.5)), 4),
        n_fail=int(np.isnan(reps).sum()), B_effective=len(ok),
    )


def main() -> None:
    print('=== leverage_matched_test.py ===')
    proc = pd.read_csv(os.path.join(PROC_DIR, 'processed_features.csv'), low_memory=False)
    morris = {
        'ATP': pd.read_csv(os.path.join(PROC_DIR, 'atp_morris_importance.csv')),
        'WTA': pd.read_csv(os.path.join(PROC_DIR, 'wta_morris_importance.csv')),
    }

    results = []
    for tour in ['ATP', 'WTA']:
        lo, hi = BANDS[tour]
        print(f'\n--- {tour}  matched band [{lo}, {hi}] ---')
        for flag in ['High_Leverage_BP', 'High_Leverage_SGP']:
            res = bootstrap(proc, morris[tour], tour, flag, lo, hi)
            print(f'  {flag}: n_treated={res["n_treated"]}  '
                  f'ATE={res["ate_point"]}  CI=[{res["ci_lo"]}, {res["ci_hi"]}]  '
                  f'n_fail={res["n_fail"]}')
            results.append(res)

    out_path = os.path.join(OUT_DIR, 'matched_leverage_bootstrap_results.csv')
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f'\n  Saved {out_path}')
    print('=== Done ===')


if __name__ == '__main__':
    main()
