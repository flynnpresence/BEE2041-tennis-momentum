"""
bootstrap_ate.py
----------------
Match-clustered bootstrap standard errors for the Causal Forest ATE.

WHY THIS EXISTS
    model.py computes  ate_se = cates.std() / sqrt(n).
    That is the dispersion of the forest's per-unit CATE *predictions* divided by
    sqrt(n) -- NOT the sampling variance of the ATE. It also treats points as
    independent, but points within a match are serially dependent (the entire
    subject of this study). econml's own inference (bootstrap-of-little-bags) is
    better but still assumes independent observations, so it is also too small
    under within-match dependence. This module resamples MATCHES (not points),
    refits the exact same estimator on each resample, and takes the spread of the
    resulting ATEs as the standard error -- the assumption-light, referee-standard
    fix.

WHAT IT REPORTS PER SPEC
    ate_point        : ATE at the published config (seed=42) -- should match model.py
    blb_se           : econml bootstrap-of-little-bags SE  (independent-obs; reference)
    cluster_boot_se  : match-clustered bootstrap SE        (headline SE)
    ci_lo, ci_hi     : 2.5 / 97.5 percentile CI from the cluster bootstrap
    n_fail           : bootstrap replicates that errored (watch this for TB specs)

    The honest inference ladder is:  naive s/sqrt(n)  <  blb_se  <  cluster_boot_se

USAGE
    python scripts/bootstrap_ate.py --spec atp_bp --B 199
    python scripts/bootstrap_ate.py --spec all --B 199 --n-jobs 8

NOTE ON RUNTIME
    Every spec (including *_tb) is stratified-subsampled to the SAME 15,000-row
    cap as model.py, so every replicate costs roughly the same forest-fit time --
    there is no "cheap" spec. wta_tb (209 treated) is the most numerically
    fragile, not the fastest; calibrate per-replicate wall time with B=1 before
    committing to a large B.

NOTE ON GROUPED CROSS-FITTING
    This passes groups=match_id to cf.fit(), so DML cross-fitting keeps each match
    within a single fold (model.py's plain cv=2 did not, a mild leak). Check that
    ate_point here is close to model.py's published value. If it moves materially,
    that movement is itself evidence the old non-grouped CV was leaking -- report it.

NOTE ON WHAT IS AND ISN'T COPIED
    build_spec() and fit_ate() reproduce model.py's treatment construction, the
    15,000-row stratified subsample, and the CausalForestDML config VERBATIM, so the
    bootstrapped estimator is identical to the published one. If you change model.py,
    change it here too (or import from a shared module).
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
from econml.dml import CausalForestDML, LinearDML
from sklearn.ensemble import GradientBoostingRegressor

warnings.filterwarnings('ignore', category=UserWarning, module='econml')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='econml')

try:
    from joblib import Parallel, delayed
    _HAVE_JOBLIB = True
except Exception:
    _HAVE_JOBLIB = False

# ── Constants: MUST match model.py ────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR  = os.path.join(BASE_DIR, 'data', 'processed')
CONTROLS  = ['Focal_Ranking', 'Rolling_Win_Pct', 'Streak_k4', 'CUSUM']
OUTCOME   = 'Next_Point_Won'
SEED      = 42
SUBSAMPLE_CAP = 15000

# spec key -> (tour, treatment_label, controls)
SPECS = {
    'atp_combined': ('ATP', 'combined', CONTROLS),
    'wta_combined': ('WTA', 'combined', CONTROLS),
    'atp_bp':       ('ATP', 'bp',       CONTROLS),
    'wta_bp':       ('WTA', 'bp',       CONTROLS),
    'atp_tb':       ('ATP', 'tb',       CONTROLS),
    'wta_tb':       ('WTA', 'tb',       CONTROLS),
    # §7a.1 matched-comparison group: high-leverage points where the winner
    # does not thereby serve the next point (see clean.py's High_Leverage_SGP).
    # Well-powered (3,645 ATP / 1,464 WTA treated, both larger than atp_bp /
    # wta_bp), so it stays on the forest like the BP specs.
    'atp_sgp':      ('ATP', 'sgp',      CONTROLS),
    'wta_sgp':      ('WTA', 'sgp',      CONTROLS),
    # ranking-only robustness specs (Combined + BP only; TB is unstable at 1 control,
    # per model.py's own note -- omitted deliberately)
    'atp_combined_rank': ('ATP', 'combined', ['Focal_Ranking']),
    'wta_combined_rank': ('WTA', 'combined', ['Focal_Ranking']),
    'atp_bp_rank':       ('ATP', 'bp',       ['Focal_Ranking']),
    'wta_bp_rank':       ('WTA', 'bp',       ['Focal_Ranking']),
}

# ESTIMATOR-SPLIT RULE
# ---------------------
# CausalForestDML is used only where treatment is well-powered AND the forest's
# defining feature -- per-unit heterogeneous effects -- is actually consumed
# downstream (atp_combined / wta_combined feed Output 4's CATE-by-ranking-band
# plot and Output 7's feature importances). atp_bp / wta_bp are also
# well-powered (1,497 / 1,001 treated) and stay on the forest.
#
# LinearDML is used everywhere the forest's point estimate is not reliably
# identified, for two distinct, independently-confirmed reasons:
#   1. Sparse treatment (atp_tb: 729 treated / 169 matches, wta_tb: 209 / 128).
#      A cv-fold sweep (cv=2/3/5, grouped vs ungrouped) showed wta_tb's forest
#      ATE swinging sign and magnitude under grouped cross-fitting (+0.039 /
#      -0.104 / -0.015) despite looking stable ungrouped -- grouped
#      cross-fitting is the methodologically correct fix for cross-fit leakage,
#      but this cell doesn't have enough independent matches to support it.
#   2. Single-control propensity fragility (the four *_rank robustness specs).
#      These have plenty of treated points but only one covariate; a cluster
#      bootstrap (B=199) showed a small fraction of resamples (0-2/199, worse
#      for wta_combined_rank) producing degenerate propensity scores and
#      wildly outlying ATEs (e.g. -3.91) under the forest, inflating its SE by
#      an order of magnitude while the percentile CI stayed sane.
# In all four rank-only specs and both tiebreak specs, re-fitting with
# LinearDML instead of the forest reproduced the forest's own outlier-trimmed
# median ATE to within ~5% with a clean, uncontaminated bootstrap distribution
# -- confirming the forest's instability was estimator noise, not a real
# disagreement about the effect, and that LinearDML is the stable tool for
# these six cells rather than a workaround chosen to get a preferred number.
LINEAR_SPECS = {
    'atp_tb', 'wta_tb',
    'atp_bp_rank', 'wta_bp_rank', 'atp_combined_rank', 'wta_combined_rank',
}


def build_spec(df_tour: pd.DataFrame, treatment_label: str) -> pd.DataFrame:
    """Construct the Treatment column exactly as model.py does. Keeps match_id.
    Returns the FULL (un-subsampled) spec frame; subsampling happens in fit_ate so
    that its randomness is captured by the bootstrap."""
    extra = ['Point_Won']
    if treatment_label == 'bp':
        extra.append('High_Leverage_BP')
        flag = 'High_Leverage_BP'
    elif treatment_label == 'tb':
        extra.append('High_Leverage_TB')
        flag = 'High_Leverage_TB'
    elif treatment_label == 'sgp':
        extra.append('High_Leverage_SGP')
        flag = 'High_Leverage_SGP'
    else:
        flag = 'High_Leverage'

    keep = list(dict.fromkeys(CONTROLS + [OUTCOME, 'match_id'] + extra + [flag]))
    data = df_tour[keep].dropna().copy()
    data['Treatment'] = ((data[flag] == 1) & (data['Point_Won'] == 1)).astype(float)
    return data


def fit_ate(data: pd.DataFrame, controls: list, subsample_seed: int,
            forest_seed: int = SEED, want_blb: bool = False, n_jobs: int = 1,
            estimator: str = 'forest'):
    """Subsample (as model.py), fit the estimator with grouped cross-fitting,
    return (ate, blb_se_or_None). 'forest' config is verbatim from model.py.
    'linear' (LinearDML) is used for the two rare-treatment tiebreak cells,
    where the forest's point estimate is not identified (see LINEAR_SPECS)."""
    d = data
    # Stratified subsample: keep all treated, sample controls to reach the cap.
    if len(d) > SUBSAMPLE_CAP:
        treated = d[d['Treatment'] == 1]
        n_control = min(len(d) - len(treated), max(0, SUBSAMPLE_CAP - len(treated)))
        control = d[d['Treatment'] == 0].sample(n_control, random_state=subsample_seed)
        d = pd.concat([treated, control])

    T = d['Treatment'].values
    Y = d[OUTCOME].astype(float).values
    X = d[controls].astype(float).values
    groups = d['match_id'].values  # grouped cross-fitting: keep a match within one fold

    nuisance_kwargs = dict(
        model_y=GradientBoostingRegressor(n_estimators=200, random_state=forest_seed),
        model_t=GradientBoostingRegressor(n_estimators=200, random_state=forest_seed),
        cv=2,
        random_state=forest_seed,
    )

    if estimator == 'linear':
        cf = LinearDML(**nuisance_kwargs)
        cf.fit(Y, T, X=X, groups=groups)
        ate = float(cf.ate(X))
        blb_se = None
        if want_blb:
            try:
                inf = cf.ate_inference(X=X)
                blb_se = float(inf.stderr_mean)
            except Exception:
                blb_se = float('nan')
        return ate, blb_se

    # inference=True changes CausalForestDML's internal RNG consumption (BLB
    # needs its own random subsampling of trees/data), which shifts the point
    # estimate by a small but nonzero amount even at a fixed random_state. The
    # point estimate must not depend on whether the caller also wants the BLB
    # reference SE, so it's always fit with the plain (inference=False) config;
    # BLB, when wanted, comes from a fully separate fit whose own .ate() is
    # discarded.
    cf = CausalForestDML(
        n_estimators=200,
        n_jobs=n_jobs,
        verbose=0,
        **nuisance_kwargs,
    )
    cf.fit(Y, T, X=X, groups=groups)
    ate = float(cf.ate(X))

    blb_se = None
    if want_blb:
        try:
            cf_blb = CausalForestDML(
                n_estimators=200,
                n_jobs=n_jobs,
                inference=True,
                verbose=0,
                **nuisance_kwargs,
            )
            cf_blb.fit(Y, T, X=X, groups=groups)
            inf = cf_blb.ate_inference(X=X)
            blb_se = float(inf.stderr_mean)
        except Exception:
            blb_se = float('nan')
    return ate, blb_se


def cluster_resample(data: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Resample match_ids with replacement; reassign fresh unique ids to drawn
    matches so duplicates are treated as distinct clusters (correct for the
    cluster bootstrap and for grouped cross-fitting)."""
    matches = data['match_id'].unique()
    drawn = rng.choice(matches, size=len(matches), replace=True)
    parts = []
    for k, m in enumerate(drawn):
        rows = data[data['match_id'] == m].copy()
        rows['match_id'] = f'boot_{k}'   # fresh unique cluster id
        parts.append(rows)
    return pd.concat(parts, ignore_index=True)


def bootstrap_spec(df: pd.DataFrame, spec_key: str, B: int,
                   seed: int = SEED, n_jobs: int = 1) -> dict:
    tour, tlabel, controls = SPECS[spec_key]
    estimator = 'linear' if spec_key in LINEAR_SPECS else 'forest'
    data = build_spec(df[df['Tour'] == tour].copy(), tlabel)
    n_treated = int(data['Treatment'].sum())
    n_matches = data['match_id'].nunique()

    # Point estimate at the published config (seed=42), plus BLB reference SE.
    ate_point, blb_se = fit_ate(data, controls, subsample_seed=SEED,
                                forest_seed=SEED, want_blb=True, n_jobs=1,
                                estimator=estimator)

    # Group-level resampling grid uses distinct seeds so subsample noise is captured.
    def one_rep(b):
        rng = np.random.default_rng(seed + 1 + b)
        boot = cluster_resample(data, rng)
        try:
            ate_b, _ = fit_ate(boot, controls, subsample_seed=seed + 1 + b,
                               forest_seed=SEED, want_blb=False, n_jobs=1,
                               estimator=estimator)
            return ate_b
        except Exception:
            return np.nan

    if _HAVE_JOBLIB and n_jobs != 1:
        reps = Parallel(n_jobs=n_jobs)(delayed(one_rep)(b) for b in range(B))
    else:
        reps = [one_rep(b) for b in range(B)]

    reps = np.array(reps, dtype=float)
    ok = reps[~np.isnan(reps)]
    n_fail = int(np.isnan(reps).sum())

    return {
        'spec': spec_key, 'tour': tour, 'type': tlabel, 'estimator': estimator,
        'n_matches': n_matches, 'n_treated': n_treated,
        'ate_point': round(ate_point, 4),
        'blb_se': round(blb_se, 4) if blb_se is not None else None,
        'cluster_boot_se': round(float(ok.std(ddof=1)), 4),
        'ci_lo': round(float(np.percentile(ok, 2.5)), 4),
        'ci_hi': round(float(np.percentile(ok, 97.5)), 4),
        'B_effective': len(ok), 'n_fail': n_fail,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', default='all',
                    help="one of " + ", ".join(SPECS) + ", or 'all'")
    ap.add_argument('--B', type=int, default=199, help='bootstrap replicates')
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--n-jobs', type=int, default=1,
                    help='parallel replicates (forest stays single-threaded)')
    ap.add_argument('--out', default=os.path.join(BASE_DIR, 'outputs',
                                                  'ate_results_bootstrap.csv'))
    args = ap.parse_args()

    df = pd.read_csv(os.path.join(PROC_DIR, 'processed_features.csv'),
                     low_memory=False)

    specs = list(SPECS) if args.spec == 'all' else [args.spec]
    rows = []
    for s in specs:
        print(f'\n=== {s}  (B={args.B}) ===')
        res = bootstrap_spec(df, s, B=args.B, seed=args.seed, n_jobs=args.n_jobs)
        for k, v in res.items():
            print(f'  {k:16s}: {v}')
        if res['n_fail'] > 0:
            print(f"  !! {res['n_fail']} replicate(s) failed -- inspect before trusting this spec's SE")
        rows.append(res)

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f'\nSaved {args.out}')
    print('\nInference ladder check (expect naive < blb_se < cluster_boot_se):')
    print(out[['spec', 'ate_point', 'blb_se', 'cluster_boot_se', 'ci_lo', 'ci_hi']]
          .to_string(index=False))


if __name__ == '__main__':
    main()
