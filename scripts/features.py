"""
features.py
-----------
Loads cleaned point-by-point data for ATP and WTA.
Engineers momentum and control features strictly forward-rolling.
Runs per-player Chi-squared and runs tests (Warning 7).
Outputs data/processed/processed_features.csv as the model checkpoint.

All features computed within match groups — no look-ahead bias.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats


# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, 'data', 'processed')

# ── Constants ─────────────────────────────────────────────────────────────────
STREAK_K    = 4    # Consecutive points for hot hand flag
ROLLING_WIN = 10   # Rolling window for win percentage
ALPHA       = 0.05  # Significance level for per-player tests


# ── Feature Engineering ───────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers all momentum and control features for a single tour dataframe.
    All features are strictly forward-rolling within each match.

    Features added:
        Streak_k4       : 1 if focal player won last STREAK_K consecutive points
        Rolling_Win_Pct : Rolling win % over last ROLLING_WIN points in match
        CUSUM           : Cumulative sum of deviations from expanding mean
        BPOR            : Rolling historical BP win rate minus return point win rate
        TBOE            : Rolling historical TB win rate minus rolling win %
    """
    df = df.sort_values(['match_id', 'Set1', 'Gm1', 'Pt']).copy()

    # ── Streak_k4 ─────────────────────────────────────────────────────────────
    df['Streak_k4'] = (
        df.groupby('match_id')['Point_Won']
        .transform(lambda x: x.shift(1).rolling(STREAK_K, min_periods=STREAK_K).min())
        .fillna(0)
        .astype(int)
    )

    # ── Rolling_Win_Pct ───────────────────────────────────────────────────────
    df['Rolling_Win_Pct'] = (
        df.groupby('match_id')['Point_Won']
        .transform(lambda x: x.shift(1).rolling(ROLLING_WIN, min_periods=1).mean())
        .fillna(0.5)
    )

    # ── CUSUM (fixed — expanding mean, no look-ahead) ─────────────────────────
    def cusum_expanding(series: pd.Series) -> pd.Series:
        shifted = series.shift(1).fillna(0)
        # Expanding mean uses only past observations — no look-ahead
        expanding_mean = shifted.expanding().mean()
        deviations = shifted - expanding_mean
        return deviations.cumsum()

    df['CUSUM'] = (
        df.groupby('match_id')['Point_Won']
        .transform(cusum_expanding)
    )

    # ── BPOR & TBOE — Forward-Rolling Luck Proxies ───────────────────────────
    # np.where avoids .map() boolean dtype issues in groupby context
    df['is_returning'] = (df['Svr'] != np.where(
        df['focal_is_p1'], 1, 2)).astype(int)

    bp_p1_serving = (df['Svr'] == 1) & df['Pts'].astype(
        'string').str.endswith(('-40', '-AD'))
    bp_p2_serving = (df['Svr'] == 2) & df['Pts'].astype(
        'string').str.startswith(('40-', 'AD-'))
    df['is_bp'] = (bp_p1_serving | bp_p2_serving).astype(int)

    # Point-level tiebreak flag — consistent with clean.py, more reliable than TbSet
    _pts_split = df['Pts'].astype('string').str.split('-', expand=True)
    _tennis_vals = {'0', '15', '30', '40', 'AD'}
    df['_is_tb_point'] = (~(
        _pts_split[0].isin(_tennis_vals) & _pts_split[1].isin(_tennis_vals)
    )).astype(int)

    def calculate_luck_proxies(match_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate BPOR and TBOE using historical (shifted) expanding sums."""
        match_id = match_df['match_id'].iloc[0]
        # BPOR: historical break point win rate minus historical return point win rate
        ret_played = match_df['is_returning'].shift(1).expanding().sum()
        ret_won    = (match_df['is_returning']
                      * match_df['Point_Won']).shift(1).expanding().sum()
        ret_rate   = (ret_won / ret_played).fillna(0.38)

        bp_played  = match_df['is_bp'].shift(1).expanding().sum()
        bp_won     = (match_df['is_bp']
                      * match_df['Point_Won']).shift(1).expanding().sum()
        bp_rate    = (bp_won / bp_played).fillna(0.38)

        match_df['BPOR'] = bp_rate - ret_rate

        # TBOE: uses point-level flag, not TbSet (set-level and unreliable)
        tb_played  = match_df['_is_tb_point'].shift(1).expanding().sum()
        tb_won     = (match_df['_is_tb_point']
                      * match_df['Point_Won']).shift(1).expanding().sum()
        tb_rate    = (tb_won / tb_played).fillna(0.5)

        match_df['TBOE'] = tb_rate - match_df['Rolling_Win_Pct']
        match_df['match_id'] = match_id

        return match_df

    df = pd.concat(
        [calculate_luck_proxies(group) for _, group in df.groupby('match_id')],
        ignore_index=True
    )
    df = df.drop(columns=['is_returning', 'is_bp', '_is_tb_point'])

    return df


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Selects and casts the final feature set for modelling.
    """
    keep_cols = [
        'match_id',
        'Pt',
        'Tournament',
        'Round',
        'Surface',
        'Focal_Player',
        'Opponent_Player',
        'Focal_Ranking',
        'Opponent_Ranking',
        'Ranking_Diff',
        'Svr',
        'TbSet',
        'High_Leverage',
        'High_Leverage_BP',
        'High_Leverage_TB',
        'Streak_k4',
        'Rolling_Win_Pct',
        'CUSUM',
        'BPOR',
        'TBOE',
        'Point_Won',
        'Next_Point_Won',
    ]
    df = df[keep_cols].copy()

    df['Surface']         = df['Surface'].astype('string')
    df['Round']           = df['Round'].astype('string')
    df['Tournament']      = df['Tournament'].astype('string')
    df['Focal_Player']    = df['Focal_Player'].astype('string')
    df['Opponent_Player'] = df['Opponent_Player'].astype('string')
    df['Rolling_Win_Pct'] = df['Rolling_Win_Pct'].astype(float)
    df['CUSUM']           = df['CUSUM'].astype(float)
    df['BPOR']            = df['BPOR'].astype(float)
    df['TBOE']            = df['TBOE'].astype(float)

    return df


# ── Per-Player Statistical Tests (Warning 7) ──────────────────────────────────
def run_per_player_tests(df: pd.DataFrame, tour_name: str) -> pd.DataFrame:
    """
    Runs per-player Chi-squared independence test and Wald-Wolfowitz runs test.
    Warning 7: Do not pool players — aggregation bias risk.

    Returns a summary DataFrame with one row per player.
    """
    print(f'\n  Running per-player tests for {tour_name}...')
    results = []

    players = df['Focal_Player'].unique()

    for player in players:
        pdata = df[df['Focal_Player'] == player].copy()

        if len(pdata) < 20:
            continue

        # ── Chi-squared independence test ─────────────────────────────────────
        # Tests whether Point_Won and Next_Point_Won are independent
        try:
            contingency = pd.crosstab(
                pdata['Point_Won'],
                pdata['Next_Point_Won']
            )
            if contingency.shape == (2, 2):
                chi2, p_chi2, _, _ = stats.chi2_contingency(contingency)
            else:
                chi2, p_chi2 = np.nan, np.nan
        except Exception:
            chi2, p_chi2 = np.nan, np.nan

        # ── Wald-Wolfowitz runs test ───────────────────────────────────────────
        # Tests whether the sequence of wins/losses is random
        try:
            sequence = pdata['Point_Won'].values
            n1 = sequence.sum()
            n2 = len(sequence) - n1
            runs = 1 + np.sum(sequence[1:] != sequence[:-1])
            # Expected runs and variance under null of randomness
            exp_runs = (2 * n1 * n2 / (n1 + n2)) + 1
            var_runs = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / (
                ((n1 + n2) ** 2) * (n1 + n2 - 1)
            )
            z_runs = (
                (runs - exp_runs) / np.sqrt(var_runs)
                if var_runs > 0 else np.nan
            )
            p_runs = 2 * (1 - stats.norm.cdf(abs(z_runs))
                          ) if not np.isnan(z_runs) else np.nan
        except Exception:
            z_runs, p_runs = np.nan, np.nan

        results.append({
            'Tour':            tour_name,
            'Player':          player,
            'N_Points':        len(pdata),
            'Win_Rate':        pdata['Point_Won'].mean(),
            'Chi2_Stat':       chi2,
            'Chi2_P':          p_chi2,
            'Chi2_Sig':        int(p_chi2 < ALPHA) if not np.isnan(p_chi2) else np.nan,
            'Runs_Z':          z_runs,
            'Runs_P':          p_runs,
            'Runs_Sig':        int(p_runs < ALPHA) if not np.isnan(p_runs) else np.nan,
        })

    results_df = pd.DataFrame(results)

    n_tested   = len(results_df)
    n_chi2_sig = results_df['Chi2_Sig'].sum()
    n_runs_sig = results_df['Runs_Sig'].sum()

    print(f'  Players tested: {n_tested}')
    pct_chi2 = 100 * n_chi2_sig / n_tested
    pct_runs = 100 * n_runs_sig / n_tested
    print(f'  Chi-squared significant (p<{ALPHA}): {n_chi2_sig} ({pct_chi2:.1f}%)')
    print(f'  Runs test significant (p<{ALPHA}): {n_runs_sig} ({pct_runs:.1f}%)')

    return results_df


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print('=== features.py ===')

    print('\nLoading ATP cleaned points...')
    atp = pd.read_csv(os.path.join(
        PROC_DIR, 'atp_cleaned_points.csv'), low_memory=False)
    print(f'  Loaded {len(atp):,} rows')

    print('Loading WTA cleaned points...')
    wta = pd.read_csv(os.path.join(
        PROC_DIR, 'wta_cleaned_points.csv'), low_memory=False)
    print(f'  Loaded {len(wta):,} rows')

    # Engineer features separately — no cross-tour contamination
    print('\nEngineering ATP features...')
    atp = engineer_features(atp)
    atp = select_features(atp)
    atp['Tour'] = 'ATP'
    print(f'  ATP feature rows: {len(atp):,}')

    print('Engineering WTA features...')
    wta = engineer_features(wta)
    wta = select_features(wta)
    wta['Tour'] = 'WTA'
    print(f'  WTA feature rows: {len(wta):,}')

    # Per-player statistical tests (Warning 7 — no pooling)
    atp_tests = run_per_player_tests(atp, 'ATP')
    wta_tests = run_per_player_tests(wta, 'WTA')
    all_tests = pd.concat([atp_tests, wta_tests], ignore_index=True)

    # Save test results for Output 1
    tests_path = os.path.join(PROC_DIR, 'per_player_tests.csv')
    all_tests.to_csv(tests_path, index=False)
    print(f'\n  Saved per_player_tests.csv ({len(all_tests):,} players)')

    # Combine tours and save checkpoint
    combined = pd.concat([atp, wta], ignore_index=True)
    combined['Tour'] = combined['Tour'].astype('string')

    out_path = os.path.join(PROC_DIR, 'processed_features.csv')
    combined.to_csv(out_path, index=False)

    print(f'\n  Total rows: {len(combined):,}')
    print(f'  Saved processed_features.csv')
    print(f'\n  ATP Streak_k4 mean:       {atp["Streak_k4"].mean():.4f}')
    print(f'  ATP Rolling_Win_Pct mean:  {atp["Rolling_Win_Pct"].mean():.4f}')
    print(f'  ATP CUSUM mean:            {atp["CUSUM"].mean():.4f}')
    print(f'  ATP BPOR mean:             {atp["BPOR"].mean():.4f}')
    print(f'  ATP TBOE mean:             {atp["TBOE"].mean():.4f}')
    print(f'\n  WTA Streak_k4 mean:       {wta["Streak_k4"].mean():.4f}')
    print(f'  WTA Rolling_Win_Pct mean:  {wta["Rolling_Win_Pct"].mean():.4f}')
    print(f'  WTA CUSUM mean:            {wta["CUSUM"].mean():.4f}')
    print(f'  WTA BPOR mean:             {wta["BPOR"].mean():.4f}')
    print(f'  WTA TBOE mean:             {wta["TBOE"].mean():.4f}')
    print('\n=== Done ===')


if __name__ == '__main__':
    main()
