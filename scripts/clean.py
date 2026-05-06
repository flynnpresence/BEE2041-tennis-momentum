"""
clean.py
--------
Loads raw point-by-point and match metadata files for ATP and WTA.
Filters to 2023 Grand Slams main draw only (drops qualifying rounds).
Casts columns to modern pandas extension types.
Merges datasets with validation.
Adds official rankings using primary GS merge + season median fallback.
Outputs data/processed/atp_cleaned_points.csv and wta_cleaned_points.csv.

Treatment flags (High_Leverage, High_Leverage_BP, High_Leverage_TB),
outcome variables (Point_Won, Next_Point_Won), and player identity
columns are engineered here. Rolling momentum features (CUSUM, BPOR,
TBOE, Streak_k4) are computed separately in features.py.
"""

import os
import numpy as np
import pandas as pd

# ── Paths & Constants ─────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR     = os.path.join(BASE_DIR, 'data', 'raw')
PROC_DIR    = os.path.join(BASE_DIR, 'data', 'processed')
os.makedirs(PROC_DIR, exist_ok=True)

TARGET_YEAR = 2023
GRAND_SLAMS = ['Australian Open', 'Roland Garros', 'Wimbledon', 'US Open']


# ── Data Loading & Cleaning ───────────────────────────────────────────────────
def load_and_filter_matches(filepath: str) -> pd.DataFrame:
    """Load match metadata, clean headers, filter to 2023 Grand Slams main draw."""
    # low_memory=False forces pandas to read the full column before inferring
    # dtype. Without it, mixed-type columns in large files get inferred
    # incorrectly mid-read, producing silent type errors downstream.
    df = pd.read_csv(filepath, low_memory=False)
    df.columns = df.columns.str.strip().str.replace(' ', '_')
    df['match_id']   = df['match_id'].astype('string').str.strip()
    df['Tournament'] = df['Tournament'].astype(
        'string').str.strip().str.replace('_', ' ')

    # errors='coerce' converts unparseable date strings to NaT rather than
    # raising. The raw file contains occasional nulls and non-standard
    # entries; coercing lets us filter by year on the valid rows without
    # crashing on the bad ones.
    df['Date'] = pd.to_datetime(df['Date'].astype(
        str), format='%Y%m%d', errors='coerce')
    df = df[df['Date'].dt.year == TARGET_YEAR]

    # Filter to Grand Slams and drop Juniors. The tournament name field
    # contains entries like "Australian Open Junior", which would otherwise
    # pass the Grand Slam filter. Juniors play under entirely different
    # conditions and should not contribute to a professional momentum study.
    gs_mask      = df['Tournament'].str.contains(
        '|'.join(GRAND_SLAMS), case=False, na=False)
    juniors_mask = df['Tournament'].str.contains(
        'Junior', case=False, na=False)
    df = df[gs_mask & ~juniors_mask]

    # Qualifying rounds are excluded because qualifiers face a different
    # player pool, play under less broadcast scrutiny, and are systematically
    # lower-ranked than main draw entrants. Including them would mix two
    # distinct competitive contexts and bias any ranking-based controls.
    df = df[~df['Round'].str.match(r'^Q\d', na=False)]

    return df


def load_points(filepath: str) -> pd.DataFrame:
    """Load point-by-point CSV and cast to extension types."""
    # low_memory=False, same reason as load_and_filter_matches: prevents
    # mid-file dtype inference errors on large point-by-point CSVs.
    df = pd.read_csv(filepath, low_memory=False)
    df['match_id']  = df['match_id'].astype('string').str.strip()
    df['Pts']       = df['Pts'].astype('string')
    df['Svr']       = df['Svr'].astype('Int64')
    df['PtWinner']  = df['PtWinner'].astype('Int64')
    df['Gm1']       = df['Gm1'].astype('Int64')
    df['Gm2']       = df['Gm2'].astype('Int64')
    df['Set1']      = df['Set1'].astype('Int64')
    df['Set2']      = df['Set2'].astype('Int64')
    df['Pt']        = df['Pt'].astype('Int64')
    df['TbSet']     = df['TbSet'].astype('boolean')
    return df


def prep_rankings_gs(filepath: str, gs_names: list) -> pd.DataFrame:
    """
    Build player + tournament -> ranking lookup from Grand Slam matches only.
    Joins on both Tournament_Key and player name to guarantee m:1 uniqueness.
    Satisfies project log requirement of ranking at match date.
    """
    df = pd.read_csv(filepath, low_memory=False)

    # Walkovers and retirements are dropped because their ranking entries
    # may reflect injury or absence rather than competitive performance,
    # and they do not produce usable point-by-point data in any case.
    df = df[~df['score'].astype(str).str.contains(
        'W/O|RET', case=False, na=False)]

    gs_mask = df['tourney_name'].str.contains(
        '|'.join(gs_names), case=False, na=False
    )
    df = df[gs_mask].copy()

    winners = df[['tourney_name', 'winner_name', 'winner_rank']].rename(
        columns={
            'tourney_name': 'Tournament_Key',
            'winner_name':  'player_name',
            'winner_rank':  'ranking'
        }
    )
    losers = df[['tourney_name', 'loser_name', 'loser_rank']].rename(
        columns={
            'tourney_name': 'Tournament_Key',
            'loser_name':   'player_name',
            'loser_rank':   'ranking'
        }
    )
    rankings = pd.concat([winners, losers], ignore_index=True)
    rankings = rankings.dropna(subset=['ranking'])
    rankings['ranking']        = rankings['ranking'].astype('Int64')
    rankings['player_name']    = rankings['player_name'].astype(
        'string').str.strip()
    rankings['Tournament_Key'] = (
        rankings['Tournament_Key']
        .astype('string')
        .str.strip()
        # The ATP/WTA results file uses "Us Open" (mixed case); the charting
        # project uses "US Open". Normalising here ensures the merge key
        # matches across both sources.
        .str.replace('Us Open', 'US Open', regex=False)
    )

    # A player can appear multiple times in the same tournament if the
    # results file includes qualifying rows. Sort by ranking and keep the
    # first (best) entry so the merge produces exactly one row per player
    # per tournament, a requirement for validate='m:1' to pass.
    rankings = rankings.sort_values('ranking')
    rankings = rankings.drop_duplicates(
        subset=['Tournament_Key', 'player_name'], keep='first'
    )
    return rankings


def prep_rankings_fallback(filepath: str) -> pd.DataFrame:
    """
    Build season-wide median ranking lookup as fallback for qualifiers
    and wildcards missing from the Grand Slam subset.
    One row per player, safe for m:1 merge on player name alone.
    """
    df = pd.read_csv(filepath, low_memory=False)

    # Same walkover/retirement filter as above: consistency matters here
    # because a player who retired mid-tournament may have a missing rank
    # entry that would otherwise contaminate the season median.
    df = df[~df['score'].astype(str).str.contains(
        'W/O|RET', case=False, na=False)]

    winners = df[['winner_name', 'winner_rank']].rename(
        columns={'winner_name': 'player_name', 'winner_rank': 'ranking'}
    )
    losers = df[['loser_name', 'loser_rank']].rename(
        columns={'loser_name': 'player_name', 'loser_rank': 'ranking'}
    )
    rankings = pd.concat([winners, losers], ignore_index=True)
    rankings = rankings.dropna(subset=['ranking'])
    rankings['ranking'] = rankings['ranking'].astype(float)

    # Median rather than mean because ATP/WTA rankings are ordinal and
    # skewed: a player ranked 1 in January and 200 in December should not
    # be assigned rank 100. The median is more robust to injury-driven
    # ranking swings within the season.
    rankings = (
        rankings.groupby('player_name')['ranking'].median().reset_index()
    )
    rankings['ranking']     = rankings['ranking'].round().astype('Int64')
    rankings['player_name'] = rankings['player_name'].astype(
        'string').str.strip()
    return rankings


def process_tour(
    tour_name: str,
    points_file: str,
    matches_file: str,
    rankings_file: str
) -> pd.DataFrame:
    """Executes the load, filter, validated merge, rankings attach, and feature prep."""

    # ATP and WTA are processed through identical logic in separate calls
    # rather than being pooled. Men's and women's tours differ in format
    # (best-of-five vs best-of-three), player depth, and tiebreak rules.
    # Mixing them would require tour as a control variable and risk
    # suppressing tour-specific momentum patterns, exactly the kind of
    # heterogeneity this analysis is designed to detect.
    print(f'\n--- Processing {tour_name} ---')

    matches = load_and_filter_matches(os.path.join(RAW_DIR, matches_file))
    print(
        f'  Retained {len(matches):,} {tour_name} Grand Slam matches for {TARGET_YEAR}')

    pts = load_points(os.path.join(RAW_DIR, points_file))
    print(f'  Loaded {len(pts):,} raw points')

    # validate='m:1' asserts that match_id is unique in the matches table:
    # many points map to one match. '1:1' would be wrong here (multiple
    # points share a match_id) and 'm:m' would allow duplicate matches,
    # silently inflating the dataset. The indicator=True argument keeps the
    # merge result column so we can inspect how many points matched.
    merged = pts.merge(
        matches,
        on='match_id',
        how='inner',
        validate='m:1',
        indicator=True
    )
    print(f'  Merge indicator counts:\n{merged["_merge"].value_counts()}')
    merged = merged.drop(columns=['_merge'])
    print(f'  Retained {len(merged):,} points after merge')
    assert len(merged) > 0, (
        f"Points/matches merge produced empty dataframe for {tour_name}"
    )
    assert 'High_Leverage' not in merged.columns, (
        "High_Leverage already exists before engineering"
    )

    # ── 1. Engineer High_Leverage Treatment Flag ──────────────────────────────
    # Break points: detected from Pts score string
    bp_p1_serving = (merged['Svr'] == 1) & merged['Pts'].astype(
        'string').isin(['0-40', '15-40', '30-40', '40-AD'])
    bp_p2_serving = (merged['Svr'] == 2) & merged['Pts'].astype(
        'string').isin(['40-0', '40-15', '40-30', 'AD-40'])
    is_bp = bp_p1_serving | bp_p2_serving

    # Tiebreak POINTS: score values are NOT standard tennis values {0,15,30,40,AD}
    # Regular game scores use exactly those values; tiebreak uses sequential counting
    pts_split = merged['Pts'].astype('string').str.split('-', expand=True)
    tennis_vals = {'0', '15', '30', '40', 'AD'}
    left_is_tennis  = pts_split[0].isin(tennis_vals)
    right_is_tennis = pts_split[1].isin(tennis_vals)
    is_tb_point = ~(left_is_tennis & right_is_tennis)

    merged['High_Leverage'] = (is_bp | is_tb_point).astype(int)
    merged['High_Leverage_BP'] = is_bp.astype(int)
    merged['High_Leverage_TB'] = is_tb_point.astype(int)

    # ── 2. Apply Random Positional Mask ───────────────────────────────────────
    # Each match has two players. We randomly designate one as the "focal"
    # player per match and track outcomes from their perspective. Without
    # this, every point would be counted twice (once per player), doubling
    # the dataset and creating perfectly correlated duplicate observations
    # that would violate the independence assumption of the causal forest.
    # Seed 42 ensures the same focal assignment is used every pipeline run,
    # making results exactly reproducible.
    np.random.seed(42)
    unique_matches = merged['match_id'].unique()
    match_mask = pd.DataFrame({
        'match_id':    unique_matches,
        'focal_is_p1': np.random.rand(len(unique_matches)) > 0.5
    })
    merged = merged.merge(match_mask, on='match_id', validate='m:1')

    # Identify focal player and opponent names
    merged['Focal_Player'] = np.where(
        merged['focal_is_p1'],
        merged['Player_1'],
        merged['Player_2']
    )
    merged['Opponent_Player'] = np.where(
        merged['focal_is_p1'],
        merged['Player_2'],
        merged['Player_1']
    )

    merged['Point_Won'] = np.where(
        merged['focal_is_p1'],
        merged['PtWinner'] == 1,
        merged['PtWinner'] == 2
    ).astype(int)

    # ── 3. Create Next_Point_Won Outcome Variable ─────────────────────────────
    # The outcome variable is whether the focal player wins the *next* point,
    # not the current one. shift(-1) within each match group moves the
    # following row's Point_Won value back to the current row. After
    # shifting, the last point of each match has no successor and produces
    # NaN: those rows are dropped because they cannot contribute a valid
    # outcome observation. Sorting by Set1/Gm1/Pt first ensures the shift
    # respects the actual chronological sequence of points within a match.
    merged = merged.sort_values(['match_id', 'Set1', 'Gm1', 'Pt'])
    merged['Next_Point_Won'] = (
        merged.groupby('match_id')['Point_Won']
        .shift(-1)
        .astype('Int64')
    )
    merged = merged.dropna(subset=['Next_Point_Won']).copy()
    print(f'  Rows after dropping last points: {len(merged):,}')

    # ── 4. Attach Official Rankings ───────────────────────────────────────────
    # Rankings are attached in two steps to maximise coverage without
    # sacrificing accuracy. The primary merge uses the GS-specific rankings
    # table, which gives each player's ranking at the time of that specific
    # tournament, the most accurate measure of skill at match date.
    # However, qualifiers and some wildcards appear in the charting data but
    # not in the GS results file (they lost in qualifying and are absent from
    # the main draw results). For these players the fallback merge supplies a
    # season-wide median ranking, which is less precise but far better than
    # leaving ranking as missing and dropping those observations entirely.
    # The two-step approach follows the same logic as a primary source /
    # administrative fallback join: use the best available data first, then
    # fill gaps from a broader but less granular source.
    rankings_filepath = os.path.join(RAW_DIR, rankings_file)
    gs_rankings  = prep_rankings_gs(rankings_filepath, GRAND_SLAMS)
    all_rankings = prep_rankings_fallback(rankings_filepath)
    print(f'  GS rankings lookup: {len(gs_rankings):,} rows')
    print(f'  Fallback rankings lookup: {len(all_rankings):,} players')

    # Standardise Tournament_Key to match GS rankings lookup
    merged['Tournament_Key'] = (
        merged['Tournament']
        .astype('string')
        .str.strip()
        .str.replace('Us Open', 'US Open', regex=False)
    )

    # ── Focal player ranking ──────────────────────────────────────────────────
    # validate='m:1' confirms the rankings lookup has exactly one row per
    # (tournament, player) pair. If a duplicate slipped through prep_rankings_gs
    # the merge would silently fan out and inflate row counts. The validator
    # catches that before it contaminates downstream analysis.
    merged = merged.merge(
        gs_rankings.rename(columns={
            'player_name': 'Focal_Player',
            'ranking':     'Focal_Ranking_GS'
        }),
        on=['Tournament_Key', 'Focal_Player'],
        how='left',
        validate='m:1',
        indicator=True
    )
    print(f'  Focal GS merge:\n{merged["_merge"].value_counts()}')
    merged = merged.drop(columns=['_merge'])

    merged = merged.merge(
        all_rankings.rename(columns={
            'player_name': 'Focal_Player',
            'ranking':     'Focal_Ranking_Fallback'
        }),
        on='Focal_Player',
        how='left',
        validate='m:1',
        indicator=True
    )
    print(f'  Focal fallback merge:\n{merged["_merge"].value_counts()}')
    merged = merged.drop(columns=['_merge'])

    merged['Focal_Ranking'] = merged['Focal_Ranking_GS'].fillna(
        merged['Focal_Ranking_Fallback']
    ).astype('Int64')
    merged = merged.drop(
        columns=['Focal_Ranking_GS', 'Focal_Ranking_Fallback'])
    focal_valid = merged['Focal_Ranking'].notna().sum() / len(merged)
    assert focal_valid >= 0.90, (
        f"Focal rankings coverage {focal_valid:.1%} below 90% threshold for {tour_name}"
    )

    # ── Opponent ranking ──────────────────────────────────────────────────────
    merged = merged.merge(
        gs_rankings.rename(columns={
            'player_name': 'Opponent_Player',
            'ranking':     'Opponent_Ranking_GS'
        }),
        on=['Tournament_Key', 'Opponent_Player'],
        how='left',
        validate='m:1',
        indicator=True
    )
    print(f'  Opponent GS merge:\n{merged["_merge"].value_counts()}')
    merged = merged.drop(columns=['_merge'])

    merged = merged.merge(
        all_rankings.rename(columns={
            'player_name': 'Opponent_Player',
            'ranking':     'Opponent_Ranking_Fallback'
        }),
        on='Opponent_Player',
        how='left',
        validate='m:1',
        indicator=True
    )
    print(f'  Opponent fallback merge:\n{merged["_merge"].value_counts()}')
    merged = merged.drop(columns=['_merge'])

    merged['Opponent_Ranking'] = merged['Opponent_Ranking_GS'].fillna(
        merged['Opponent_Ranking_Fallback']
    ).astype('Int64')
    merged = merged.drop(
        columns=['Opponent_Ranking_GS', 'Opponent_Ranking_Fallback'])
    opp_valid = merged['Opponent_Ranking'].notna().sum() / len(merged)
    assert opp_valid >= 0.90, (
        f"Opponent rankings coverage {opp_valid:.1%} below 90% threshold"
        f" for {tour_name}"
    )

    # Ranking difference (positive = focal is worse ranked)
    merged['Ranking_Diff'] = (
        merged['Focal_Ranking'] - merged['Opponent_Ranking']
    ).astype('Int64')

    print(
        f'  Rankings missing: Focal: {merged["Focal_Ranking"].isna().sum()}')
    print(
        f'  Rankings missing: Opponent: {merged["Opponent_Ranking"].isna().sum()}')
    print(f'  Final row count: {len(merged):,}')

    return merged


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print('=== clean.py ===')

    atp = process_tour(
        tour_name='ATP',
        points_file='charting-m-points-2020s.csv',
        matches_file='charting-m-matches.csv',
        rankings_file='atp_matches_2023.csv',
    )
    wta = process_tour(
        tour_name='WTA',
        points_file='charting-w-points-2020s.csv',
        matches_file='charting-w-matches.csv',
        rankings_file='wta_matches_2023.csv',
    )

    atp_out = os.path.join(PROC_DIR, 'atp_cleaned_points.csv')
    wta_out = os.path.join(PROC_DIR, 'wta_cleaned_points.csv')

    atp.to_csv(atp_out, index=False)
    wta.to_csv(wta_out, index=False)

    print(f'\n  Saved atp_cleaned_points.csv ({len(atp):,} rows)')
    print(f'  Saved wta_cleaned_points.csv ({len(wta):,} rows)')
    print('\n=== Done ===')


if __name__ == '__main__':
    main()
