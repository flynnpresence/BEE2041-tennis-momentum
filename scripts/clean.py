"""
clean.py
--------
Loads raw point-by-point and match metadata files for ATP and WTA.
Filters to 2023 Grand Slams main draw only (drops qualifying rounds).
Casts columns to modern pandas extension types.
Merges datasets with validation.
Adds official rankings using primary GS merge + season median fallback.
Outputs data/processed/atp_cleaned_points.csv and wta_cleaned_points.csv.

No feature engineering occurs here — that happens in features.py.
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
    df = pd.read_csv(filepath, low_memory=False)
    df.columns = df.columns.str.strip().str.replace(' ', '_')
    df['match_id']   = df['match_id'].astype('string').str.strip()
    df['Tournament'] = df['Tournament'].astype('string').str.strip().str.replace('_', ' ')
    df['Date'] = pd.to_datetime(df['Date'].astype(str), format='%Y%m%d', errors='coerce')
    df = df[df['Date'].dt.year == TARGET_YEAR]

    # Filter to Grand Slams and drop Juniors
    gs_mask      = df['Tournament'].str.contains('|'.join(GRAND_SLAMS), case=False, na=False)
    juniors_mask = df['Tournament'].str.contains('Junior', case=False, na=False)
    df = df[gs_mask & ~juniors_mask]

    # Drop qualifying rounds — main draw only
    df = df[~df['Round'].str.startswith('Q', na=False)]

    return df


def load_points(filepath: str) -> pd.DataFrame:
    """Load point-by-point CSV and cast to extension types."""
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

    # Drop walkovers and retirements
    df = df[~df['score'].astype(str).str.contains('W/O|RET', case=False, na=False)]

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
    rankings['player_name']    = rankings['player_name'].astype('string').str.strip()
    rankings['Tournament_Key'] = (
        rankings['Tournament_Key']
        .astype('string')
        .str.strip()
        .str.replace('Us Open', 'US Open', regex=False)
    )

    # One row per player per tournament
    rankings = rankings.sort_values('ranking')
    rankings = rankings.drop_duplicates(
        subset=['Tournament_Key', 'player_name'], keep='first'
    )
    return rankings


def prep_rankings_fallback(filepath: str) -> pd.DataFrame:
    """
    Build season-wide median ranking lookup as fallback for qualifiers
    and wildcards missing from the Grand Slam subset.
    One row per player — safe for m:1 merge on player name alone.
    """
    df = pd.read_csv(filepath, low_memory=False)

    # Drop walkovers and retirements
    df = df[~df['score'].astype(str).str.contains('W/O|RET', case=False, na=False)]

    winners = df[['winner_name', 'winner_rank']].rename(
        columns={'winner_name': 'player_name', 'winner_rank': 'ranking'}
    )
    losers = df[['loser_name', 'loser_rank']].rename(
        columns={'loser_name': 'player_name', 'loser_rank': 'ranking'}
    )
    rankings = pd.concat([winners, losers], ignore_index=True)
    rankings = rankings.dropna(subset=['ranking'])
    rankings['ranking'] = rankings['ranking'].astype(float)

    # Median rank across full season — one row per player
    rankings = rankings.groupby('player_name')['ranking'].median().reset_index()
    rankings['ranking']     = rankings['ranking'].round().astype('Int64')
    rankings['player_name'] = rankings['player_name'].astype('string').str.strip()
    return rankings


def process_tour(
    tour_name: str,
    points_file: str,
    matches_file: str,
    rankings_file: str
) -> pd.DataFrame:
    """Executes the load, filter, validated merge, rankings attach, and feature prep."""
    print(f'\n--- Processing {tour_name} ---')

    matches = load_and_filter_matches(os.path.join(RAW_DIR, matches_file))
    print(f'  Retained {len(matches):,} {tour_name} Grand Slam matches for {TARGET_YEAR}')

    pts = load_points(os.path.join(RAW_DIR, points_file))
    print(f'  Loaded {len(pts):,} raw points')

    # Merge points with match metadata
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

    # ── 1. Engineer High_Leverage Treatment Flag ──────────────────────────────
    bp_p1_serving = (merged['Svr'] == 1) & merged['Pts'].astype('string').str.endswith(('-40', '-AD'))
    bp_p2_serving = (merged['Svr'] == 2) & merged['Pts'].astype('string').str.startswith(('40-', 'AD-'))
    is_bp = bp_p1_serving | bp_p2_serving
    merged['High_Leverage'] = (is_bp | merged['TbSet'].fillna(False)).astype(int)

    # ── 2. Apply Random Positional Mask ───────────────────────────────────────
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
    merged['Next_Point_Won'] = (
        merged.groupby('match_id')['Point_Won']
        .shift(-1)
        .astype('Int64')
    )
    merged = merged.dropna(subset=['Next_Point_Won']).copy()
    print(f'  Rows after dropping last points: {len(merged):,}')

    # ── 4. Attach Official Rankings ───────────────────────────────────────────
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
    merged = merged.drop(columns=['Focal_Ranking_GS', 'Focal_Ranking_Fallback'])

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
    merged = merged.drop(columns=['Opponent_Ranking_GS', 'Opponent_Ranking_Fallback'])

    # Ranking difference (positive = focal is worse ranked)
    merged['Ranking_Diff'] = (
        merged['Focal_Ranking'] - merged['Opponent_Ranking']
    ).astype('Int64')

    print(f'  Rankings missing — Focal: {merged["Focal_Ranking"].isna().sum()}')
    print(f'  Rankings missing — Opponent: {merged["Opponent_Ranking"].isna().sum()}')
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