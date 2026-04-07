"""
clean.py
--------
Loads raw point-by-point and match metadata files for ATP and WTA.
Filters to 2023 Grand Slams only.
Casts columns to modern pandas extension types.
Merges datasets with validation.
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
    """Load match metadata, clean headers, and filter to 2023 Grand Slams."""
    df = pd.read_csv(filepath, low_memory=False)
    df.columns = df.columns.str.strip().str.replace(' ', '_')
    df['match_id']   = df['match_id'].astype('string').str.strip()
    df['Tournament'] = df['Tournament'].astype('string').str.strip().str.replace('_', ' ')
    df['Date'] = pd.to_datetime(df['Date'].astype(str), format='%Y%m%d', errors='coerce')
    df = df[df['Date'].dt.year == TARGET_YEAR]
    gs_mask      = df['Tournament'].str.contains('|'.join(GRAND_SLAMS), case=False, na=False)
    juniors_mask = df['Tournament'].str.contains('Junior', case=False, na=False)
    df = df[gs_mask & ~juniors_mask]
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


def process_tour(
    tour_name: str,
    points_file: str,
    matches_file: str
) -> pd.DataFrame:
    """Executes the load, filter, validated merge, and feature prep for a specific tour."""
    print(f'\n--- Processing {tour_name} ---')

    matches = load_and_filter_matches(os.path.join(RAW_DIR, matches_file))
    print(f'  Retained {len(matches):,} {tour_name} Grand Slam matches for {TARGET_YEAR}')

    pts = load_points(os.path.join(RAW_DIR, points_file))
    print(f'  Loaded {len(pts):,} raw points')

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
        'match_id': unique_matches,
        'focal_is_p1': np.random.rand(len(unique_matches)) > 0.5
    })
    merged = merged.merge(match_mask, on='match_id', validate='m:1')
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
    print(f'  Final row count after dropping last points: {len(merged):,}')

    return merged


def main() -> None:
    """
    Run the full cleaning pipeline for ATP and WTA.

    Loads raw files from data/raw/, filters to 2023 Grand Slams, merges
    match metadata with point-by-point data, engineers outcome variables,
    and saves two cleaned CSVs to data/processed/.
    """
    print('=== clean.py ===')

    atp = process_tour(
        tour_name='ATP',
        points_file='charting-m-points-2020s.csv',
        matches_file='charting-m-matches.csv',
    )
    wta = process_tour(
        tour_name='WTA',
        points_file='charting-w-points-2020s.csv',
        matches_file='charting-w-matches.csv',
    )

    atp_out = os.path.join(PROC_DIR, 'atp_cleaned_points.csv')
    wta_out = os.path.join(PROC_DIR, 'wta_cleaned_points.csv')

    atp.to_csv(atp_out, index=False)
    wta.to_csv(wta_out, index=False)

    print(f'\nSaved: {atp_out}')
    print(f'Saved: {wta_out}')
    print('\n=== Done ===')


if __name__ == '__main__':
    main()