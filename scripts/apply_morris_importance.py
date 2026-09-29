"""
apply_morris_importance.py
---------------------------
Scores every won break-point / server-game-point in the cleaned points data
with the Morris (1977) / Klaassen & Magnus (2001) point-importance measure
(morris_importance.py), and reports the BP-vs-SGP leverage-comparability
result written up in methodology.md Β§7a.3 / Β§4d.

Outputs data/processed/{atp,wta}_morris_importance.csv: the won-BP/SGP subset
of each tour's cleaned points, with a Morris_Importance column added.
"""

import os
import pandas as pd
from scipy import stats

from morris_importance import point_importance, make_match_prob

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, 'data', 'processed')

PTS_MAP = {'0': 0, '15': 1, '30': 2, '40': 3, 'AD': 4}

# Estimated directly from each tour's cleaned points data (overall fraction
# of points won by whoever is serving), not assumed -- see clean.py's Svr/
# PtWinner columns. Recompute via:
#   (df['PtWinner'] == df['Svr']).mean()
ATP_P, WTA_P = 0.6386, 0.5686


def compute_importance_for_tour(tour: str, filename: str, p: float, best_of: int) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(PROC_DIR, filename), low_memory=False)
    is_bp = df['High_Leverage_BP'] == 1
    is_sgp = df['High_Leverage_SGP'] == 1
    d = df[(is_bp | is_sgp) & (df['Point_Won'] == 1)].copy()
    print(f'  {tour}: scoring {len(d)} won-BP/SGP points')

    # Built ONCE per tour and reused across every point -- rebuilding the
    # match/set/tiebreak machinery per point would rebuild the tiebreak
    # table's full bottom-up DP on every single row (see morris_importance.py
    # docstring and methodology.md Β§4d).
    M = make_match_prob(p, best_of)

    importances = []
    for _, row in d.iterrows():
        a_str, b_str = row['Pts'].split('-')
        a, b = PTS_MAP[a_str], PTS_MAP[b_str]
        # Pts is always server-receiver format regardless of which player
        # (1 or 2) is serving -- see clean.py's is_bp/is_sgp comments. Only
        # which of Set1/Set2, Gm1/Gm2 maps to "server's" sx/gx vs
        # "receiver's" sy/gy depends on Svr.
        if row['Svr'] == 1:
            sx, sy, gx, gy = row['Set1'], row['Set2'], row['Gm1'], row['Gm2']
        else:
            sx, sy, gx, gy = row['Set2'], row['Set1'], row['Gm2'], row['Gm1']
        imp = point_importance(p, best_of, int(sx), int(sy), int(gx), int(gy), a, b, M=M)
        importances.append(imp)

    d['Morris_Importance'] = importances
    return d


def main() -> None:
    print('=== apply_morris_importance.py ===')

    atp = compute_importance_for_tour('ATP', 'atp_cleaned_points.csv', ATP_P, best_of=5)
    wta = compute_importance_for_tour('WTA', 'wta_cleaned_points.csv', WTA_P, best_of=3)

    atp.to_csv(os.path.join(PROC_DIR, 'atp_morris_importance.csv'), index=False)
    wta.to_csv(os.path.join(PROC_DIR, 'wta_morris_importance.csv'), index=False)
    print('  Saved atp_morris_importance.csv, wta_morris_importance.csv')

    print('\n--- BP vs SGP leverage comparability ---')
    for tour, d in [('ATP', atp), ('WTA', wta)]:
        bp = d[d['High_Leverage_BP'] == 1]['Morris_Importance']
        sgp = d[d['High_Leverage_SGP'] == 1]['Morris_Importance']
        u, pval = stats.mannwhitneyu(bp, sgp)
        print(f'  {tour}: BP mean={bp.mean():.4f} (n={len(bp)})  '
              f'SGP mean={sgp.mean():.4f} (n={len(sgp)})  '
              f'ratio={bp.mean()/sgp.mean():.2f}x  Mann-Whitney p={pval:.3g}')

    print('=== Done ===')


if __name__ == '__main__':
    main()
