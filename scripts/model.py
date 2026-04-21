"""
model.py
--------
Loads processed_features.csv checkpoint.
Runs logistic regression and Causal Forest separately for ATP and WTA.
Generates all 6 outputs to outputs/.

Standalone — does not depend on clean.py or features.py.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
from econml.dml import CausalForestDML
from sklearn.ensemble import GradientBoostingRegressor

warnings.filterwarnings('ignore')

rng = np.random.default_rng(seed=42)

plt.rcParams.update({
    'font.family':      'sans-serif',
    'font.size':        11,
    'axes.spines.top':  False,
    'axes.spines.right': False,
    'figure.dpi':       150,
    'savefig.dpi':      150,
    'savefig.bbox':     'tight',
})

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR  = os.path.join(BASE_DIR, 'data', 'processed')
OUT_DIR   = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
CONTROLS  = ['Focal_Ranking', 'Rolling_Win_Pct', 'Streak_k4', 'CUSUM']
TREATMENT = 'High_Leverage'
OUTCOME   = 'Next_Point_Won'
ALPHA     = 0.05
SEED      = 42


# ── Output 1: Per-player Chi-squared summary table ────────────────────────────
def plot_chi2_table(tests_path: str) -> None:
    tests = pd.read_csv(tests_path)

    summary = tests.groupby('Tour').agg(
        Players_Tested=('Player', 'count'),
        Chi2_Significant=('Chi2_Sig', 'sum'),
        Runs_Significant=('Runs_Sig', 'sum'),
    ).reset_index()
    summary['Chi2_Pct'] = (summary['Chi2_Significant']
                           / summary['Players_Tested'] * 100).round(1)
    summary['Runs_Pct'] = (summary['Runs_Significant']
                           / summary['Players_Tested'] * 100).round(1)

    fig, ax = plt.subplots(figsize=(10, 2))
    ax.axis('off')
    table_data = [
        ['Tour', 'Players\nTested',
            'Chi² Sig.\n(n)', 'Chi² Sig.\n(%)', 'Runs Sig.\n(n)', 'Runs Sig.\n(%)'],
    ]
    for _, row in summary.iterrows():
        table_data.append([
            row['Tour'],
            int(row['Players_Tested']),
            int(row['Chi2_Significant']),
            f"{row['Chi2_Pct']}%",
            int(row['Runs_Significant']),
            f"{row['Runs_Pct']}%",
        ])

    t = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                 loc='center', cellLoc='center')
    t.auto_set_font_size(False)
    t.set_fontsize(11)
    t.scale(1.4, 2.2)

    ax.set_title('Per-Player Momentum Test Results (p < 0.05)',
                 fontsize=12, fontweight='bold', pad=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'output1_chi2_table.png'))
    plt.close()
    print('  Saved output1_chi2_table.png')


# ── Output 2: CUSUM line chart ────────────────────────────────────────────────
def plot_cusum(df: pd.DataFrame, tour_name: str) -> None:
    # Pick the match with the most points for a clear chart
    match_counts = df.groupby('match_id').size()
    target_match = match_counts.idxmax()
    match_data = df[df['match_id']
                    == target_match].copy().reset_index(drop=True)

    # Extract player names from match_id
    parts = target_match.split('-')
    p1 = parts[-2].replace('_', ' ') if len(parts) >= 2 else 'Player 1'
    p2 = parts[-1].replace('_', ' ') if len(parts) >= 1 else 'Player 2'

    fig, ax = plt.subplots(figsize=(10, 4))
    color = 'steelblue' if tour_name == 'ATP' else 'coral'
    ax.plot(match_data.index, match_data['CUSUM'], color=color, linewidth=1.5)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.fill_between(
        match_data.index, match_data['CUSUM'], 0,
        where=match_data['CUSUM'] > 0,
        alpha=0.3, color='green', label='Above expectation'
    )
    ax.fill_between(
        match_data.index, match_data['CUSUM'], 0,
        where=match_data['CUSUM'] < 0,
        alpha=0.3, color='red', label='Below expectation'
    )

    ax.set_title(f'Cumulative Momentum Score — {tour_name}\n{p1} vs {p2}',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Point Number')
    ax.set_ylabel('Cumulative Deviation from Mean')
    ax.legend(fontsize=9)
    plt.tight_layout()
    fname = f'output2_cusum_{tour_name.lower()}.png'
    plt.savefig(os.path.join(OUT_DIR, fname))
    plt.close()
    print(f'  Saved {fname}')


# ── Output 3: TBOE scatter plot ───────────────────────────────────────────────
def plot_tboe(atp: pd.DataFrame, wta: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, df, label, color in zip(
        axes, [atp, wta], ['ATP', 'WTA'], ['steelblue', 'coral']
    ):
        player_tboe = df.groupby('Focal_Player')['TBOE'].mean().reset_index()
        player_tboe = player_tboe.sort_values('TBOE')
        player_tboe['rank'] = range(len(player_tboe))

        ax.scatter(player_tboe['rank'], player_tboe['TBOE'],
                   color=color, alpha=0.7, s=40, edgecolors='none')
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        ax.set_title(f'{label} — Tiebreak Over-Expectation per Player',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Player Rank (by TBOE)')
        ax.set_ylabel('TBOE (actual minus expected win rate)')
    fig.suptitle('Tiebreak Over-Expectation (TBOE) by Player',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'output3_tboe_scatter.png'))
    plt.close()
    print('  Saved output3_tboe_scatter.png')


# ── Logistic Regression ───────────────────────────────────────────────────────
def run_logistic(df: pd.DataFrame, tour_name: str) -> pd.DataFrame:
    print(f'\n  Logistic regression — {tour_name}')
    keep = CONTROLS + [TREATMENT, OUTCOME, 'Point_Won', 'match_id']
    data = df[keep].dropna().copy()
    data['HL_Win'] = ((data['High_Leverage'] == 1) & (
        data['Point_Won'] == 1)).astype(float)

    X = sm.add_constant(data[CONTROLS + ['HL_Win']].astype(float))
    y = data[OUTCOME].astype(float)

    model = sm.Logit(y, X)
    result = model.fit(
        disp=0,
        method='bfgs',
        cov_type='cluster',
        cov_kwds={'groups': data['match_id']}
    )

    # Compute marginal effects at the mean
    margins = result.get_margeff()
    coef_df = pd.DataFrame({
        'Feature':  [v for v in margins.summary_frame().index],
        'Coef':     margins.margeff,
        'SE':       margins.margeff_se,
        'P_value':  margins.pvalues,
    })
    coef_df['Tour'] = tour_name
    print(f'    N = {len(data):,} | Log-likelihood = {result.llf:.1f}')
    return coef_df


# ── Causal Forest ─────────────────────────────────────────────────────────────
def run_causal_forest(df: pd.DataFrame, tour_name: str,
                      treatment_label: str = 'combined',
                      controls: list = None) -> tuple:
    if controls is None:
        controls = CONTROLS
    print(f'\n  Causal Forest — {tour_name} [{treatment_label}]')

    extra_cols = ['Point_Won']
    if treatment_label == 'bp':
        extra_cols.append('High_Leverage_BP')
    elif treatment_label == 'tb':
        extra_cols.append('High_Leverage_TB')

    keep = controls + [TREATMENT, OUTCOME] + extra_cols
    data = df[keep].dropna().copy()

    if treatment_label == 'bp':
        data['Treatment'] = ((data['High_Leverage_BP'] == 1) & (
            data['Point_Won'] == 1)).astype(float)
    elif treatment_label == 'tb':
        data['Treatment'] = ((data['High_Leverage_TB'] == 1) & (
            data['Point_Won'] == 1)).astype(float)
    else:
        data['Treatment'] = ((data['High_Leverage'] == 1) & (
            data['Point_Won'] == 1)).astype(float)

    # Stratified subsampling — preserve treatment balance
    if len(data) > 15000:
        treated = data[data['Treatment'] == 1]
        control = data[data['Treatment'] == 0].sample(
            min(len(data) - len(treated), 15000 - len(treated)),
            random_state=SEED
        )
        data = pd.concat([treated, control])
        print(
            f'    Stratified sample: {len(treated):,} treated,'
            f' {len(control):,} control'
        )

    T = data['Treatment'].values
    Y = data[OUTCOME].astype(float).values
    X = data[controls].astype(float).values

    # Fix 6: Treatment overlap check
    print(f'    Treatment rate: {T.mean()*100:.1f}%')
    print(f'    Treated N: {int(T.sum()):,} | Control N: {int((1-T).sum()):,}')

    cf = CausalForestDML(
        model_y=GradientBoostingRegressor(n_estimators=200, random_state=SEED),
        model_t=GradientBoostingRegressor(n_estimators=200, random_state=SEED),
        n_estimators=200,
        cv=2,
        n_jobs=1,
        random_state=SEED,
        verbose=0
    )
    cf.fit(Y, T, X=X)

    cates = cf.effect(X)
    ate   = float(cf.ate(X))
    ate_se = cates.std() / np.sqrt(len(cates))

    print(f'    N = {len(data):,} | ATE = {ate:.4f} (SE = {ate_se:.4f})')

    importances = cf.feature_importances_
    feat_imp = pd.DataFrame({
        'Feature':    controls,
        'Importance': importances,
        'Tour':       tour_name
    }).sort_values('Importance', ascending=False)

    return cates, ate, ate_se, feat_imp, data[controls].astype(float)


# ── Output 4: CATE plot ───────────────────────────────────────────────────────
def plot_cate(atp_cates, atp_X, wta_cates, wta_X) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, cates, X, label, color in zip(
        axes,
        [atp_cates, wta_cates],
        [atp_X, wta_X],
        ['ATP', 'WTA'],
        ['steelblue', 'coral']
    ):
        ranking = X['Focal_Ranking'].values
        ax.scatter(ranking, cates, alpha=0.3, s=15,
                   color=color, edgecolors='none')

        # Polynomial trend line
        from numpy.polynomial import polynomial as P
        sort_idx = np.argsort(ranking)
        sorted_ranking = ranking[sort_idx]
        sorted_cates = cates[sort_idx]
        coeffs = P.polyfit(sorted_ranking, sorted_cates, 3)
        smoothed = P.polyval(sorted_ranking, coeffs)
        ax.plot(sorted_ranking, smoothed, color='black',
                linewidth=2, label='Trend')

        ax.axhline(0, color='red', linewidth=0.8, linestyle='--', alpha=0.7)
        ax.set_title(f'{label} — Causal Effect by Player Ranking',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel(
            'Player Ranking (most Grand Slam players ranked below 200)', fontsize=11)
        ax.set_ylabel('CATE (causal effect on next point)', fontsize=11)
        ax.tick_params(labelsize=10)
        ax.legend(fontsize=9)

    fig.suptitle('Heterogeneous Causal Effects (CATE) of High-Leverage Points',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'output4_cate_plot.png'))
    plt.close()
    print('  Saved output4_cate_plot.png')


# ── Output 5: Coefficient plot ────────────────────────────────────────────────
def plot_model_table(
    atp_coef, wta_coef, atp_ate, atp_ate_se, wta_ate, wta_ate_se
) -> None:
    label_map = {
        'Focal_Ranking':    'Player Rank',
        'Rolling_Win_Pct':  'Rolling Win %',
        'Streak_k4':        'Winning Streak (Last 4 Points)',
        'CUSUM':            'Momentum Score',
        'High_Leverage':    'Pressure Point',
        'HL_Win':           'Break Point Win (ATP 23.5% / WTA 25.7%)',
    }
    atp_coef = atp_coef.copy()
    wta_coef = wta_coef.copy()
    atp_coef['Feature'] = atp_coef['Feature'].replace(label_map)
    wta_coef['Feature'] = wta_coef['Feature'].replace(label_map)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'ATP': 'steelblue', 'WTA': 'coral'}
    offsets = {'ATP': -0.15, 'WTA': 0.15}
    y_labels = []
    y_positions = []

    all_features = atp_coef[atp_coef['Feature'] != 'const']['Feature'].tolist()

    for i, feature in enumerate(all_features):
        y_labels.append(feature)
        y_positions.append(i)
        for tour, coef_df in [('ATP', atp_coef), ('WTA', wta_coef)]:
            row = coef_df[coef_df['Feature'] == feature]
            if len(row) == 0:
                continue
            coef = row['Coef'].values[0]
            se   = row['SE'].values[0]
            y    = i + offsets[tour]
            ax.errorbar(coef, y, xerr=1.96*se,
                        fmt='o', color=colors[tour],
                        capsize=4, capthick=1.5,
                        markersize=7, linewidth=1.5,
                        label=tour if i == 0 else '')

    # Add CATE rows
    cate_y = len(all_features)
    y_labels.append('Tiebreak Win (ATP 3.5% / WTA 2.3%)')
    y_positions.append(cate_y)
    ax.errorbar(atp_ate, cate_y + offsets['ATP'],
                xerr=1.96*atp_ate_se, fmt='D',
                color='steelblue', capsize=4, capthick=1.5,
                markersize=8, linewidth=1.5)
    ax.errorbar(wta_ate, cate_y + offsets['WTA'],
                xerr=1.96*wta_ate_se, fmt='D',
                color='coral', capsize=4, capthick=1.5,
                markersize=8, linewidth=1.5)

    ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.6)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=10)
    ax.set_xlabel('Marginal Effect on Win Probability', fontsize=11)
    ax.set_title('Model Results — Logistic Regression + Causal Forest',
                 fontsize=12, fontweight='bold', pad=15)

    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=c, markersize=9, label=t)
               for t, c in colors.items()]
    ax.legend(handles=handles, fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'output5_model_table.png'))
    plt.close()
    print('  Saved output5_model_table.png')


# ── Output 6: Feature importance ─────────────────────────────────────────────
def plot_feature_importance(atp_imp: pd.DataFrame, wta_imp: pd.DataFrame) -> None:
    label_map = {
        'Focal_Ranking':    'Player Rank',
        'Rolling_Win_Pct':  'Rolling Win %',
        'Streak_k4':        'Winning Streak (Last 4 Points)',
        'CUSUM':            'Momentum Score',
    }
    atp_imp = atp_imp.copy()
    wta_imp = wta_imp.copy()
    atp_imp['Feature'] = atp_imp['Feature'].replace(label_map)
    wta_imp['Feature'] = wta_imp['Feature'].replace(label_map)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, imp, label, color in zip(
        axes,
        [atp_imp, wta_imp],
        ['ATP', 'WTA'],
        ['steelblue', 'coral']
    ):
        imp_sorted = imp.sort_values('Importance').copy()
        ax.barh(imp_sorted['Feature'],
                imp_sorted['Importance'], color=color, alpha=0.8)
        ax.set_title(f'{label} — Feature Importance (Causal Forest)',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Importance', fontsize=11)
        ax.tick_params(labelsize=10)

    fig.suptitle('Feature Importance from Causal Forest',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'output6_feature_importance.png'))
    plt.close()
    print('  Saved output6_feature_importance.png')


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print('=== model.py ===')

    # Load checkpoint
    df = pd.read_csv(os.path.join(
        PROC_DIR, 'processed_features.csv'), low_memory=False)
    print(f'Loaded processed_features.csv: {len(df):,} rows')

    atp = df[df['Tour'] == 'ATP'].copy()
    wta = df[df['Tour'] == 'WTA'].copy()
    print(f'ATP: {len(atp):,} rows | WTA: {len(wta):,} rows')

    # Output 1
    print('\n--- Output 1: Chi-squared table ---')
    plot_chi2_table(os.path.join(PROC_DIR, 'per_player_tests.csv'))

    # Output 2
    print('\n--- Output 2: CUSUM line charts ---')
    plot_cusum(atp, 'ATP')
    plot_cusum(wta, 'WTA')

    # Output 3
    print('\n--- Output 3: TBOE scatter ---')
    plot_tboe(atp, wta)

    # Logistic regression
    print('\n--- Logistic Regression ---')
    atp_coef = run_logistic(atp, 'ATP')
    wta_coef = run_logistic(wta, 'WTA')

    # Causal Forest — combined treatment, full controls
    print('\n--- Causal Forest ---')
    atp_cates, atp_ate, atp_ate_se, atp_imp, atp_X = run_causal_forest(
        atp, 'ATP')
    wta_cates, wta_ate, wta_ate_se, wta_imp, wta_X = run_causal_forest(
        wta, 'WTA')

    # Fix 4: BP vs TB breakdown
    _, atp_bp_ate, _, _, _ = run_causal_forest(
        atp, 'ATP', treatment_label='bp')
    _, wta_bp_ate, _, _, _ = run_causal_forest(
        wta, 'WTA', treatment_label='bp')
    _, atp_tb_ate, _, _, _ = run_causal_forest(
        atp, 'ATP', treatment_label='tb')
    _, wta_tb_ate, _, _, _ = run_causal_forest(
        wta, 'WTA', treatment_label='tb')
    print(f'\n  ATE by leverage type:')
    print(
        f'    ATP — Break Point: {atp_bp_ate:.4f} | Tiebreak: {atp_tb_ate:.4f}')
    print(
        f'    WTA — Break Point: {wta_bp_ate:.4f} | Tiebreak: {wta_tb_ate:.4f}')

    # Fix 5: Robustness — reduced controls (ranking only)
    _, atp_ate_r, _, _, _ = run_causal_forest(
        atp, 'ATP', controls=['Focal_Ranking'])
    _, wta_ate_r, _, _, _ = run_causal_forest(
        wta, 'WTA', controls=['Focal_Ranking'])
    print(f'\n  Robustness (ranking-only controls):')
    print(f'    ATP — Full: {atp_ate:.4f} | Reduced: {atp_ate_r:.4f}')
    print(f'    WTA — Full: {wta_ate:.4f} | Reduced: {wta_ate_r:.4f}')

    # Output 4
    print('\n--- Output 4: CATE plot ---')
    plot_cate(atp_cates, atp_X, wta_cates, wta_X)

    # Output 5
    print('\n--- Output 5: Model table ---')
    plot_model_table(atp_coef, wta_coef, atp_ate,
                     atp_ate_se, wta_ate, wta_ate_se)

    # Output 6
    print('\n--- Output 6: Feature importance ---')
    plot_feature_importance(atp_imp, wta_imp)

    print('\n=== Done — all 6 outputs saved to outputs/ ===')


if __name__ == '__main__':
    main()
