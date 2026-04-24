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
from numpy.polynomial import polynomial as P

# Suppress known, safe warnings from econml and sklearn only
# Global suppression avoided — specific known warnings caught locally
warnings.filterwarnings('ignore', category=UserWarning, module='econml')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='econml')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='statsmodels')

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
np.random.seed(SEED)


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

    match_row = df[df['match_id'] == target_match].iloc[0]
    p1 = match_row['Focal_Player']
    p2 = match_row['Opponent_Player']

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(match_data.index, match_data['CUSUM'], color='#444444', linewidth=1.2)
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
    """
    Two-panel Plotly scatter — ATP and WTA separately, with player name hover.
    Separate panels because ATP and WTA players are ranked independently.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['', '']
    )

    for col, (df, label, color) in enumerate(zip(
        [atp, wta],
        ['ATP', 'WTA'],
        ['#4a90d9', '#e8715a']
    ), start=1):
        player_tboe = df.groupby('Focal_Player')['TBOE'].mean().reset_index()
        player_tboe = player_tboe.sort_values('TBOE').reset_index(drop=True)
        player_tboe['rank'] = range(len(player_tboe))

        fig.add_trace(go.Scatter(
            x=player_tboe['rank'],
            y=player_tboe['TBOE'],
            mode='markers',
            name=label,
            marker=dict(size=7, color=color, opacity=0.75,
                        line=dict(width=0)),
            hovertemplate=(
                '<b>%{customdata}</b><br>'
                'TBOE: %{y:.3f}<br>'
                '<extra>' + label + '</extra>'
            ),
            customdata=player_tboe['Focal_Player'],
            showlegend=True
        ), row=1, col=col)

        fig.add_hline(y=0, line_dash='dash', line_color='black',
                      line_width=1, opacity=0.4, row=1, col=col)

    # Manual annotations to avoid flatly theme link styling
    fig.add_annotation(
        text='ATP — Tiebreak Over-Expectation per Player',
        xref='paper', yref='paper', x=0.18, y=1.08,
        showarrow=False,
        font=dict(size=12, color='#111',
                  family='Helvetica Neue, Arial, sans-serif')
    )
    fig.add_annotation(
        text='WTA — Tiebreak Over-Expectation per Player',
        xref='paper', yref='paper', x=0.82, y=1.08,
        showarrow=False,
        font=dict(size=12, color='#111',
                  family='Helvetica Neue, Arial, sans-serif')
    )

    fig.update_layout(
        title=dict(
            text='Tiebreak Over-Expectation (TBOE) by Player',
            font=dict(size=13, family='Helvetica Neue, Arial, sans-serif',
                      color='#111'),
            x=0.5
        ),
        height=420,
        font=dict(family='Helvetica Neue, Arial, sans-serif',
                  size=11, color='#111'),
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(l=60, r=40, t=100, b=80),
        hoverlabel=dict(bgcolor='white', font_size=12),
        legend=dict(
            orientation='h', x=0.5, xanchor='center', y=-0.2,
            bgcolor='white'
        )
    )

    fig.update_xaxes(
        title_text='Player Rank (by TBOE, lowest to highest)',
        showgrid=False, zeroline=False
    )
    fig.update_yaxes(
        title_text='TBOE (actual minus expected win rate)',
        showgrid=True, gridcolor='#eeeeee', zeroline=False,
        col=1
    )
    fig.update_yaxes(
        showgrid=True, gridcolor='#eeeeee', zeroline=False,
        col=2
    )

    out_path = os.path.join(OUT_DIR, 'output3_tboe_scatter.html')
    fig.write_html(out_path, include_plotlyjs='cdn', full_html=False)
    print('  Saved output3_tboe_scatter.html')


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
    """
    Single grouped dot plot — ATP and WTA side by side per ranking band.
    Cleaner than two subplots, avoids annotation overlap issues.
    """
    import plotly.graph_objects as go

    bins   = [0, 50, 100, 200, 1000]
    labels = ['Top 50', '51–100', '101–200', '200+']

    def bin_cates(cates, X):
        ranking = X['Focal_Ranking'].values
        means, errors = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (ranking >= lo) & (ranking < hi)
            if mask.sum() < 5:
                means.append(None)
                errors.append(None)
                continue
            vals = cates[mask]
            means.append(round(float(vals.mean()), 4))
            errors.append(round(float(1.96 * vals.std() / np.sqrt(len(vals))), 4))
        return means, errors

    atp_means, atp_errors = bin_cates(atp_cates, atp_X)
    wta_means, wta_errors = bin_cates(wta_cates, wta_X)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=labels,
        y=atp_means,
        error_y=dict(type='data', array=atp_errors, visible=True,
                     color='rgba(74,144,217,0.5)', thickness=1.5, width=6),
        mode='markers',
        name='ATP',
        marker=dict(size=12, color='#4a90d9',
                    line=dict(width=1.5, color='white')),
        hovertemplate='<b>ATP — %{x}</b><br>CATE: %{y:.4f}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=labels,
        y=wta_means,
        error_y=dict(type='data', array=wta_errors, visible=True,
                     color='rgba(232,113,90,0.5)', thickness=1.5, width=6),
        mode='markers',
        name='WTA',
        marker=dict(size=12, color='#e8715a',
                    line=dict(width=1.5, color='white')),
        hovertemplate='<b>WTA — %{x}</b><br>CATE: %{y:.4f}<extra></extra>'
    ))

    fig.add_hline(y=0, line_dash='dash', line_color='red',
                  line_width=1, opacity=0.5)

    fig.update_layout(
        title=dict(
            text='Causal Effect of Winning a High-Leverage Point by Player Ranking',
            font=dict(size=13, family='Helvetica Neue, Arial, sans-serif',
                      color='#111'),
            x=0.5
        ),
        height=420,
        font=dict(family='Helvetica Neue, Arial, sans-serif', size=11,
                  color='#111'),
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(l=60, r=40, t=80, b=80),
        hoverlabel=dict(bgcolor='white', font_size=12),
        legend=dict(
            orientation='h', x=0.5, xanchor='center', y=-0.2,
            bgcolor='white'
        ),
        xaxis=dict(
            title='Player Ranking Band',
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            title='CATE (causal effect on next point)',
            showgrid=True,
            gridcolor='#eeeeee',
            zeroline=False
        )
    )

    out_path = os.path.join(OUT_DIR, 'output4_cate_plot.html')
    fig.write_html(out_path, include_plotlyjs='cdn', full_html=False)
    print('  Saved output4_cate_plot.html')


# ── Output 5: Coefficient plot ────────────────────────────────────────────────
def plot_model_table(
    atp_coef, wta_coef, atp_ate, atp_ate_se, wta_ate, wta_ate_se,
    atp_bp_ate, atp_bp_ate_se, wta_bp_ate, wta_bp_ate_se,
    atp_tb_ate, atp_tb_ate_se, wta_tb_ate, wta_tb_ate_se
) -> None:
    """
    Clean Plotly forest plot — logistic regression coefficients
    plus causal forest ATE rows for BP and TB.
    """
    import plotly.graph_objects as go

    label_map = {
        'Focal_Ranking':   'Player Rank',
        'Rolling_Win_Pct': 'Rolling Win %',
        'Streak_k4':       'Winning Streak (Last 4)',
        'CUSUM':           'Momentum Score',
    }

    atp_coef = atp_coef.copy()
    wta_coef = wta_coef.copy()
    # Exclude raw feature names before mapping
    exclude_raw = {'const', 'HL_Win', 'High_Leverage'}
    atp_coef = atp_coef[~atp_coef['Feature'].isin(exclude_raw)].copy()
    wta_coef = wta_coef[~wta_coef['Feature'].isin(exclude_raw)].copy()
    atp_coef['Feature'] = atp_coef['Feature'].replace(label_map)
    wta_coef['Feature'] = wta_coef['Feature'].replace(label_map)

    # Build rows — logistic regression coefficients
    features = atp_coef['Feature'].tolist()

    rows = []
    for feat in features:
        atp_row = atp_coef[atp_coef['Feature'] == feat]
        wta_row = wta_coef[wta_coef['Feature'] == feat]
        if len(atp_row) == 0 or len(wta_row) == 0:
            continue
        rows.append({
            'label': feat,
            'atp_coef': atp_row['Coef'].values[0],
            'atp_se':   atp_row['SE'].values[0],
            'wta_coef': wta_row['Coef'].values[0],
            'wta_se':   wta_row['SE'].values[0],
            'type': 'logistic'
        })

    # Add causal forest rows
    rows.append({
        'label': 'Break Point Win (ATP 13.5% / WTA 13.5%)',
        'atp_coef': atp_bp_ate, 'atp_se': atp_bp_ate_se,
        'wta_coef': wta_bp_ate, 'wta_se': wta_bp_ate_se,
        'type': 'forest'
    })
    rows.append({
        'label': 'Tiebreak Win (ATP 3.5% / WTA 2.3%)',
        'atp_coef': atp_tb_ate, 'atp_se': atp_tb_ate_se,
        'wta_coef': wta_tb_ate, 'wta_se': wta_tb_ate_se,
        'type': 'forest'
    })

    # Y axis — reversed so top row appears first
    y_labels = [r['label'] for r in rows][::-1]

    fig = go.Figure()

    for i, row in enumerate(rows[::-1]):
        symbol = 'diamond' if row['type'] == 'forest' else 'circle'
        y_pos = i

        # ATP
        fig.add_trace(go.Scatter(
            x=[row['atp_coef']],
            y=[y_pos - 0.15],
            error_x=dict(
                type='data',
                array=[1.96 * row['atp_se']],
                visible=True,
                color='rgba(74,144,217,0.6)',
                thickness=1.5,
                width=5
            ),
            mode='markers',
            marker=dict(
                symbol=symbol,
                size=10,
                color='#4a90d9',
                line=dict(width=1.5, color='white')
            ),
            name='ATP',
            showlegend=(i == 0),
            hovertemplate=f"<b>ATP — {row['label']}</b><br>"
                          f"Effect: {row['atp_coef']:.4f}<br>"
                          f"95% CI: [{row['atp_coef']-1.96*row['atp_se']:.4f}, "
                          f"{row['atp_coef']+1.96*row['atp_se']:.4f}]"
                          f"<extra></extra>",
            legendgroup='ATP'
        ))

        # WTA
        fig.add_trace(go.Scatter(
            x=[row['wta_coef']],
            y=[y_pos + 0.15],
            error_x=dict(
                type='data',
                array=[1.96 * row['wta_se']],
                visible=True,
                color='rgba(232,113,90,0.6)',
                thickness=1.5,
                width=5
            ),
            mode='markers',
            marker=dict(
                symbol=symbol,
                size=10,
                color='#e8715a',
                line=dict(width=1.5, color='white')
            ),
            name='WTA',
            showlegend=(i == 0),
            hovertemplate=f"<b>WTA — {row['label']}</b><br>"
                          f"Effect: {row['wta_coef']:.4f}<br>"
                          f"95% CI: [{row['wta_coef']-1.96*row['wta_se']:.4f}, "
                          f"{row['wta_coef']+1.96*row['wta_se']:.4f}]"
                          f"<extra></extra>",
            legendgroup='WTA'
        ))

    fig.add_vline(x=0, line_dash='dash', line_color='black',
                  line_width=1, opacity=0.5)

    fig.update_layout(
        title=dict(
            text='Model Results — Logistic Regression + Causal Forest',
            font=dict(size=14, family='Helvetica Neue, Arial, sans-serif',
                      color='#111'),
            x=0.5
        ),
        height=480,
        font=dict(family='Helvetica Neue, Arial, sans-serif', size=11,
                  color='#111'),
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(l=260, r=40, t=80, b=80),
        hoverlabel=dict(bgcolor='white', font_size=12),
        legend=dict(
            orientation='v',
            x=1.02, y=1,
            bgcolor='white',
            bordercolor='#eeeeee',
            borderwidth=1
        ),
        xaxis=dict(
            title=dict(text='Effect on Win Probability', font=dict(color='#111')),
            showgrid=True,
            gridcolor='#eeeeee',
            zeroline=False
        ),
        yaxis=dict(
            tickvals=list(range(len(rows))),
            ticktext=y_labels,
            showgrid=False,
            zeroline=False,
            tickfont=dict(color='#111')
        )
    )

    out_path = os.path.join(OUT_DIR, 'output5_model_table.html')
    fig.write_html(out_path, include_plotlyjs='cdn', full_html=False)
    print('  Saved output5_model_table.html')


# ── Output 7: Feature importance ─────────────────────────────────────────────
def plot_feature_importance(atp_imp: pd.DataFrame,
                            wta_imp: pd.DataFrame) -> None:
    """
    Interactive Plotly horizontal bar chart — feature importance.
    Single chart with ATP and WTA side by side per feature.
    """
    import plotly.graph_objects as go

    label_map = {
        'Focal_Ranking':   'Player Rank',
        'Rolling_Win_Pct': 'Rolling Win %',
        'Streak_k4':       'Winning Streak (Last 4)',
        'CUSUM':           'Momentum Score',
    }

    atp_imp = atp_imp.copy()
    wta_imp = wta_imp.copy()
    atp_imp['Feature'] = atp_imp['Feature'].replace(label_map)
    wta_imp['Feature'] = wta_imp['Feature'].replace(label_map)

    # Sort by ATP importance descending
    atp_sorted = atp_imp.sort_values('Importance', ascending=True)
    features = atp_sorted['Feature'].tolist()
    wta_sorted = wta_imp.set_index('Feature').loc[features].reset_index()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=features,
        x=atp_sorted['Importance'],
        name='ATP',
        orientation='h',
        marker=dict(color='#4a90d9', opacity=0.85),
        hovertemplate='<b>ATP — %{y}</b><br>Importance: %{x:.4f}<extra></extra>'
    ))

    fig.add_trace(go.Bar(
        y=features,
        x=wta_sorted['Importance'],
        name='WTA',
        orientation='h',
        marker=dict(color='#e8715a', opacity=0.85),
        hovertemplate='<b>WTA — %{y}</b><br>Importance: %{x:.4f}<extra></extra>'
    ))

    fig.update_layout(
        title=dict(
            text='Feature Importance from Causal Forest',
            font=dict(size=13, family='Helvetica Neue, Arial, sans-serif',
                      color='#111', weight='bold'),
            x=0.5
        ),
        barmode='group',
        height=380,
        font=dict(family='Helvetica Neue, Arial, sans-serif',
                  size=11, color='#111'),
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(l=160, r=40, t=80, b=80),
        hoverlabel=dict(bgcolor='white', font_size=12),
        legend=dict(
            orientation='h', x=0.5, xanchor='center', y=-0.2,
            bgcolor='white'
        ),
        xaxis=dict(
            title='Importance (higher = stronger driver of heterogeneity)',
            showgrid=True, gridcolor='#eeeeee', zeroline=False
        ),
        yaxis=dict(showgrid=False, zeroline=False)
    )

    out_path = os.path.join(OUT_DIR, 'output7_feature_importance.html')
    fig.write_html(out_path, include_plotlyjs='cdn', full_html=False)
    print('  Saved output7_feature_importance.html')


# ── Reveal chart ─────────────────────────────────────────────────────────────
def plot_reveal_chart(
    atp_ate, wta_ate,
    atp_bp_ate, wta_bp_ate,
    atp_tb_ate, wta_tb_ate
) -> None:
    """
    Interactive reveal chart — Plotly replacement for D3 reveal chart.
    Toggles between combined ATE view and split BP/TB view.
    """
    import plotly.graph_objects as go

    # Colours
    NEG = '#e05252'
    POS = '#4caf7d'

    def bar_color(val):
        return POS if val >= 0 else NEG

    # Combined view data
    combined_x = ['ATP', 'WTA']
    combined_y = [atp_ate, wta_ate]
    combined_colors = [bar_color(v) for v in combined_y]

    # Split view data
    split_x = ['ATP Break Point', 'WTA Break Point',
                'ATP Tiebreak', 'WTA Tiebreak']
    split_y = [atp_bp_ate, wta_bp_ate, atp_tb_ate, wta_tb_ate]
    split_colors = [bar_color(v) for v in split_y]

    fig = go.Figure()

    # Trace 0 — combined (visible by default)
    fig.add_trace(go.Bar(
        x=combined_x,
        y=combined_y,
        marker_color=combined_colors,
        marker_line=dict(width=0),
        text=[f'{v:+.4f}' for v in combined_y],
        textposition='outside',
        textfont=dict(size=12, family='Helvetica Neue, Arial, sans-serif'),
        hovertemplate='<b>%{x}</b><br>ATE: %{y:.4f}<extra></extra>',
        visible=True,
        name='Combined'
    ))

    # Trace 1 — split (hidden by default)
    fig.add_trace(go.Bar(
        x=split_x,
        y=split_y,
        marker_color=split_colors,
        marker_line=dict(width=0),
        text=[f'{v:+.4f}' for v in split_y],
        textposition='outside',
        textfont=dict(size=12, family='Helvetica Neue, Arial, sans-serif'),
        hovertemplate='<b>%{x}</b><br>ATE: %{y:.4f}<extra></extra>',
        visible=False,
        name='Split'
    ))

    fig.add_hline(y=0, line_color='black', line_width=1, opacity=0.4)

    fig.update_layout(
        title=dict(
            text='Average Treatment Effect of Winning a High-Leverage Point',
            font=dict(size=13, family='Helvetica Neue, Arial, sans-serif',
                      color='#111'),
            x=0.5
        ),
        height=420,
        font=dict(family='Helvetica Neue, Arial, sans-serif',
                  size=11, color='#111'),
        paper_bgcolor='white',
        plot_bgcolor='white',
        showlegend=False,
        margin=dict(l=60, r=40, t=120, b=80),
        hoverlabel=dict(bgcolor='white', font_size=12),
        yaxis=dict(
            title='Effect on probability of winning next point',
            showgrid=True,
            gridcolor='#eeeeee',
            zeroline=False,
            range=[-0.12, 0.10]
        ),
        xaxis=dict(showgrid=False, zeroline=False),
        updatemenus=[dict(
            type='buttons',
            direction='left',
            x=0.5,
            xanchor='center',
            y=1.18,
            yanchor='top',
            buttons=[
                dict(
                    label='Combined Effect',
                    method='update',
                    args=[{'visible': [True, False]}]
                ),
                dict(
                    label='Split by Pressure Type',
                    method='update',
                    args=[{'visible': [False, True]}]
                )
            ],
            bgcolor='white',
            bordercolor='#cccccc',
            borderwidth=1,
            font=dict(size=12,
                      family='Helvetica Neue, Arial, sans-serif',
                      color='#111'),
            pad=dict(r=10, t=5, b=5, l=10),
            showactive=True,
            active=0
        )]
    )

    out_path = os.path.join(OUT_DIR, 'output6_reveal_chart.html')
    fig.write_html(out_path, include_plotlyjs='cdn', full_html=False)
    print('  Saved output6_reveal_chart.html')


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
    _, atp_bp_ate, atp_bp_ate_se, _, _ = run_causal_forest(
        atp, 'ATP', treatment_label='bp')
    _, wta_bp_ate, wta_bp_ate_se, _, _ = run_causal_forest(
        wta, 'WTA', treatment_label='bp')
    _, atp_tb_ate, atp_tb_ate_se, _, _ = run_causal_forest(
        atp, 'ATP', treatment_label='tb')
    _, wta_tb_ate, wta_tb_ate_se, _, _ = run_causal_forest(
        wta, 'WTA', treatment_label='tb')
    print(f'\n  ATE by leverage type:')
    print(
        f'    ATP — Break Point: {atp_bp_ate:.4f} | Tiebreak: {atp_tb_ate:.4f}')
    print(
        f'    WTA — Break Point: {wta_bp_ate:.4f} | Tiebreak: {wta_tb_ate:.4f}')

    print('\n--- Reveal chart ---')
    plot_reveal_chart(
        atp_ate, wta_ate,
        atp_bp_ate, wta_bp_ate,
        atp_tb_ate, wta_tb_ate
    )

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
    plot_model_table(
        atp_coef, wta_coef,
        atp_ate, atp_ate_se,
        wta_ate, wta_ate_se,
        atp_bp_ate, atp_bp_ate_se,
        wta_bp_ate, wta_bp_ate_se,
        atp_tb_ate, atp_tb_ate_se,
        wta_tb_ate, wta_tb_ate_se
    )

    # Output 6
    print('\n--- Output 6: Feature importance ---')
    plot_feature_importance(atp_imp, wta_imp)

    # Export ATE results as CSV — provides a Python-generated data file
    # backing the D3 reveal chart values displayed in the blog
    ate_summary = pd.DataFrame([
        {'Tour': 'ATP', 'Type': 'Combined',    'ATE': round(atp_ate, 4),    'SE': round(atp_ate_se, 4)},  # noqa: E501
        {'Tour': 'WTA', 'Type': 'Combined',    'ATE': round(wta_ate, 4),    'SE': round(wta_ate_se, 4)},  # noqa: E501
        {'Tour': 'ATP', 'Type': 'Break Point', 'ATE': round(atp_bp_ate, 4), 'SE': round(atp_bp_ate_se, 4)},  # noqa: E501
        {'Tour': 'WTA', 'Type': 'Break Point', 'ATE': round(wta_bp_ate, 4), 'SE': round(wta_bp_ate_se, 4)},  # noqa: E501
        {'Tour': 'ATP', 'Type': 'Tiebreak',   'ATE': round(atp_tb_ate, 4), 'SE': round(atp_tb_ate_se, 4)},  # noqa: E501
        {'Tour': 'WTA', 'Type': 'Tiebreak',   'ATE': round(wta_tb_ate, 4), 'SE': round(wta_tb_ate_se, 4)},  # noqa: E501
    ])
    ate_summary.to_csv(os.path.join(OUT_DIR, 'ate_results.csv'), index=False)
    print('  Saved ate_results.csv')

    # Export feature importance for blog.js dynamic rendering
    _label_map = {
        'CUSUM':           'Cumulative Momentum Score',
        'Focal_Ranking':   'Player Ranking',
        'Rolling_Win_Pct': 'Rolling Win % (last 10)',
        'Streak_k4':       'Winning Streak (last 4)',
    }
    feat_imp_export = pd.DataFrame({
        'feature': [_label_map[f] for f in CONTROLS],
        'ATP': [round(float(atp_imp.loc[atp_imp['Feature'] == f, 'Importance'].values[0]), 4)
                for f in CONTROLS],
        'WTA': [round(float(wta_imp.loc[wta_imp['Feature'] == f, 'Importance'].values[0]), 4)
                for f in CONTROLS]
    })
    feat_imp_export.to_csv(os.path.join(OUT_DIR, 'feature_importance.csv'), index=False)
    print('  Saved feature_importance.csv')

    print('\n=== Done — all 6 outputs saved to outputs/ ===')


if __name__ == '__main__':
    main()
