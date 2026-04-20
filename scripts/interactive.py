"""
interactive.py
--------------
Generates interactive Plotly charts for the HTML blog version.
Outputs:
  outputs/interactive_cusum.html  — CUSUM line chart with hover
  outputs/interactive_tboe.html   — TBOE scatter with player names
"""

import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, 'data', 'processed')
OUT_DIR  = os.path.join(BASE_DIR, 'outputs')

# ── Interactive CUSUM ─────────────────────────────────────────────────────────
def make_interactive_cusum() -> None:
    df = pd.read_csv(os.path.join(PROC_DIR, 'processed_features.csv'))
    wta = df[df['Tour'] == 'WTA'].copy()

    # Pick longest match
    match_counts = wta.groupby('match_id').size()
    target_match = match_counts.idxmax()
    match_data = wta[wta['match_id'] == target_match].copy().reset_index(drop=True)

    # Extract player names
    parts = target_match.split('-')
    p1 = parts[-2].replace('_', ' ') if len(parts) >= 2 else 'Player 1'
    p2 = parts[-1].replace('_', ' ') if len(parts) >= 1 else 'Player 2'

    # Hover text
    hover = []
    for idx, row in match_data.iterrows():
        hl = 'High-leverage' if row.get('High_Leverage', 0) == 1 else 'Regular'
        hover.append(
            f"Point: {int(row['Pt'])}<br>"
            f"Type: {hl}<br>"
            f"Momentum Score: {row['CUSUM']:.2f}"
        )

    fig = go.Figure()

    # Fill areas
    fig.add_trace(go.Scatter(
        x=match_data.index,
        y=match_data['CUSUM'].clip(lower=0),
        fill='tozeroy',
        fillcolor='rgba(144,238,144,0.3)',
        line=dict(color='rgba(0,0,0,0)'),
        showlegend=False,
        hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=match_data.index,
        y=match_data['CUSUM'].clip(upper=0),
        fill='tozeroy',
        fillcolor='rgba(255,182,193,0.3)',
        line=dict(color='rgba(0,0,0,0)'),
        showlegend=False,
        hoverinfo='skip'
    ))

    # Main line
    fig.add_trace(go.Scatter(
        x=match_data.index,
        y=match_data['CUSUM'],
        mode='lines',
        line=dict(color='#4682B4', width=1.5),
        name='Momentum Score',
        hovertext=hover,
        hoverinfo='text'
    ))

    # Zero line
    fig.add_hline(y=0, line_dash='dash', line_color='black', opacity=0.4)

    fig.update_layout(
        title=dict(
            text=f'<b>Cumulative Momentum Score — WTA</b><br>{p1} vs {p2}',
            font=dict(size=13, color='#111')
        ),
        xaxis_title='Point Number',
        yaxis_title='Cumulative Deviation from Mean',
        template='plotly_white',
        height=400,
        showlegend=False,
        font=dict(family='Helvetica Neue, Arial, sans-serif', size=11, color='#222'),
        paper_bgcolor='white',
        plot_bgcolor='white',
        hoverlabel=dict(bgcolor='white', font_size=12)
    )

    fig.write_html(
        os.path.join(OUT_DIR, 'interactive_cusum.html'),
        include_plotlyjs='cdn',
        full_html=True
    )
    print('  Saved interactive_cusum.html')


# ── Interactive TBOE ──────────────────────────────────────────────────────────
def make_interactive_tboe() -> None:
    df = pd.read_csv(os.path.join(PROC_DIR, 'processed_features.csv'))

    fig = make_subplots(rows=1, cols=2, subplot_titles=['ATP', 'WTA'])

    for col, (tour, color) in enumerate([('ATP', 'steelblue'), ('WTA', 'coral')], 1):
        tour_df = df[df['Tour'] == tour].copy()
        player_tboe = tour_df.groupby('Focal_Player')['TBOE'].mean().reset_index()
        player_tboe = player_tboe.sort_values('TBOE').reset_index(drop=True)
        player_tboe['rank'] = range(len(player_tboe))

        hover = [
            f"<b>{row['Focal_Player']}</b><br>TBOE: {row['TBOE']:.3f}"
            for _, row in player_tboe.iterrows()
        ]

        fig.add_trace(
            go.Scatter(
                x=player_tboe['rank'],
                y=player_tboe['TBOE'],
                mode='markers',
                marker=dict(color=color, size=8, opacity=0.7),
                hovertext=hover,
                hoverinfo='text',
                name=tour
            ),
            row=1, col=col
        )

        fig.add_hline(y=0, line_dash='dash', line_color='black',
                      opacity=0.4, row=1, col=col)

    fig.update_xaxes(title_text='Player Rank (by TBOE)')
    fig.update_yaxes(title_text='TBOE (actual minus expected win rate)')
    fig.update_layout(
        title=dict(
            text='<b>Tiebreak Over-Expectation (TBOE) by Player</b> — Hover to See Names',
            font=dict(size=13, color='#111')
        ),
        template='plotly_white',
        height=450,
        showlegend=False,
        font=dict(family='Helvetica Neue, Arial, sans-serif', size=11, color='#222'),
        paper_bgcolor='white',
        plot_bgcolor='white',
        hoverlabel=dict(bgcolor='white', font_size=12)
    )

    fig.update_annotations(font_size=12)

    fig.write_html(
        os.path.join(OUT_DIR, 'interactive_tboe.html'),
        include_plotlyjs='cdn',
        full_html=True
    )
    print('  Saved interactive_tboe.html')


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('=== interactive.py ===')
    make_interactive_cusum()
    make_interactive_tboe()
    print('=== Done ===')