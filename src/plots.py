"""Interactive Plotly visualizations."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.analysis import MetaResult, calculate_effects

BLUE = "#3278b7"
ORANGE = "#d78a32"
RED = "#b95d59"
GREEN = "#168a7b"
GREY = "#8a96a4"


def forest_plot(frame: pd.DataFrame, fit: MetaResult, title: str = "Forest Plot",
                highlight_phase: int | None = None) -> go.Figure:
    df = calculate_effects(frame)
    n = len(df)
    ys = np.arange(n, 0, -1)
    fig = go.Figure()
    for i, row in df.iterrows():
        highlight = highlight_phase is not None and "phase" in df and row["phase"] == highlight_phase
        color = ORANGE if highlight else BLUE
        weight = float(fit.weights_percent[i])
        fig.add_trace(go.Scatter(
            x=[row["yi"]], y=[ys[i]], mode="markers",
            marker=dict(symbol="square", size=8 + 1.0 * np.sqrt(weight * 10), color=color,
                        line=dict(color="#ffffff", width=1)),
            error_x=dict(type="data", symmetric=False,
                         array=[row["ci_high"] - row["yi"]],
                         arrayminus=[row["yi"] - row["ci_low"]],
                         thickness=1.5, width=4, color=color),
            hovertemplate=(f"<b>{row['author_year']}</b><br>MD: {row['yi']:.2f}"
                           f"<br>95%-KI: [{row['ci_low']:.2f}; {row['ci_high']:.2f}]"
                           f"<br>RE-Gewicht: {weight:.1f}%<extra></extra>"),
            showlegend=False,
        ))
    diamond_y = 0.0
    fig.add_trace(go.Scatter(
        x=[fit.ci_low, fit.mean, fit.ci_high, fit.mean, fit.ci_low],
        y=[diamond_y, diamond_y + .27, diamond_y, diamond_y - .27, diamond_y],
        fill="toself", fillcolor=GREEN, line=dict(color=GREEN, width=1),
        name="Gepoolter Effekt", hovertemplate=(f"MD {fit.mean:.2f}<br>95%-KI "
        f"[{fit.ci_low:.2f}; {fit.ci_high:.2f}]<extra></extra>")))
    fig.add_vline(x=0, line_dash="dash", line_color=GREY)
    # Study labels include estimated RE weight for transparency.
    labels = [f"{row.author_year}   ({fit.weights_percent[i]:.1f} %)"
              for i, row in enumerate(df.itertuples())] + ["Gesamt (REML)"]
    fig.update_layout(
        title=title, template="plotly_white",
        height=max(380, 160 + 36 * n), margin=dict(l=12, r=20, t=55, b=38),
        xaxis_title="Mittelwertdifferenz (ISI-Punkte) · negativ = Intervention besser",
        yaxis=dict(tickmode="array", tickvals=list(ys) + [0], ticktext=labels,
                   range=[-0.65, n + 1.1], zeroline=False, automargin=True),
        showlegend=False, hovermode="closest",
    )
    return fig


def history_plot(history: pd.DataFrame, selected: str, threshold: float) -> go.Figure:
    fig = go.Figure()
    for row in history.itertuples():
        color = ORANGE if row.version == selected else BLUE
        fig.add_trace(go.Scatter(
            x=[row.mean], y=[row.version], mode="markers",
            marker=dict(color=color, size=12 if row.version == selected else 9),
            error_x=dict(type="data", symmetric=False,
                         array=[row.ci_high - row.mean],
                         arrayminus=[row.mean - row.ci_low], thickness=2.2, width=5,
                         color=color), showlegend=False,
            hovertemplate=(f"{row.version}: MD {row.mean:.2f}<br>95%-KI "
                           f"[{row.ci_low:.2f}; {row.ci_high:.2f}]<extra></extra>")))
    fig.add_vline(x=0, line_color=GREY, line_dash="dash", annotation_text="Nulleffekt")
    fig.add_vline(x=threshold, line_color=ORANGE, line_dash="dot",
                  annotation_text="Illustrative Relevanzschwelle")
    fig.update_layout(template="plotly_white", height=345,
                      margin=dict(l=10,r=20,t=35,b=40),
                      xaxis_title="Gepoolte MD und 95%-KI", yaxis_title=None,
                      yaxis=dict(categoryorder="array", categoryarray=["V4", "V3", "V2", "V1", "V0"]))
    return fig


def posterior_plot(density: pd.DataFrame, threshold: float, ci_low: float, ci_high: float):
    fig = go.Figure()
    x = density["MD"].to_numpy()
    y = density["Posterior-Dichte"].to_numpy()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=BLUE, width=3),
                             name="Posterior", fill="tozeroy", fillcolor="rgba(50,120,183,.14)"))
    fig.add_vline(x=0, line_color=GREY, line_dash="dash", annotation_text="0")
    fig.add_vline(x=threshold, line_color=ORANGE, line_dash="dot", annotation_text="Schwelle")
    fig.add_vrect(x0=ci_low, x1=ci_high, fillcolor="rgba(22,138,123,.11)", line_width=0,
                  annotation_text="95%-KI (Bayes)", annotation_position="top left")
    fig.update_layout(template="plotly_white", height=355,
                      xaxis_title="Gesamteffekt μ (ISI-Punkte)",
                      yaxis_title="Posterior-Dichte", margin=dict(l=8,r=10,t=40,b=40))
    return fig
