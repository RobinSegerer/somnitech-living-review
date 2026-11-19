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


def forest_plot(frame: pd.DataFrame, fit: MetaResult, title: str = "Forest plot",
                highlight_phase: int | None = None,
                highlight_ids: set[str] | None = None,
                previous_effects: pd.DataFrame | None = None,
                threshold: float | None = None,
                x_range: tuple[float, float] | None = None) -> go.Figure:
    df = calculate_effects(frame)
    n = len(df)
    ys = np.arange(n, 0, -1)
    fig = go.Figure()
    highlight_ids = set(highlight_ids or [])
    old = previous_effects.set_index("study_id") if previous_effects is not None else None
    for i, row in df.iterrows():
        corrected = str(row["study_id"]) in highlight_ids
        newly_added = (not corrected and highlight_phase is not None
                       and "phase" in df and row["phase"] == highlight_phase)
        color = RED if corrected else (ORANGE if newly_added else BLUE)
        symbol = "diamond" if corrected else ("circle" if newly_added else "square")
        weight = float(fit.weights_percent[i])
        status = "corrected" if corrected else ("new study" if newly_added else "included")
        fig.add_trace(go.Scatter(
            x=[row["yi"]], y=[ys[i]], mode="markers",
            marker=dict(symbol=symbol, size=8 + np.sqrt(weight * 10), color=color,
                        line=dict(color="#ffffff", width=1)),
            error_x=dict(type="data", symmetric=False,
                         array=[row["ci_high"] - row["yi"]],
                         arrayminus=[row["yi"] - row["ci_low"]],
                         thickness=1.5, width=4, color=color),
            hovertemplate=(f"<b>{row['author_year']}</b> ({status})<br>MD: {row['yi']:.2f}"
                           f"<br>95% CI: [{row['ci_low']:.2f}; {row['ci_high']:.2f}]"
                           f"<br>Random-effects weight: {weight:.1f}%<extra></extra>"),
            showlegend=False,
        ))
        if corrected and old is not None and row["study_id"] in old.index:
            old_row = old.loc[row["study_id"]]
            fig.add_trace(go.Scatter(
                x=[old_row["yi"]], y=[ys[i]], mode="markers",
                marker=dict(symbol="x", size=13, color=GREY,
                            line=dict(color=GREY, width=2)),
                name="Superseded S06 value", showlegend=True,
                hovertemplate=(f"<b>{row['author_year']}</b><br>Previously extracted MD "
                               f"{old_row['yi']:.2f}<br>Replaced in V4<extra></extra>"),
            ))
    if getattr(fit, 'pi_low', None) is not None and getattr(fit, 'pi_high', None) is not None:
        fig.add_trace(go.Scatter(
            x=[fit.pi_low, fit.pi_high], y=[-.55, -.55],
            mode="lines", line=dict(color=GREEN, dash="dot", width=3),
            name="Approx. 95% prediction interval", showlegend=True,
            hovertemplate=(f"Approx. 95% PI: [{fit.pi_low:.2f}; "
                           f"{fit.pi_high:.2f}]<extra></extra>"),
        ))
    diamond_y = 0.0
    fig.add_trace(go.Scatter(
        x=[fit.ci_low, fit.mean, fit.ci_high, fit.mean, fit.ci_low],
        y=[diamond_y, diamond_y + .27, diamond_y, diamond_y - .27, diamond_y],
        mode="lines", hoveron="fills",
        fill="toself", fillcolor=GREEN, line=dict(color=GREEN, width=1),
        name="Pooled effect", hovertemplate=(f"MD {fit.mean:.2f}<br>95% CI "
        f"[{fit.ci_low:.2f}; {fit.ci_high:.2f}]<extra></extra>")))
    fig.add_vline(x=0, line_dash="dash", line_color=GREY)
    if threshold is not None:
        fig.add_vline(x=threshold, line_dash="dot", line_color=RED, opacity=.7)
    # Study labels include estimated RE weight for transparency.
    labels = [f"{row.author_year}   ({fit.weights_percent[i]:.1f} %)"
              for i, row in enumerate(df.itertuples())] + ["Overall (REML)"]
    fig.update_layout(
        title=title, template="plotly_white",
        height=max(420, 190 + 37 * n), margin=dict(l=12, r=20, t=55, b=55),
        xaxis_title="Mean difference (ISI points) · negative values favour the intervention",
        yaxis=dict(tickmode="array", tickvals=list(ys) + [0], ticktext=labels,
                   range=[-1.05, n + 1.1], zeroline=False, automargin=True),
        showlegend=True, hovermode="closest",
        legend=dict(orientation="h", yanchor="top", y=-.18, x=0),
    )
    if x_range is not None:
        fig.update_xaxes(range=list(x_range), autorange=False)
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
            hovertemplate=(f"{row.version}: MD {row.mean:.2f}<br>95% CI "
                           f"[{row.ci_low:.2f}; {row.ci_high:.2f}]<extra></extra>")))
    fig.add_vline(x=0, line_color=GREY, line_dash="dash", annotation_text="No effect", annotation_position="top right")
    fig.add_vline(x=threshold, line_color=ORANGE, line_dash="dot",
                  annotation_text="Illustrative relevance threshold", annotation_position="top left")
    fig.update_layout(template="plotly_white", height=345,
                      margin=dict(l=10,r=20,t=35,b=40),
                      xaxis_title="Pooled MD and 95% CI", yaxis_title=None,
                      yaxis=dict(categoryorder="array", categoryarray=list(reversed(history["version"].tolist()))))
    return fig


def posterior_plot(density: pd.DataFrame, threshold: float, ci_low: float, ci_high: float):
    fig = go.Figure()
    x = density["MD"].to_numpy()
    y = density["Posterior density"].to_numpy()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=BLUE, width=3),
                             name="Posterior", fill="tozeroy",
                             fillcolor="rgba(50,120,183,.10)",
                             hovertemplate="MD %{x:.2f}<br>Density %{y:.3f}<extra></extra>"))
    mask = x <= threshold
    if np.any(mask):
        fig.add_trace(go.Scatter(x=x[mask], y=y[mask], mode="lines",
                                 line=dict(color=BLUE, width=0),
                                 fill="tozeroy", fillcolor="rgba(50,120,183,.36)",
                                 showlegend=False,
                                 hovertemplate="Below threshold: MD %{x:.2f}<extra></extra>"))
    fig.add_vline(x=0, line_color=GREY, line_dash="dash", annotation_text="0")
    fig.add_vline(x=threshold, line_color=ORANGE, line_dash="dot",
                  annotation_text="Threshold", annotation_position="top left")
    fig.add_vrect(x0=ci_low, x1=ci_high, fillcolor="rgba(22,138,123,.11)", line_width=0,
                  annotation_text="95% credible interval", annotation_position="top left")
    fig.update_layout(template="plotly_white", height=355,
                      xaxis_title="Overall effect μ (ISI points)",
                      yaxis_title="Posterior density", margin=dict(l=8,r=10,t=40,b=40))
    return fig


def updating_plot(story, threshold: float) -> go.Figure:
    """Display the marginal Bayesian update; the factor uses its own scale.

    Prior/posterior curves are probability *densities* for the overall effect.
    The dotted curve is a relative, peak-normalised update/revision factor:
    it is NOT a third probability distribution.
    """
    x = story.curves["MD"].to_numpy(float)
    prior = story.curves["Prior density"].to_numpy(float)
    post = story.curves["Posterior density"].to_numpy(float)
    factor = story.curves["Relative update factor"].to_numpy(float)
    label = "Relative revision factor" if story.kind == "correction" else "Relative evidence update"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=prior, name="Before update · prior for this step",
        mode="lines", line=dict(color=BLUE, width=3),
        hovertemplate="MD %{x:.2f}<br>Prior density %{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=post, name="After update · posterior",
        mode="lines", line=dict(color=GREEN, width=3),
        fill="tozeroy", fillcolor="rgba(22,138,123,.08)",
        hovertemplate="MD %{x:.2f}<br>Posterior density %{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=factor, name=label + " (scaled)",
        mode="lines", yaxis="y2",
        line=dict(color=ORANGE, width=2.5, dash="dot"),
        hovertemplate="MD %{x:.2f}<br>Relative factor (scaled) %{y:.3f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=GREY, line_dash="dash")
    fig.add_vline(x=threshold, line_color=RED, line_dash="dash",
                  annotation_text=f"Threshold ({threshold:g})", annotation_position="top left")
    fig.update_layout(
        title="How evidence changes beliefs about the overall effect",
        template="plotly_white",
        height=465,
        margin=dict(l=10, r=15, t=75, b=100),
        hovermode="x unified",
        xaxis_title="Overall mean difference μ (ISI points)",
        yaxis=dict(title="Probability density of μ", rangemode="tozero"),
        yaxis2=dict(
            title="Relative update factor (scaled to 1)",
            overlaying="y", side="right", range=[0, 1.08],
            showgrid=False,
        ),
        legend=dict(orientation="h", yanchor="top", y=-.28, x=0),
    )
    return fig
