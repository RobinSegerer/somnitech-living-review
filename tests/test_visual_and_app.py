"""Smoke tests for the Streamlit app and accessibility-related visual choices."""
from pathlib import Path

from src.analysis import bayesian_density, calculate_effects, fit_bayesian, fit_random_effects
from src.data import snapshot
from src.plots import forest_plot, posterior_plot
from src.updating import build_updating_story


def test_app_compiles():
    source = Path("app.py").read_text(encoding="utf-8")
    compile(source, "app.py", "exec")


def test_v4_highlights_corrected_study_and_old_effect():
    current = snapshot("V4")
    old = calculate_effects(snapshot("V3"))
    fit = fit_random_effects(current)
    fig = forest_plot(
        current, fit, highlight_ids={"S06"},
        previous_effects=old, threshold=-2, x_range=(-10, 10),
    )
    assert fig.layout.xaxis.range == (-10, 10)
    assert any(trace.name == "Superseded S06 value" for trace in fig.data)
    assert any(trace.name == "Approx. 95% prediction interval" for trace in fig.data)
    assert any(getattr(getattr(trace, "marker", None), "symbol", None) == "diamond"
               for trace in fig.data)


def test_newly_included_studies_use_non_square_symbol():
    frame = snapshot("V1")
    fig = forest_plot(frame, fit_random_effects(frame), highlight_phase=1)
    assert any(getattr(getattr(trace, "marker", None), "symbol", None) == "circle"
               for trace in fig.data)


def test_posterior_plot_shades_threshold_probability():
    frame = snapshot("V2")
    b = fit_bayesian(frame)
    d = bayesian_density(frame)
    fig = posterior_plot(d, -2, b.ci_low, b.ci_high)
    assert len(fig.data) >= 2
    assert fig.data[1].fill == "tozeroy"


def test_updating_tracks_tau_mean():
    v1 = build_updating_story("V1")
    v2 = build_updating_story("V2")
    assert v1.posterior_tau_mean > 0
    assert v2.prior_tau_mean == v1.posterior_tau_mean
    assert v2.posterior_tau_mean > v2.prior_tau_mean


def test_streamlit_application_smoke_render():
    """Run the UI without a browser to catch widget and runtime regressions."""
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=120).run()
    assert len(app.exception) == 0, [str(x.message) for x in app.exception]
