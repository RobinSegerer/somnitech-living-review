"""Smoke tests for the Streamlit app and accessibility-related visual choices."""
from pathlib import Path

from src.analysis import bayesian_density, calculate_effects, fit_bayesian, fit_random_effects
from src.data import snapshot
from src.plots import forest_plot, posterior_plot
from src.updating import build_updating_story


def test_forest_plot_tolerates_pre_prediction_interval_cached_results():
    """Old pickled model objects must not crash the forest plot during redeploy."""
    from types import SimpleNamespace
    frame = snapshot("V0")
    fit = fit_random_effects(frame)
    old_fields = {key: value for key, value in vars(fit).items()
                  if key not in ("pi_low", "pi_high")}
    old_fit = SimpleNamespace(**old_fields)
    fig = forest_plot(frame, old_fit)
    assert all(trace.name != "Approx. 95% prediction interval"
               for trace in fig.data)


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


def test_forest_plot_interface_matches_streamlit_caller():
    from inspect import signature
    from src.plots import PLOT_INTERFACE_VERSION, forest_plot
    assert PLOT_INTERFACE_VERSION == "forest-2026-11-v3"
    expected = {"highlight_phase", "highlight_ids", "previous_effects", "threshold", "x_range"}
    assert expected.issubset(signature(forest_plot).parameters)


def test_streamlit_application_renders_all_review_versions():
    """Test all version-specific forest plots, including the V4 correction."""
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=120).run()
    assert len(app.exception) == 0, [str(x.message) for x in app.exception]
    for version in ("V1", "V2", "V3", "V4"):
        app.sidebar.select_slider[0].set_value(version).run()
        assert len(app.exception) == 0, [f'{version}: {x.message}' for x in app.exception]

def test_streamlit_application_smoke_render():
    """Run the UI without a browser to catch widget and runtime regressions."""
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=120).run()
    assert len(app.exception) == 0, [str(x.message) for x in app.exception]


def test_optional_teaching_toggles_render_without_errors():
    """Exercise the weight and repeated-look plots, not only their default state."""
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=120).run()
    assert not app.exception, [str(e.message) for e in app.exception]
    for key in ("compare_review_weights", "show_repeated_look_intervals"):
        widget = next(x for x in app.toggle if x.key == key)
        widget.set_value(True).run()
        assert not app.exception, [f"{key}: {e.message}" for e in app.exception]


def test_file_drawer_button_runs_and_preserves_include_flags():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=120).run()
    button = next(b for b in app.button if b.key == "add_file_drawer_preset")
    button.click().run()
    assert not app.exception, [str(e.message) for e in app.exception]
    data = app.session_state["exp_frame"]
    assert len(data.loc[data["study_id"].astype(str).str.startswith("FD-")]) == 2
    assert "include" in data.columns

