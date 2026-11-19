"""SomniTech Living Evidence Lab — fictional, interactive Streamlit demonstration."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analysis import (
    bayesian_density, calculate_effects, fit_bayesian,
    fit_random_effects, validate_studies,
)
from src.data import VERSION_METADATA, VERSIONS, load_events, snapshot
from src.explanations import LAB_EXPERIMENTS, VERSION_GUIDE
from src.plots import BLUE, forest_plot, history_plot, posterior_plot, updating_plot
from src.updating import build_updating_story


@st.cache_data(show_spinner=False)
def cached_snapshot(version_name: str):
    return snapshot(version_name)


@st.cache_data(show_spinner=False)
def cached_frequentist(version_name: str):
    return fit_random_effects(snapshot(version_name))


@st.cache_data(show_spinner=False)
def cached_history():
    rows = []
    for name in VERSIONS:
        fit_at_version = fit_random_effects(snapshot(name))
        rows.append(dict(version=name, date=VERSION_METADATA[name]["date"],
                         k=fit_at_version.k, n_total=fit_at_version.n_total,
                         mean=fit_at_version.mean, ci_low=fit_at_version.ci_low,
                         ci_high=fit_at_version.ci_high, i2=fit_at_version.i2,
                         tau2=fit_at_version.tau2))
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def cached_bayesian(version_name: str, clinical_threshold: float,
                    effect_prior_sd: float, heterogeneity_prior_sd: float):
    return fit_bayesian(snapshot(version_name), threshold=clinical_threshold,
                        mu_prior_sd=effect_prior_sd,
                        tau_prior_sd=heterogeneity_prior_sd)


@st.cache_data(show_spinner=False)
def cached_density(version_name: str, clinical_threshold: float,
                   effect_prior_sd: float, heterogeneity_prior_sd: float):
    return bayesian_density(snapshot(version_name), threshold=clinical_threshold,
                            mu_prior_sd=effect_prior_sd,
                            tau_prior_sd=heterogeneity_prior_sd)


@st.cache_data(show_spinner=False)
def cached_updating(version_name: str, clinical_threshold: float,
                    effect_prior_sd: float, heterogeneity_prior_sd: float):
    return build_updating_story(version_name, threshold=clinical_threshold,
                                mu_prior_sd=effect_prior_sd,
                                tau_prior_sd=heterogeneity_prior_sd)


@st.cache_data(show_spinner=False)
def canonical_forest_range(clinical_threshold: float):
    extent = [0.0, clinical_threshold]
    for name in VERSIONS:
        effects_at_version = calculate_effects(snapshot(name))
        fit_at_version = fit_random_effects(snapshot(name))
        extent.extend(effects_at_version["ci_low"].tolist())
        extent.extend(effects_at_version["ci_high"].tolist())
        extent.extend([fit_at_version.ci_low, fit_at_version.ci_high])
        if fit_at_version.pi_low is not None:
            extent.extend([fit_at_version.pi_low, fit_at_version.pi_high])
    left, right = min(extent), max(extent)
    pad = max(0.75, .05 * (right - left))
    return (left - pad, right + pad)

st.set_page_config(
    page_title="SomniTech | Living Evidence Lab",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
  .block-container {padding-top: 2.3rem; padding-bottom: 3rem; max-width: 1330px;}
  [data-testid="stMetric"] {background:rgba(127,139,158,.075); border:1px solid rgba(127,139,158,.18);
    border-radius: 12px; padding: 15px 18px;}
  .eyebrow {letter-spacing:.15em; text-transform:uppercase; font-weight:700; font-size:.76rem;
    color:#4785b9; margin-bottom:.2rem;}
  .hero-sub {font-size:1.05rem; opacity:.78; max-width:900px;}
  .notice {border-left:4px solid #d2933f; background:rgba(210,147,63,.09);
    padding:.85rem 1rem; border-radius:0 10px 10px 0; margin:.75rem 0 1.1rem 0;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="eyebrow">Educational simulation · Living evidence maintenance</div>', unsafe_allow_html=True)
st.title("SomniTech · Living Evidence Lab")
st.markdown('<p class="hero-sub">How do new RCTs, conflicting findings and data corrections change a continuously updated meta-analysis?</p>', unsafe_allow_html=True)
st.markdown('<div class="notice"><b>Entirely fictional evidence:</b> SomniTech and all studies are invented. The data, study names and publication dates (including 2027) were created solely for teaching. Do not draw clinical conclusions.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Explore the review")
    version = st.select_slider("Review version", options=list(VERSIONS), value="V0",
                               help="Versions represent the evidence available at each point in time; V4 contains a data correction but no new study.")
    meta = VERSION_METADATA[version]
    st.caption(f"{meta['date']} · {meta['type']}")
    st.markdown(f"**{meta['title']}**")
    st.write(meta["description"])
    st.divider()
    threshold = st.number_input(
        "Illustrative relevance threshold (ISI points)", value=-2.0,
        min_value=-10.0, max_value=-0.1, step=0.5,
        help="Illustrative teaching threshold; not an empirically validated minimum clinically important difference.",
    )
    st.caption("Negative mean differences favour SomniTech. A value of −2 means an average of two fewer ISI points in the intervention group.")
    st.divider()
    st.markdown("**Analysis specification**")
    st.caption("Random-effects model; REML for τ²; modified Hartung–Knapp intervals using t(k−1). I² based on Cochran's Q.")
    st.caption("Bayesian analysis: normal–normal hierarchical model; priors can be adjusted in the methods comparison.")

frame = cached_snapshot(version)
fit = cached_frequentist(version)
effects = calculate_effects(frame)
history = cached_history()
prev_version = VERSIONS[VERSIONS.index(version) - 1] if version != "V0" else None
prev_fit = cached_frequentist(prev_version) if prev_version else None

section_review, section_lab, section_methods, section_workflow = st.tabs([
    "01 · Living Review", "02 · Experiment Lab",
    "03 · Frequentist vs. Bayesian", "04 · Workflow & Data",
])

with section_review:
    with st.expander("Case & PICO · fictional SomniTech trial", expanded=(version == "V0")):
        st.subheader("The fictional case: Does the SomniTech sleep app work?")
        st.write(
            "A research team wants to know whether a **fictional digital sleep intervention** "
            "works better for adults with insomnia than standard "
            "**sleep-hygiene advice**. The review includes only "
            "randomised controlled trials (RCTs). Over time, "
            "new studies become available, and an incorrectly extracted outcome is later corrected. "
            "Follow how these events change the evidence."
        )
        st.markdown("**The research question (PICO)**")
        p1, p2 = st.columns(2, gap="large")
        with p1:
            st.markdown("**P – Population:** Adults with insomnia  \n**I – Intervention:** digital app *SomniTech*")
        with p2:
            st.markdown("**C – Comparator:** general sleep-hygiene advice  \n**O – Outcome:** Insomnia Severity Index (ISI) at eight weeks")
        st.markdown(
            "**What is being compared?** ISI scores range from 0 to 28; lower scores "
            "indicate fewer insomnia symptoms. We analyse the **mean difference "
            "in ISI scores at eight weeks** (intervention minus control), "
            "not changes from baseline within individual participants. "
            "An MD of **−2** means that, on average, the intervention group "
            "scores two ISI points lower than the control group. The adjustable "
            "threshold of −2 is an **illustrative teaching assumption**, not a validated "
            "threshold for a clinically meaningful difference."
        )
        st.info(
            "**Getting started:** Select **V0** in the sidebar and inspect the forest plot. "
            "Then move through **V1–V4**: What happens to the pooled estimate, "
            "confidence interval, and heterogeneity? Studies shown in orange "
            "are newly included. Use the sidebar threshold to "
            "explore different illustrative levels of clinical relevance."
        )
        with st.expander("Why is this a *living* review rather than just a meta-analysis?"):
            st.write(
                "The initial literature search produces V0. Newly published studies are then "
                "identified through scheduled searches and alerts, screened, assessed, "
                "and added to the synthesis. Each review version preserves its own "
                "evidence snapshot. V4 shows how **corrections to existing data** can also "
                "trigger reanalysis. A search that identifies no eligible new studies "
                "does not necessarily warrant a new published version."
            )
    st.divider()
    st.subheader("Evidence snapshot · " + version)
    guide = VERSION_GUIDE[version]
    st.markdown(f"**Guiding question: {guide['question']}**")
    st.caption("The following metrics reflect only the selected fictional evidence snapshot.")
    a,b,c,d = st.columns(4)
    a.metric("Studies", f"{fit.k}", delta=f"{fit.k-prev_fit.k:+d}" if prev_fit else None, delta_color="off")
    b.metric("Participants", f"{fit.n_total:,}",
             delta=f"{fit.n_total-prev_fit.n_total:+d}" if prev_fit else None, delta_color="off")
    c.metric("Pooled MD", f"{fit.mean:+.2f}",
             delta=f"{fit.mean-prev_fit.mean:+.2f}" if prev_fit else None, delta_color="off")
    d.metric("I² (Q-based)", f"{fit.i2:.1f} %",
             delta=f"{fit.i2-prev_fit.i2:+.1f} pp" if prev_fit else None, delta_color="off")
    st.caption(f"95% confidence interval: [{fit.ci_low:+.2f}; {fit.ci_high:+.2f}] ISI points · τ² = {fit.tau2:.3f} · Q-p = {fit.q_pvalue:.3g}")
    if fit.pi_low is not None:
        st.caption(
            f"Approximate 95% prediction interval for the true effect in a new setting: "
            f"[{fit.pi_low:+.2f}; {fit.pi_high:+.2f}] ISI points."
        )
    st.markdown(
        "**Reading the metrics:** The **MD** is the mean difference "
        "between groups (negative values favour SomniTech). The "
        "**95% confidence interval** describes statistical uncertainty. "
        "**I²** describes the proportion of observed variation in "
        "study estimates that exceeds what is expected from sampling error "
        "(based on Q); it is **not a study-quality score**."
    )

    col_forest, col_side = st.columns([2.25, 1.0], gap="large")
    with col_forest:
        st.plotly_chart(
            forest_plot(
                frame, fit, title=f"Forest plot · {version}",
                highlight_phase=int(version[-1]) if version in ("V1", "V2", "V3") else None,
                highlight_ids={"S06"} if version == "V4" else None,
                previous_effects=calculate_effects(cached_snapshot("V3")) if version == "V4" else None,
                threshold=threshold, x_range=canonical_forest_range(threshold),
            ),
            width="stretch", key=f"forest_{version}",
        )
        with st.expander("How to read the forest plot: squares, lines, and the diamond"):
            st.markdown(
                "- **Square:** an individual study's estimated mean difference; "
                "larger squares indicate greater *random-effects weight*.\n"
                "- **Horizontal line:** the study's 95% confidence interval.\n"
                "- **Green diamond at the bottom:** the pooled mean difference; its "
                "width represents the 95% confidence interval of the pooled effect.\n"
                "- **Dashed line at zero:** no average difference "
                "between intervention and control.\n"
                "- **Orange circles:** studies newly included in V1–V3. "
                "- **Red diamond:** corrected study S06 in V4; the grey X marks "
                "its superseded estimate.\n"
                "- **Green dotted line below the diamond:** approximate 95% prediction "
                "interval for a true effect in a new setting (available for k ≥ 3).\n"
                "- **Red vertical dotted line:** illustrative benefit threshold.\n\n"
                "**Hover tip:** Point to a square to see the study's MD, "
                "confidence interval, and weight."
            )
    with col_side:
        st.markdown("#### What changed?")
        st.info(meta["description"])
        st.write("**What does this version show?**")
        st.write(guide["observation"])
        st.write("**What should you look for?**")
        st.caption(guide["try_this"])
        st.write("**Statistical interpretation**")
        if fit.ci_high < 0:
            st.success("The 95% confidence interval lies entirely below zero. Under the selected model, the pooled MD differs statistically from zero.")
        else:
            st.warning("The 95% confidence interval includes zero; the data remain compatible with no average effect.")
        if fit.i2 > 75:
            st.warning("Very high heterogeneity: the pooled MD alone may not adequately describe the variation across studies.")
        st.caption("Statistical significance does not establish clinical importance or high certainty of evidence.")

    st.markdown("#### How has the evidence changed?")
    st.plotly_chart(history_plot(history, selected=version, threshold=threshold),
                    width="stretch", key="history_plot")
    st.caption(
        "Each row shows the pooled effect for one review version, with its "
        "95% confidence interval. The dashed zero line denotes no difference; "
        "the dotted line marks the illustrative "
        "relevance threshold set in the sidebar. The selected version is highlighted in orange."
    )
    st.info(
        "**Key point:** A later version may confirm, weaken, or reverse "
        "earlier conclusions. The repeated frequentist "
        "confidence intervals have not been adjusted for sequential testing."
    )
    table_hist = history.copy()
    table_hist["95% CI"] = table_hist.apply(lambda r:f"[{r.ci_low:.2f}; {r.ci_high:.2f}]", axis=1)
    table_hist["MD"] = table_hist["mean"].round(2)
    table_hist["I² (%)"] = table_hist["i2"].round(1)
    st.dataframe(table_hist[["version", "date", "k", "MD", "95% CI", "I² (%)"]],
                 hide_index=True, width="stretch")

    st.markdown("#### Risk-of-bias sensitivity analysis")
    st.write(
        "Not all studies are equally trustworthy. Here, we rerun "
        "the meta-analysis using only those studies assigned the "
        "**simulated risk-of-bias category 'Low'**. Compare this result with "
        "the full analysis. This is a simplified sensitivity analysis, "
        "not a complete RoB 2 or GRADE assessment."
    )
    lowrisk = frame.loc[frame["rob"].eq("Low")].copy()
    if len(lowrisk) >= 2:
        lower_fit = fit_random_effects(lowrisk)
        a,b = st.columns(2)
        a.metric("Low risk of bias only", f"{lower_fit.mean:+.2f}",
                 help=f"{len(lowrisk)} of {len(frame)} studies.")
        b.metric("95% CI for this subset", f"[{lower_fit.ci_low:+.2f}; {lower_fit.ci_high:+.2f}]")
        st.caption("Exploratory sensitivity analysis only. No automatic GRADE certainty assessment; individual RoB 2 domains are not simulated.")
        if len(lowrisk) == 2:
            st.caption("Only two studies: the modified Hartung–Knapp interval uses t(1) ≈ 12.71 and can be very wide. A prediction interval is not reported.")
    else:
        st.caption("Fewer than two studies are classified as low risk of bias in this version.")

with section_lab:
    st.subheader("Experiment Lab · How do individual studies affect the synthesis?")
    st.write(
        "Here you can change the meta-analysis yourself. Begin with the "
        f"**{fit.k} studies in {version}** as your starting point. This is a "
        "**what-if simulation**: it does not change published "
        "source data or create additional review versions."
    )
    st.markdown("**What can you try?**")
    st.markdown(
        "1. **Edit or exclude a study:** Change a mean, standard deviation "
        "or sample size, or uncheck **Include** for leave-one-out analysis. "
        "Then select **Apply changes and recalculate**.\n"
        "2. **Add a new study:** Open the entry form below "
        "the table, enter data for both arms, and select "
        "**Add study**.\n"
        "3. **Compare results:** The right-hand panel shows the new pooled "
        "MD, I², and forest plot. Changes are measured "
        "against the unmodified review version.\n"
        "4. **Optional Bayesian comparison:** Turn on the Bayesian metric "
        "to see how P(μ below threshold) changes as well.\n"
        "5. **Reset or export:** Use the reset button "
        "to start again; download your experimental "
        "study data as a CSV file."
    )
    with st.expander("Three guided experiments to try"):
        for title, description in LAB_EXPERIMENTS:
            st.markdown(f"**{title}**")
            st.markdown(description)
    st.info(
        "**Think about it:** What influences the pooled effect more – "
        "a study's effect direction, its sample size, or the "
        "differences across studies? Try the changes "
        "one at a time."
    )
    st.caption(
        "Changes apply only to the current browser session and selected "
        "version. Switching between V0–V4 in the sidebar starts "
        "the lab afresh for that version."
    )
    if "exp_baseline" not in st.session_state or st.session_state.exp_baseline != version:
        st.session_state.exp_baseline = version
        st.session_state.exp_frame = snapshot(version).assign(include=True).assign(include=True)
        st.session_state.editor_revision = st.session_state.get("editor_revision", 0) + 1

    left, right = st.columns([1.4, 1.0], gap="large")
    with left:
        st.markdown("**Edit existing studies**")
        exp_cols = ["include", "study_id", "author_year", "n_int", "mean_int", "sd_int",
                    "n_ctrl", "mean_ctrl", "sd_ctrl", "rob"]
        with st.form(f"editor_{st.session_state.editor_revision}"):
            edited = st.data_editor(
                st.session_state.exp_frame[exp_cols], num_rows="fixed",
                width="stretch", hide_index=True,
                disabled=["study_id", "author_year"],
                column_config={
                    "include": st.column_config.CheckboxColumn("Include", help="Uncheck to exclude this study from the current synthesis."),
                    "study_id": st.column_config.TextColumn("ID"),
                    "author_year": st.column_config.TextColumn("Study"),
                    "n_int": st.column_config.NumberColumn("n Int.", min_value=2, step=1, format="%d"),
                    "n_ctrl": st.column_config.NumberColumn("n Control", min_value=2, step=1, format="%d"),
                    "mean_int": st.column_config.NumberColumn("Mean Int.", min_value=0.0, max_value=28.0, format="%.2f"),
                    "mean_ctrl": st.column_config.NumberColumn("Mean Control", min_value=0.0, max_value=28.0, format="%.2f"),
                    "sd_int": st.column_config.NumberColumn("SD Int.", min_value=0.01, format="%.2f"),
                    "sd_ctrl": st.column_config.NumberColumn("SD Control", min_value=0.01, format="%.2f"),
                    "rob": st.column_config.SelectboxColumn("Risk of Bias", options=["Low", "Some concerns", "High"]),
                },
                key=f"editor_key_{st.session_state.editor_revision}",
            )
            saved = st.form_submit_button("Apply changes and recalculate", type="primary")
        if saved:
            try:
                edited = edited.copy()
                edited["include"] = edited["include"].fillna(False).astype(bool)
                validate_studies(edited, min_studies=0)
                if edited["include"].sum() < 2:
                    raise ValueError("Keep at least two studies included for meta-analysis.")
                # Align on immutable study_id and keep unedited metadata, rather
                # than assigning values row-by-row (which can coerce dtypes).
                current = st.session_state.exp_frame.copy().set_index("study_id")
                update = edited.set_index("study_id")
                for col in exp_cols:
                    if col != "study_id":
                        current[col] = update[col].reindex(current.index).to_numpy()
                for col in ("n_int", "n_ctrl"):
                    current[col] = current[col].astype(int)
                current["include"] = current["include"].astype(bool)
                st.session_state.exp_frame = current.reset_index()
                st.success("The experimental meta-analysis has been updated.")
            except (ValueError, TypeError, KeyError) as exc:
                st.error(f"Invalid study data: {exc}")
        if st.button("Reset experiment to original version"):
            st.session_state.exp_frame = snapshot(version)
            st.session_state.editor_revision += 1
            st.rerun()

        with st.expander("Add a new synthetic study"):
            with st.form("add_exp_study"):
                study_name = st.text_input("Study label", value="New study (simulation)")
                n1, n2 = st.columns(2)
                n_int = n1.number_input("N intervention", min_value=2, value=200, step=10)
                n_ctrl = n2.number_input("N control", min_value=2, value=200, step=10)
                x1, x2 = st.columns(2)
                mean_int = x1.number_input("Mean ISI, intervention", min_value=0.0, max_value=28.0, value=11.0, step=.1)
                mean_ctrl = x2.number_input("Mean ISI, control", min_value=0.0, max_value=28.0, value=12.0, step=.1)
                x3, x4 = st.columns(2)
                sd_int = x3.number_input("SD intervention", min_value=.01, value=4.0, step=.1)
                sd_ctrl = x4.number_input("SD control", min_value=.01, value=4.0, step=.1)
                rob = st.selectbox("Risk of Bias", ["Low", "Some concerns", "High"])
                submitted = st.form_submit_button("Add study")
            if submitted:
                existing = st.session_state.exp_frame
                suffix = 1
                while f"EXP-{suffix:03d}" in existing["study_id"].astype(str).values:
                    suffix += 1
                row = {col: np.nan for col in existing.columns}
                row.update(study_id=f"EXP-{suffix:03d}", author_year=study_name,
                           n_int=int(n_int), n_ctrl=int(n_ctrl), mean_int=float(mean_int),
                           mean_ctrl=float(mean_ctrl), sd_int=float(sd_int),
                           sd_ctrl=float(sd_ctrl), rob=rob, phase=4, include=True)
                try:
                    st.session_state.exp_frame = validate_studies(
                        pd.concat([existing, pd.DataFrame([row])], ignore_index=True))
                    st.session_state.editor_revision += 1
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))

    with right:
        try:
            experiment = validate_studies(
                st.session_state.exp_frame.loc[
                    st.session_state.exp_frame["include"].astype(bool)
                ]
            )
            result = fit_random_effects(experiment)
            st.markdown("**Recalculated results**")
            st.caption(
                f"Comparison with {version}: The baseline estimate is "
                f"{fit.mean:+.2f} ISI points; the analysis below uses "
                "only your experimental data."
            )
            a,b=st.columns(2)
            a.metric("Experimental MD", f"{result.mean:+.2f}",
                     delta=f"{result.mean-fit.mean:+.2f} vs. {version}", delta_color="off")
            b.metric("I²", f"{result.i2:.1f} %")
            st.write(f"95% CI **[{result.ci_low:+.2f}; {result.ci_high:+.2f}]** · {result.k} studies")
            if result.pi_low is not None:
                st.caption(f"Approx. 95% prediction interval: [{result.pi_low:+.2f}; {result.pi_high:+.2f}]")
            if st.toggle("Show Bayesian sensitivity", value=False, key="lab_bayesian_toggle"):
                simulated_bayes = fit_bayesian(experiment, threshold=threshold)
                st.metric(f"P(μ < {threshold:g}) · experimental", f"{simulated_bayes.p_relevant:.1%}")
                st.caption(f"Original version: {cached_bayesian(version, threshold, 5.0, 2.0).p_relevant:.1%}; same default priors in both analyses.")
            st.caption(
                "Leave-one-out: uncheck **Include** and apply changes. "
                "The delta indicator is **new MD minus original MD**. "
                "A positive value indicates a shift towards a less "
                "negative effect; a negative delta indicates a shift towards "
                "a more negative effect."
            )
            st.plotly_chart(forest_plot(experiment, result, title="Experimental forest plot"),
                            width="stretch", key="experimental_forest")
            out = calculate_effects(st.session_state.exp_frame)
            st.download_button("Download experimental study data (CSV)",
                               data=out.to_csv(index=False).encode("utf-8-sig"),
                               file_name="somnitech_experiment.csv", mime="text/csv")
        except ValueError as exc:
            st.error(f"Unable to calculate meta-analysis: {exc}")
    st.caption("The experiment lab modifies aggregate study data; it does not simulate individual participant records.")

with section_methods:
    st.subheader("Frequentist vs. Bayesian · How should we interpret the results?")
    st.write(
        f"Both approaches use the **same {fit.k} synthetic studies "
        f"from {version}** and a random-effects model. "
        "They address related questions but express uncertainty "
        "differently. Change the review version in the sidebar and "
        "see how both results evolve as evidence accumulates."
    )
    concept_f, concept_b = st.columns(2, gap="large")
    with concept_f:
        st.markdown("**Frequentist perspective**")
        st.write(
            "We estimate the overall mean difference and "
            "construct a 95% **confidence interval**. If the same "
            "procedure were repeated many times under comparable conditions, "
            "approximately 95% of the intervals constructed would contain the "
            "true parameter. A particular confidence interval "
            "does **not** directly assign a 95% probability to μ."
        )
    with concept_b:
        st.markdown("**Bayesian perspective**")
        st.write(
            "Before seeing these data, we specify **priors** for the overall effect (μ) "
            "and between-study heterogeneity (τ). Combining these priors with the study data "
            "produces a **posterior distribution**. A 95% " 
            "**credible interval**, conditional on the model and priors, contains "
            "95% of the posterior probability for μ."
        )
    st.info(
        "**The key updating distinction:** In the frequentist approach, the model is "
        "refitted using the currently available studies. In the Bayesian approach, "
        "the evidence updates prior beliefs. For each version, this app combines "
        "the same original prior with the complete, valid "
        "dataset; previously included studies are not counted twice."
    )
    st.markdown("#### 1 · Set the priors")
    st.write(
        "A prior represents assumptions about a parameter **before observing these data**. "
        "A larger prior standard deviation "
        "allows a wider range of effects. Explore whether "
        "your interpretation changes under narrower or broader priors."
    )
    col_prior1, col_prior2 = st.columns(2)
    mu_prior_sd = col_prior1.select_slider("Prior SD for μ (Normal(0, SD))",
                                           options=[1.0, 2.0, 5.0, 10.0], value=5.0,
                                           help="μ represents the true overall mean difference. Normal(0, 5²) allows larger departures from zero than Normal(0, 1²).")
    tau_prior_sd = col_prior2.select_slider("Prior SD for τ (Half-Normal(SD))",
                                            options=[0.5, 1.0, 2.0, 4.0], value=2.0,
                                            help="τ describes the standard deviation of the true study effects (ISI points). A half-normal prior restricts τ to non-negative values.")
    st.caption(
        f"Current priors: μ ~ Normal(0, {mu_prior_sd:g}²); "
        f"τ ~ Half-Normal({tau_prior_sd:g}). "
        "The priors do not assign study-specific weights based on methodological quality."
    )
    st.markdown("#### 2 · Compare estimates and uncertainty intervals")
    bayes = fit_bayesian(frame, threshold=threshold,
                         mu_prior_sd=mu_prior_sd, tau_prior_sd=tau_prior_sd)
    fa, fb = st.columns(2, gap="large")
    with fa:
        st.markdown("#### Frequentist")
        st.metric("Pooled MD (REML)", f"{fit.mean:+.2f}")
        st.write(f"95% confidence interval: **[{fit.ci_low:+.2f}; {fit.ci_high:+.2f}]**")
        st.write(f"Between-study variance τ²: **{fit.tau2:.3f}**")
        st.caption("A 95% confidence interval is interpreted through the procedure's long-run coverage, not as a posterior probability.")
    with fb:
        st.markdown("#### Bayesian")
        st.metric("Posterior mean μ", f"{bayes.mean:+.2f}")
        st.write(f"95% credible interval: **[{bayes.ci_low:+.2f}; {bayes.ci_high:+.2f}]**")
        st.write(f"P(μ < 0 | data): **{bayes.p_benefit:.1%}**")
        st.write(f"P(μ < {threshold:g} | data): **{bayes.p_relevant:.1%}**")
        st.caption(
            f"Posterior mean τ: {bayes.tau_mean:.2f}. "
            "P(μ < 0) is the posterior probability of an "
            "average benefit of the app. P(μ < threshold) is the "
            "probability of an effect at least as large as the "
            "illustrative threshold for a meaningful benefit."
        )
    st.markdown("#### 3 · Read the posterior distribution")
    st.write(
        "The curve shows which values of the **true overall "
        "effect μ** are more or less plausible given the data and priors. "
        "The **area to the left of zero** corresponds to P(μ < 0). "
        f"The **area to the left of {threshold:g}** corresponds to P(μ < {threshold:g}). "
        "The green band contains 95% of the posterior probability mass. "
        "Change a prior and observe how the distribution and probabilities respond."
    )
    density = bayesian_density(frame, mu_prior_sd=mu_prior_sd, tau_prior_sd=tau_prior_sd)
    st.plotly_chart(posterior_plot(density, threshold, bayes.ci_low, bayes.ci_high),
                    width="stretch", key="bayes_posterior")
    st.divider()
    st.markdown("#### 4 · Watch the Bayesian update happen")
    st.write(
        "A Bayesian prior is **not fixed forever**. Once we have analysed an earlier "
        "review version, its **joint posterior for the overall effect μ and "
        "heterogeneity τ becomes the prior for the next update**. New trials "
        "can strengthen, weaken, or even reverse earlier conclusions."
    )
    st.write(
        "Choose an update below. This selector is independent of the review version "
        "in the sidebar, so you can compare any step in the evidence timeline."
    )
    update_version = st.select_slider(
        "Bayesian update to inspect",
        options=list(VERSIONS),
        value=version if version != "V0" else "V1",
        help="V0: original prior and first four trials. V1–V3: previous posterior plus new trials. V4: replace an erroneous outcome; no new trial.",
        key="bayesian_update_to_inspect",
    )
    story = build_updating_story(
        update_version,
        threshold=threshold,
        mu_prior_sd=mu_prior_sd,
        tau_prior_sd=tau_prior_sd,
    )
    before_label = "Original prior" if story.previous_version is None else f"Posterior from {story.previous_version}"
    after_label = f"Posterior for {story.version}"
    if story.kind == "initial":
        st.info(
            "**V0 · Initial synthesis:** Start from Normal(0, SD) for μ and a half-normal "
            "prior for τ, then learn from the four initial trials. This is the "
            "first update, before any published review version exists."
        )
    elif story.kind == "correction":
        st.warning(
            "**V4 · Data correction, not a new trial:** The old S06 four-week result "
            "is removed and replaced by its corrected eight-week result. "
            "Simply multiplying the V3 posterior by the corrected study's "
            "likelihood would **double-count S06**. This figure instead shows "
            "the correct before/after revision based on the valid snapshots."
        )
        st.caption(f"Revised study IDs: {', '.join(story.corrected_studies)}.")
    else:
        st.success(
            f"**{story.previous_version} → {story.version}:** "
            f"{len(story.added_studies)} newly included trial(s): "
            + ", ".join(story.added_studies)
        )
    step_a, step_b, step_c = st.columns(3, gap="medium")
    with step_a:
        st.markdown(f"**1 · Before update**")
        st.caption(f"{before_label}. Blue curve: our uncertainty about μ before this step.")
    with step_b:
        st.markdown("**2 · Information from the update**")
        st.caption(
            "Dotted orange: which effect values gain relative support from the new "
            "evidence (or the correction in V4), allowing for uncertainty in τ."
        )
    with step_c:
        st.markdown("**3 · After update**")
        st.caption(f"{after_label}. Green curve: the revised probability distribution for μ.")
    st.plotly_chart(
        updating_plot(story, threshold),
        width="stretch",
        key=f"bayesian_step_chart_{update_version}",
    )
    st.caption(
        "**Important:** The two solid curves are actual marginal probability densities. "
        "The dotted orange curve is the **relative updating factor**, scaled to a "
        "maximum of 1 on the visible overlap range and plotted on the right axis. "
        "It is **not** a probability density or a separate posterior. "
        "The factor accounts for uncertainty about τ inherited from the preceding data."
    )
    st.markdown("**How much did our probability assessments change?**")
    probability_comparison = pd.DataFrame({
        "Question": [
            "P(μ < 0) · Any average benefit",
            f"P(μ < {threshold:g}) · Benefit beyond illustrative threshold",
        ],
        "Before": [story.prior_benefit, story.prior_relevant],
        "After": [story.posterior_benefit, story.posterior_relevant],
    })
    probability_comparison["Change (percentage points)"] = (
        100.0 * (probability_comparison["After"] - probability_comparison["Before"])
    )
    probability_comparison["Before"] = probability_comparison["Before"].map(lambda p: f"{p:.1%}")
    probability_comparison["After"] = probability_comparison["After"].map(lambda p: f"{p:.1%}")
    probability_comparison["Change (percentage points)"] = (
        probability_comparison["Change (percentage points)"].map(lambda p: f"{p:+.1f} pp")
    )
    st.dataframe(probability_comparison, hide_index=True, width="stretch")
    st.caption(
        f"Evidence snapshots: {story.study_count_before} studies before and "
        f"{story.study_count_after} after this step. A change in probability "
        "does not by itself establish certainty of evidence or clinical benefit."
    )
    with st.expander("Why does the entire joint posterior matter?"):
        if story.kind == "correction":
            st.latex(
                r"p(\mu,\tau\mid D_{\mathrm{revised}})\propto "
                r"p(\mu,\tau\mid D_{\mathrm{old}})\,"
                r"\frac{p(y_{S06}^{\mathrm{corrected}}\mid\mu,\tau)}"
                r"{p(y_{S06}^{\mathrm{old}}\mid\mu,\tau)}"
            )
            st.write(
                "The old S06 likelihood contribution must be removed before adding "
                "the corrected contribution. Recalculating with the corrected "
                "snapshot is equivalent and avoids double-counting."
            )
        else:
            st.latex(
                r"p(\mu,\tau\mid D_{\mathrm{old}},D_{\mathrm{new}})\propto "
                r"p(D_{\mathrm{new}}\mid\mu,\tau)\,"
                r"p(\mu,\tau\mid D_{\mathrm{old}})"
            )
            st.write(
                "The current joint posterior is formed from the previous **joint** "
                "posterior and the likelihood of only the newly added studies. "
                "Keeping merely the marginal posterior of μ would discard "
                "dependence between μ and τ."
            )
        st.write(
            "For numerical stability and reproducibility, the app recomputes each "
            "complete snapshot using the same initial priors. With an unchanged "
            "model this yields the same posterior as exact sequential updating. "
            "The orange curve is calculated as the ratio of the two marginal "
            "posterior densities; it is an **effective update factor for μ**, "
            "not a conventional likelihood based on a fixed value of τ."
        )

    st.markdown("#### 5 · How does the probability change across review versions?")
    st.write(
        "For each review version, the chart shows the Bayesian "
        "probability that the true overall MD is below "
        "the selected relevance threshold. This is **not** "
        "the probability that an individual study will be statistically significant. "
        "Pay particular attention to the transition from V1 to V2 and the correction in V4."
    )
    b_history = []
    for v in VERSIONS:
        b = fit_bayesian(snapshot(v), threshold=threshold,
                         mu_prior_sd=mu_prior_sd, tau_prior_sd=tau_prior_sd)
        b_history.append(dict(Version=v, Probability=b.p_relevant, P_benefit=b.p_benefit))
    b_table = pd.DataFrame(b_history)
    fig = go.Figure(go.Scatter(x=b_table["Version"], y=100*b_table["Probability"],
                               mode="lines+markers", marker=dict(size=11, color=BLUE),
                               line=dict(width=3, color=BLUE)))
    fig.update_layout(template="plotly_white", height=290,
                      xaxis_title="Review version", yaxis_title=f"P(μ < {threshold:g}) in %",
                      yaxis_range=[0,100], margin=dict(l=10,r=10,t=25,b=40))
    st.plotly_chart(fig, width="stretch", key="bayes_history")
    with st.expander("Frequently asked questions: confidence intervals, credible intervals, priors and updating"):
        st.markdown(
            "**Why do the point estimates differ?** The "
            "frequentist analysis estimates heterogeneity using REML; "
            "the Bayesian analysis integrates uncertainty about τ "
            "and incorporates the priors for μ and τ.\n\n"
            "**Can I say there is a 95% probability the intervention works?** "
            "Only a clearly specified Bayesian probability such as "
            "P(μ < 0 | data, model, priors) can be "
            "interpreted this way. A frequentist 95% confidence interval "
            "does not mean P(μ lies in this interval) = 95%.\n\n"
            "**Are narrower priors always better?** No. Overly narrow priors "
            "may dominate the observed evidence; "
            "this is why prior sensitivity analysis matters.\n\n"
            "**What does 'living' mean statistically?** New or "
            "corrected data change the current evidence snapshot. "
            "The repeated frequentist intervals in this demonstration "
            "are not adjusted for sequential testing; Bayesian "
            "probabilities depend on model assumptions and do not protect "
            "against bias or selective reporting."
        )
    st.info("Each snapshot uses the same original prior and only the data valid for that version. Corrected values replace earlier values rather than being treated as an additional study.")
    with st.expander("Statistical model and limitations"):
        st.latex(r"y_i\mid \mu,\tau \sim \mathcal{N}(\mu,\,v_i+\tau^2)")
        st.latex(r"\mu \sim \mathcal{N}(0,\sigma_{\mu}^2),\quad \tau\sim\operatorname{HalfNormal}(\sigma_{\tau})")
        st.write("The Bayesian posterior is computed without MCMC, integrating analytically over μ and numerically over τ. Studies are assumed independent, and the estimated within-study variances are treated as known.")
        st.write("The two-point ISI threshold is illustrative only. Neither a high posterior probability nor a small p-value compensates for poor internal validity, publication bias, or model misspecification.")

with section_workflow:
    st.subheader("From search alert to updated review version")
    st.write("A living review is more than a meta-analysis run repeatedly: surveillance, screening, data verification, reanalysis, interpretation, and publication are separate steps.")
    st.markdown("**Process overview**")
    p1,p2,p3,p4,p5 = st.columns(5)
    for column, txt in zip((p1,p2,p3,p4,p5),
                           ("1 · Surveillance", "2 · Screening", "3 · Extraction & QA",
                            "4 · Reanalysis", "5 · Publication")):
        column.info(txt)
    events = load_events()
    allowed = list(VERSIONS[:VERSIONS.index(version)+1])
    events = events[events["version"].isin(allowed)]
    st.markdown(f"**Event log through {version}**")
    st.dataframe(events.rename(columns={"version":"Version", "event_date":"Date",
                                             "event_type":"Event", "study_id":"Study", "details":"Description"}),
                 hide_index=True, width="stretch")
    st.markdown("**Study data for this snapshot**")
    st.dataframe(effects[["study_id", "author_year", "n_int", "n_ctrl", "mean_int",
                          "mean_ctrl", "sd_int", "sd_ctrl", "yi", "vi", "rob"]],
                 hide_index=True, width="stretch")
    a,b,c = st.columns(3)
    a.download_button("Snapshot CSV", data=frame.to_csv(index=False).encode("utf-8-sig"),
                      file_name=f"somnitech_{version}_studies.csv", mime="text/csv")
    b.download_button("Event log CSV", data=events.to_csv(index=False).encode("utf-8-sig"),
                      file_name=f"somnitech_{version}_events.csv", mime="text/csv")
    c.download_button("Version comparison CSV", data=history.to_csv(index=False).encode("utf-8-sig"),
                      file_name="somnitech_versions.csv", mime="text/csv")
    st.divider()
    st.markdown("**What decisions should the review support?**")
    st.write("The simulation separates three decisions: **(1)** Does monitoring identify records that need screening? **(2)** Do new or corrected data require a new synthesis? **(3)** Does the reassessment warrant publishing a new review version?")
    st.caption("All events and publications shown in the log are fictional. The application does not perform live PubMed searches or generate real clinical evidence.")
