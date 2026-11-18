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
from src.plots import BLUE, forest_plot, history_plot, posterior_plot

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

frame = snapshot(version)
fit = fit_random_effects(frame)
effects = calculate_effects(frame)
history_rows = []
for v in VERSIONS:
    model = fit_random_effects(snapshot(v))
    history_rows.append(dict(version=v, date=VERSION_METADATA[v]["date"],
                             k=model.k, mean=model.mean, ci_low=model.ci_low,
                             ci_high=model.ci_high, i2=model.i2, tau2=model.tau2))
history = pd.DataFrame(history_rows)

section_review, section_lab, section_methods, section_workflow = st.tabs([
    "01 · Living Review", "02 · Experiment Lab",
    "03 · Frequentist vs. Bayesian", "04 · Workflow & Data",
])

with section_review:
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
    a.metric("Studies", f"{fit.k}")
    b.metric("Participants", f"{fit.n_total:,}".replace(",", "’"))
    c.metric("Pooled MD", f"{fit.mean:+.2f}")
    d.metric("I² (Q-based)", f"{fit.i2:.1f} %")
    st.caption(f"95% confidence interval: [{fit.ci_low:+.2f}; {fit.ci_high:+.2f}] ISI-Punkte · τ² = {fit.tau2:.3f} · Q-p = {fit.q_pvalue:.3g}")
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
        st.plotly_chart(forest_plot(frame, fit, title=f"Forest Plot · {version}",
                                    highlight_phase=int(version[-1]) if version in ("V1", "V2", "V3") else None),
                        width="stretch", key=f"forest_{version}")
        with st.expander("How to read the forest plot: squares, lines, and the diamond"):
            st.markdown(
                "- **Square:** an individual study's estimated mean difference; "
                "larger squares indicate greater *random-effects weight*.\n"
                "- **Horizontal line:** the study's 95% confidence interval.\n"
                "- **Green diamond at the bottom:** the pooled mean difference; its "
                "width represents the 95% confidence interval of the pooled effect.\n"
                "- **Dashed line at zero:** no average difference "
                "between intervention and control.\n"
                "- **Orange:** newly included studies in V1–V3. By contrast, V4 corrects "
                "existing study S06.\n\n"
                "**Hover tip:** Point to a square to see the study's MD, "
                "confidence interval, and weight."
            )
    with col_side:
        st.markdown("#### What changed?")
        st.info(meta["description"])
        if version != "V0":
            prev_name = VERSIONS[VERSIONS.index(version) - 1]
            prev = history.iloc[VERSIONS.index(version)-1]
            st.metric("Change in MD", f"{fit.mean - prev['mean']:+.2f}",
                      help=f"Current effect minus the effect in {prev_name}.")
            st.metric("Change in number of studies", f"{fit.k-int(prev['k']):+d}")
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
        "Jede Zeile zeigt den Gesamteffekt einer eigenen Review version mit "
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
                 help=f"{len(lowrisk)} von {len(frame)} Studien.")
        b.metric("95% CI for this subset", f"[{lower_fit.ci_low:+.2f}; {lower_fit.ci_high:+.2f}]")
        st.caption("Exploratory sensitivity analysis only. No automatic GRADE certainty assessment; individual RoB 2 domains are not simulated.")
    else:
        st.caption("Fewer than two studies are classified as low risk of bias in this version.")

with section_lab:
    st.subheader("Experimentierlabor · Wie beeinflussen einzelne Studien die Synthese?")
    st.write(
        "Hier kannst du die Metaanalyse selbst verändern. Du arbeitest mit den "
        f"**{fit.k} Studien aus {version}** als Startpunkt. Das Labor ist eine "
        "**Was-wäre-wenn-Simulation**: Es verändert keine publizierten "
        "Originaldaten und legt keine neuen Review versionen an."
    )
    st.markdown("**Was kannst du ausprobieren?**")
    st.markdown(
        "1. **Eine Studie bearbeiten:** Verändere beispielsweise einen "
        "Gruppenmittelwert, die Standardabweichung oder die Stichprobengrösse. "
        "Klicke anschliessend auf **‚Änderungen übernehmen und neu berechnen‘**.\n"
        "2. **Eine neue Studie hinzufügen:** Öffne das Eingabeformular unter "
        "der Tabelle, gib Daten für beide Gruppen ein und klicke auf "
        "**‚Studie einschliessen‘**.\n"
        "3. **Ergebnisse vergleichen:** Rechts siehst du die neue gepoolte "
        "MD, I² und den Forest Plot. Die angezeigte Veränderung bezieht sich "
        "auf die unveränderte Originalversion.\n"
        "4. **Zurücksetzen oder exportieren:** Mit dem Zurücksetzen-Button "
        "beginnst du neu; als CSV lassen sich die experimentellen "
        "Studiendaten herunterladen."
    )
    with st.expander("Drei angeleitete Experimente zum Ausprobieren"):
        for title, description in LAB_EXPERIMENTS:
            st.markdown(f"**{title}**")
            st.markdown(description)
    st.info(
        "**Denkfrage:** Was beeinflusst den gepoolten Effekt stärker – "
        "die Richtung einer Studie, ihre Stichprobengrösse oder die "
        "Unterschiede zwischen allen Studien? Probiere die Varianten "
        "nacheinander aus."
    )
    st.caption(
        "Die Änderungen gelten nur für diese Browsersitzung und die gewählte "
        "Version. Ein Wechsel von V0–V4 in der Seitenleiste startet das Labor "
        "für die neue Version erneut."
    )
    if "exp_baseline" not in st.session_state or st.session_state.exp_baseline != version:
        st.session_state.exp_baseline = version
        st.session_state.exp_frame = snapshot(version)
        st.session_state.editor_revision = st.session_state.get("editor_revision", 0) + 1

    left, right = st.columns([1.4, 1.0], gap="large")
    with left:
        st.markdown("**Bestehende Studien bearbeiten**")
        exp_cols = ["study_id", "author_year", "n_int", "mean_int", "sd_int",
                    "n_ctrl", "mean_ctrl", "sd_ctrl", "rob"]
        with st.form(f"editor_{st.session_state.editor_revision}"):
            edited = st.data_editor(
                st.session_state.exp_frame[exp_cols], num_rows="fixed",
                width="stretch", hide_index=True,
                disabled=["study_id", "author_year"],
                column_config={
                    "study_id": st.column_config.TextColumn("ID"),
                    "author_year": st.column_config.TextColumn("Studie"),
                    "n_int": st.column_config.NumberColumn("n Int.", min_value=2, step=1, format="%d"),
                    "n_ctrl": st.column_config.NumberColumn("n Kontr.", min_value=2, step=1, format="%d"),
                    "mean_int": st.column_config.NumberColumn("MW Int.", format="%.2f"),
                    "mean_ctrl": st.column_config.NumberColumn("MW Kontr.", format="%.2f"),
                    "sd_int": st.column_config.NumberColumn("SD Int.", min_value=0.01, format="%.2f"),
                    "sd_ctrl": st.column_config.NumberColumn("SD Kontr.", min_value=0.01, format="%.2f"),
                    "rob": st.column_config.SelectboxColumn("Risk of Bias", options=["Low", "Some concerns", "High"]),
                },
                key=f"editor_key_{st.session_state.editor_revision}",
            )
            saved = st.form_submit_button("Änderungen übernehmen und neu berechnen", type="primary")
        if saved:
            try:
                validate_studies(edited)
                # Preserve metadata, keep editable values as current exercise state.
                current = st.session_state.exp_frame.copy().set_index("study_id")
                for _, row in edited.iterrows():
                    sid = row["study_id"]
                    for col in exp_cols:
                        current.at[sid, col] = row[col]
                st.session_state.exp_frame = current.reset_index()
                st.success("Die experimentelle Metaanalyse wurde aktualisiert.")
            except (ValueError, TypeError) as exc:
                st.error(f"Ungültige Studiendaten: {exc}")
        if st.button("Experiment auf Originalversion zurücksetzen"):
            st.session_state.exp_frame = snapshot(version)
            st.session_state.editor_revision += 1
            st.rerun()

        with st.expander("Neue synthetische Studie hinzufügen"):
            with st.form("add_exp_study"):
                study_name = st.text_input("Studienbezeichnung", value="Neue Studie (Simulation)")
                n1, n2 = st.columns(2)
                n_int = n1.number_input("N Intervention", min_value=2, value=200, step=10)
                n_ctrl = n2.number_input("N Kontrolle", min_value=2, value=200, step=10)
                x1, x2 = st.columns(2)
                mean_int = x1.number_input("ISI-Mittelwert Intervention", value=11.0, step=.1)
                mean_ctrl = x2.number_input("ISI-Mittelwert Kontrolle", value=12.0, step=.1)
                x3, x4 = st.columns(2)
                sd_int = x3.number_input("SD Intervention", min_value=.01, value=4.0, step=.1)
                sd_ctrl = x4.number_input("SD Kontrolle", min_value=.01, value=4.0, step=.1)
                rob = st.selectbox("Risk of Bias", ["Low", "Some concerns", "High"])
                submitted = st.form_submit_button("Studie einschliessen")
            if submitted:
                existing = st.session_state.exp_frame
                suffix = 1
                while f"EXP-{suffix:03d}" in existing["study_id"].astype(str).values:
                    suffix += 1
                row = {col: np.nan for col in existing.columns}
                row.update(study_id=f"EXP-{suffix:03d}", author_year=study_name,
                           n_int=int(n_int), n_ctrl=int(n_ctrl), mean_int=float(mean_int),
                           mean_ctrl=float(mean_ctrl), sd_int=float(sd_int),
                           sd_ctrl=float(sd_ctrl), rob=rob, phase=4)
                try:
                    st.session_state.exp_frame = validate_studies(
                        pd.concat([existing, pd.DataFrame([row])], ignore_index=True))
                    st.session_state.editor_revision += 1
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))

    with right:
        try:
            experiment = validate_studies(st.session_state.exp_frame)
            result = fit_random_effects(experiment)
            st.markdown("**Neuberechnetes Ergebnis**")
            st.caption(
                f"Vergleich mit {version}: Die Ausgangsschätzung beträgt "
                f"{fit.mean:+.2f} ISI-Punkte; die Anzeige darunter verwendet "
                "ausschliesslich deine experimentellen Daten."
            )
            a,b=st.columns(2)
            a.metric("MD experimentell", f"{result.mean:+.2f}",
                     delta=f"{result.mean-fit.mean:+.2f} vs. {version}", delta_color="off")
            b.metric("I²", f"{result.i2:.1f} %")
            st.write(f"95%-KI **[{result.ci_low:+.2f}; {result.ci_high:+.2f}]** · {result.k} Studien")
            st.caption(
                "Die Delta-Anzeige zeigt **neue MD minus ursprüngliche MD**. "
                "Ein positiver Wert bedeutet eine Verschiebung Richtung weniger "
                "negativer Effekte; ein negativer Wert eine Verschiebung zu "
                "stärker negativem Effekt."
            )
            st.plotly_chart(forest_plot(experiment, result, title="Experimenteller Forest Plot"),
                            width="stretch", key="experimental_forest")
            out = calculate_effects(experiment)
            st.download_button("Experimentelle Studiendaten als CSV",
                               data=out.to_csv(index=False).encode("utf-8-sig"),
                               file_name="somnitech_experiment.csv", mime="text/csv")
        except ValueError as exc:
            st.error(f"Keine Analyse möglich: {exc}")
    st.caption("Das Experimentierlabor simuliert Veränderungen der Aggregatdaten, nicht die Generierung einzelner Patientendaten.")

with section_methods:
    st.subheader("Frequentistisch vs. bayesianisch · Was sagen die Ergebnisse aus?")
    st.write(
        f"Beide Analysen verwenden **dieselben {fit.k} synthetischen Studien "
        f"aus {version}** und ein Random-Effects-Modell. "
        "Sie beantworten verwandte Fragen, aber die Unsicherheit wird "
        "unterschiedlich ausgedrückt. Wechsle links die Review version und "
        "beobachte, wie sich beide Ergebnisse mit neuer Evidenz verändern."
    )
    concept_f, concept_b = st.columns(2, gap="large")
    with concept_f:
        st.markdown("**Frequentistische Perspektive**")
        st.write(
            "Wir schätzen eine durchschnittliche Mittelwertdifferenz und "
            "konstruieren ein 95%-**Konfidenzintervall**. Würde man das "
            "Untersuchungsverfahren sehr oft unter gleichen Bedingungen "
            "wiederholen, würden die so gebildeten Intervalle den wahren "
            "Parameter in etwa 95 % der Fälle überdecken. Das konkrete "
            "Intervall liefert **keine direkte 95%-Wahrscheinlichkeit** für μ."
        )
    with concept_b:
        st.markdown("**Bayesianische Perspektive**")
        st.write(
            "Wir legen vor der Analyse **Priors** für den Gesamteffekt (μ) "
            "und die Heterogenität (τ) fest. Zusammen mit den Studiendaten "
            "entsteht eine **Posteriorverteilung**. Ein 95%-" 
            "**Kredibilitätsintervall** enthält unter Modell und Priors "
            "95 % der posterioren Wahrscheinlichkeit für μ."
        )
    st.info(
        "**Die Schlüsselidee beim Updating:** Frequentistisch wird das Modell "
        "mit dem aktuellen Datenstand erneut geschätzt. Bayesianisch aktualisieren "
        "die Daten die vorherigen Annahmen. In dieser App wird für jede "
        "Version derselbe ursprüngliche Prior mit dem vollständigen gültigen "
        "Datensatz kombiniert – alte Studien werden dadurch nicht doppelt gezählt."
    )
    st.markdown("#### 1 · Priors festlegen")
    st.write(
        "Ein Prior ist die Verteilung, die wir **vor Betrachtung dieser Daten** "
        "für einen Parameter annehmen. Eine grössere Prior-Standardabweichung "
        "lässt einen breiteren Bereich von Werten zu. Probiere aus, ob sich "
        "die Schlussfolgerung bei einem engeren oder breiteren Prior verändert."
    )
    col_prior1, col_prior2 = st.columns(2)
    mu_prior_sd = col_prior1.select_slider("Prior-Standardabweichung für μ (Normal(0, SD))",
                                           options=[1.0, 2.0, 5.0, 10.0], value=5.0,
                                           help="μ ist die wahre durchschnittliche MD. Ein Prior Normal(0, 5²) lässt stärkere Abweichungen von Null eher zu als Normal(0, 1²).")
    tau_prior_sd = col_prior2.select_slider("Prior-Standardabweichung für τ (Half-Normal(SD))",
                                            options=[0.5, 1.0, 2.0, 4.0], value=2.0,
                                            help="τ beschreibt die Streuung der wahren Studieneffekte (ISI-Punkte). Eine Half-Normal-Verteilung erlaubt nur τ ≥ 0.")
    st.caption(
        f"Aktuell: μ ~ Normal(0, {mu_prior_sd:g}²); "
        f"τ ~ Half-Normal({tau_prior_sd:g}). "
        "Die Priors gewichten keine einzelnen Studien nach ihrer Qualität."
    )
    st.markdown("#### 2 · Schätzungen und Intervalle vergleichen")
    bayes = fit_bayesian(frame, threshold=threshold,
                         mu_prior_sd=mu_prior_sd, tau_prior_sd=tau_prior_sd)
    fa, fb = st.columns(2, gap="large")
    with fa:
        st.markdown("#### Frequentistisch")
        st.metric("Gepoolte MD (REML)", f"{fit.mean:+.2f}")
        st.write(f"95% confidence interval: **[{fit.ci_low:+.2f}; {fit.ci_high:+.2f}]**")
        st.write(f"Zwischenstudienvarianz τ²: **{fit.tau2:.3f}**")
        st.caption("95%-KI: Interpretation über die langfristige Überdeckungsrate des Verfahrens, nicht als Posteriorwahrscheinlichkeit.")
    with fb:
        st.markdown("#### Bayesianisch")
        st.metric("Posterior-Mittelwert μ", f"{bayes.mean:+.2f}")
        st.write(f"95%-Kredibilitätsintervall: **[{bayes.ci_low:+.2f}; {bayes.ci_high:+.2f}]**")
        st.write(f"P(μ < 0 | Daten): **{bayes.p_benefit:.1%}**")
        st.write(f"P(μ < {threshold:g} | Daten): **{bayes.p_relevant:.1%}**")
        st.caption(
            f"Posterior-Mittelwert τ: {bayes.tau_mean:.2f}. "
            "P(μ < 0) ist die bedingte Wahrscheinlichkeit eines "
            "durchschnittlichen Vorteils der App. P(μ < Schwelle) ist die "
            "Wahrscheinlichkeit eines mindestens so grossen, hier nur "
            "illustrativ als relevant definierten Vorteils."
        )
    st.markdown("#### 3 · Die Posteriorverteilung lesen")
    st.write(
        "Die Kurve zeigt, welche Werte für den **durchschnittlichen wahren "
        "Effekt μ** nach Daten und Priors mehr oder weniger plausibel sind. "
        "Die **Fläche links von Null** entspricht P(μ < 0). "
        f"Die **Fläche links von {threshold:g}** entspricht P(μ < {threshold:g}). "
        "Das grün markierte Intervall enthält 95 % der posterioren Masse. "
        "Ändere einen Prior und beobachte, ob sich diese Bereiche verschieben."
    )
    density = bayesian_density(frame, mu_prior_sd=mu_prior_sd, tau_prior_sd=tau_prior_sd)
    st.plotly_chart(posterior_plot(density, threshold, bayes.ci_low, bayes.ci_high),
                    width="stretch", key="bayes_posterior")
    st.markdown("#### 4 · Wie ändert sich die Wahrscheinlichkeit über die Versionen?")
    st.write(
        "Die Kurve zeigt für jede Review version die bayesianische "
        "Wahrscheinlichkeit, dass die wahre durchschnittliche MD unter "
        "der gewählten Relevanzschwelle liegt. Sie ist **keine** "
        "Wahrscheinlichkeit, dass eine einzelne Studie signifikant wird. "
        "Beachte besonders den Übergang von V1 zu V2 und die Korrektur in V4."
    )
    b_history = []
    for v in VERSIONS:
        b = fit_bayesian(snapshot(v), threshold=threshold,
                         mu_prior_sd=mu_prior_sd, tau_prior_sd=tau_prior_sd)
        b_history.append(dict(Version=v, Wahrscheinlichkeit=b.p_relevant, P_Nutzen=b.p_benefit))
    b_table = pd.DataFrame(b_history)
    fig = go.Figure(go.Scatter(x=b_table["Version"], y=100*b_table["Wahrscheinlichkeit"],
                               mode="lines+markers", marker=dict(size=11, color=BLUE),
                               line=dict(width=3, color=BLUE)))
    fig.update_layout(template="plotly_white", height=290,
                      xaxis_title="Review version", yaxis_title=f"P(μ < {threshold:g}) in %",
                      yaxis_range=[0,100], margin=dict(l=10,r=10,t=25,b=40))
    st.plotly_chart(fig, width="stretch", key="bayes_history")
    with st.expander("Häufige Verständnisfragen: KI, Kredibilitätsintervall, Priors und Updating"):
        st.markdown(
            "**Warum unterscheiden sich die Punktschätzungen?** Das "
            "frequentistische Modell bestimmt die Heterogenität über REML; "
            "die bayesianische Analyse integriert die Unsicherheit über τ "
            "und bezieht die Priors für μ und τ ein.\n\n"
            "**Kann ich sagen, die Intervention wirkt mit 95 % Wahrscheinlichkeit?** "
            "Nur eine klar definierte bayesianische Aussage wie "
            "P(μ < 0 | Daten, Modell, Priors) ist in dieser Weise "
            "interpretierbar. Ein frequentistisches 95%-KI "
            "bedeutet nicht P(μ im Intervall) = 95 %.\n\n"
            "**Sind engere Priors immer besser?** Nein. Zu enge Priors "
            "können die Ergebnisse unverhältnismässig beeinflussen; "
            "deshalb lohnt sich eine Prior-Sensitivitätsanalyse.\n\n"
            "**Was bedeutet 'living' für die Statistik?** Neue oder "
            "korrigierte Daten verändern den jeweiligen Snapshot. "
            "Die wiederholten frequentistischen Intervalle dieser Demo "
            "sind nicht sequenziell adjustiert; bayesianische "
            "Wahrscheinlichkeiten sind modellbedingt und keine Garantie "
            "gegen Bias oder selektive Berichterstattung."
        )
    st.info("Jeder Snapshot wird mit demselben ursprünglichen Prior und nur den jeweils gültigen Daten berechnet. Eine Datenkorrektur ersetzt alte Werte vollständig; sie wird nicht als zusätzliche Studie gezählt.")
    with st.expander("Statistisches Modell und Einschränkungen"):
        st.latex(r"y_i\mid \mu,\tau \sim \mathcal{N}(\mu,\,v_i+\tau^2)")
        st.latex(r"\mu \sim \mathcal{N}(0,\sigma_{\mu}^2),\quad \tau\sim\operatorname{HalfNormal}(\sigma_{\tau})")
        st.write("Der bayesianische Posterior wird ohne MCMC durch analytische Integration über μ und numerische Integration über τ berechnet. Die modellierten Studien sind unabhängig, und die berichteten Stichprobenvarianzen werden als bekannt behandelt.")
        st.write("Die festgelegte Schwelle von zwei ISI-Punkten ist rein didaktisch. Weder ein grosses Posterior-P noch ein p-Wert kann unzureichende interne Validität, Publikationsbias oder Modellfehlanpassung ausgleichen.")

with section_workflow:
    st.subheader("Vom Such-Alert zur neuen Review version")
    st.write("Ein Living Review ist mehr als ein wiederholt ausgeführtes Metaanalyse-Skript: Monitoring, Screening, Datenprüfung, Reanalyse, Interpretation und Veröffentlichung sind getrennte Vorgänge.")
    st.markdown("**Prozessmodell**")
    p1,p2,p3,p4,p5 = st.columns(5)
    for column, txt in zip((p1,p2,p3,p4,p5),
                           ("1 · Surveillance", "2 · Screening", "3 · Extraction & QA",
                            "4 · Reanalysis", "5 · Publication")):
        column.info(txt)
    events = load_events()
    allowed = list(VERSIONS[:VERSIONS.index(version)+1])
    events = events[events["version"].isin(allowed)]
    st.markdown(f"**Ereignisprotokoll bis {version}**")
    st.dataframe(events.rename(columns={"version":"Version", "event_date":"Datum",
                                             "event_type":"Ereignis", "study_id":"Studie", "details":"Beschreibung"}),
                 hide_index=True, width="stretch")
    st.markdown("**Studiendaten dieses Snapshots**")
    st.dataframe(effects[["study_id", "author_year", "n_int", "n_ctrl", "mean_int",
                          "mean_ctrl", "sd_int", "sd_ctrl", "yi", "vi", "rob"]],
                 hide_index=True, width="stretch")
    a,b,c = st.columns(3)
    a.download_button("Snapshot CSV", data=frame.to_csv(index=False).encode("utf-8-sig"),
                      file_name=f"somnitech_{version}_studies.csv", mime="text/csv")
    b.download_button("Event-Log CSV", data=events.to_csv(index=False).encode("utf-8-sig"),
                      file_name=f"somnitech_{version}_events.csv", mime="text/csv")
    c.download_button("Versionsvergleich CSV", data=history.to_csv(index=False).encode("utf-8-sig"),
                      file_name="somnitech_versions.csv", mime="text/csv")
    st.divider()
    st.markdown("**Worüber sollte der Review informieren?**")
    st.write("Die Simulation unterscheidet drei Entscheidungen: **(1)** Muss nach dem Monitoring neu gescreent werden? **(2)** Erfordern neue oder korrigierte Daten eine Reanalyse? **(3)** Rechtfertigt die Neubewertung eine neue publizierte Review version?")
    st.caption("Die im Ereignisprotokoll dokumentierten Updates und Veröffentlichungen sind Teil der Simulation. Die App führt keine echte PubMed-Suche durch und ermittelt keine reale klinische Evidenz.")
