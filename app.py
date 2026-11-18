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
st.markdown('<p class="hero-sub">Wie verändern neue RCTs, widersprüchliche Ergebnisse und Datenkorrekturen eine laufend aktualisierte Metaanalyse?</p>', unsafe_allow_html=True)
st.markdown('<div class="notice"><b>Vollständig fiktive Evidenz:</b> SomniTech und sämtliche Studien sind erfunden. Die Daten, Studiennamen und Publikationsdaten (einschliesslich 2027) sind ausschliesslich für die Lehre konstruiert. Keine klinischen Schlussfolgerungen ableiten.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Review steuern")
    version = st.select_slider("Review-Version", options=list(VERSIONS), value="V0",
                               help="Versionen repräsentieren historisch verfügbare Daten; V4 enthält eine Korrektur ohne neue Studie.")
    meta = VERSION_METADATA[version]
    st.caption(f"{meta['date']} · {meta['type']}")
    st.markdown(f"**{meta['title']}**")
    st.write(meta["description"])
    st.divider()
    threshold = st.number_input(
        "Illustrative Relevanzschwelle (ISI-Punkte)", value=-2.0,
        min_value=-10.0, max_value=-0.1, step=0.5,
        help="Didaktisch gesetzte Schwelle, keine empirisch validierte minimale klinisch wichtige Differenz.",
    )
    st.caption("Negative Mittelwertdifferenzen sprechen für SomniTech. Ein Wert von −2 würde zwei ISI-Punkte geringere Symptomschwere bedeuten.")
    st.divider()
    st.markdown("**Analytische Konvention**")
    st.caption("Random-Effects-Modell; REML für τ²; modifizierte Hartung-Knapp-Intervalle mit t(k−1). I² aus Cochran Q.")
    st.caption("Bayes: normal-normal-Hierarchiemodell; Priors im Methodenvergleich konfigurierbar.")

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
    "01 · Living Review", "02 · Experimentierlabor",
    "03 · Frequentistisch vs. Bayes", "04 · Workflow & Daten",
])

with section_review:
    st.subheader("Der fiktive Fall: Wirksamkeit der Schlaf-App SomniTech")
    st.write(
        "Ein Forschungsteam möchte wissen, ob eine **fiktive digitale Schlafintervention** "
        "bei Erwachsenen mit Insomnie besser wirkt als eine allgemeine "
        "**Schlafhygiene-Beratung**. Der Review berücksichtigt ausschliesslich "
        "randomisierte kontrollierte Studien (RCTs). Im Laufe der Zeit erscheinen "
        "neue Studien; ausserdem wird später eine fehlerhafte Datenextraktion korrigiert. "
        "Du verfolgst, wie sich dadurch das Wissen verändert."
    )
    st.markdown("**Die Forschungsfrage (PICO)**")
    p1, p2 = st.columns(2, gap="large")
    with p1:
        st.markdown("**P – Population:** Erwachsene mit Insomnie  \n**I – Intervention:** digitale App *SomniTech*")
    with p2:
        st.markdown("**C – Vergleich:** allgemeine Schlafhygiene-Beratung  \n**O – Outcome:** Insomnia Severity Index (ISI) nach acht Wochen")
    st.markdown(
        "**Was wird verglichen?** Der ISI umfasst 0–28 Punkte; niedrigere Werte "
        "bedeuten weniger Insomniesymptome. Wir analysieren die **Mittelwertdifferenz "
        "der ISI-Endwerte nach acht Wochen** (Intervention minus Kontrolle), "
        "nicht individuelle Veränderungen seit Studienbeginn. "
        "Eine MD von **−2** bedeutet beispielsweise, dass die Interventionsgruppe "
        "im Durchschnitt zwei ISI-Punkte niedriger liegt. Die hier einstellbare "
        "Schwelle von −2 ist nur eine **didaktische Annahme**, keine validierte "
        "Grenze klinisch relevanter Veränderungen."
    )
    st.info(
        "**So gehst du vor:** Wähle links zuerst **V0** und betrachte den Forest Plot. "
        "Wechsle dann zu **V1–V4**: Was passiert mit der Gesamtschätzung, "
        "dem Konfidenzintervall und der Heterogenität? Die orangefarbenen Studien "
        "sind neu hinzugekommen. Mit der Schwelle in der Seitenleiste kannst du "
        "eine angenommene klinische Relevanzgrenze verschieben."
    )
    with st.expander("Warum ist das ein *Living* Review und nicht nur eine Metaanalyse?"):
        st.write(
            "Die initiale Suche bildet V0. Danach werden neu erschienene Studien "
            "durch regelmässige Suchläufe und Alerts entdeckt, gescreent, geprüft "
            "und in die Analyse aufgenommen. Jede Version bewahrt einen eigenen "
            "Datenstand. V4 demonstriert, dass auch **Korrekturen alter Daten** "
            "eine Reanalyse auslösen können. Ein Suchlauf ohne neue Studien "
            "führt hingegen nicht zwingend zu einer neuen Publikation."
        )
    st.divider()
    st.subheader("Evidenzstand · " + version)
    guide = VERSION_GUIDE[version]
    st.markdown(f"**Leitfrage: {guide['question']}**")
    st.caption("Die folgenden Zahlen beziehen sich nur auf den ausgewählten, fiktiven Datenstand.")
    a,b,c,d = st.columns(4)
    a.metric("Studien", f"{fit.k}")
    b.metric("Teilnehmende", f"{fit.n_total:,}".replace(",", "’"))
    c.metric("Gepoolte MD", f"{fit.mean:+.2f}")
    d.metric("I² (Q-basiert)", f"{fit.i2:.1f} %")
    st.caption(f"95%-Konfidenzintervall: [{fit.ci_low:+.2f}; {fit.ci_high:+.2f}] ISI-Punkte · τ² = {fit.tau2:.3f} · Q-p = {fit.q_pvalue:.3g}")
    st.markdown(
        "**Zahlen lesen:** Die **MD** ist der durchschnittliche Unterschied "
        "zwischen den Gruppen (negativ = günstiger für SomniTech). Das "
        "**95%-KI** quantifiziert die statistische Unsicherheit. "
        "**I²** beschreibt den Anteil beobachteter Variabilität der "
        "Studieneffekte, der über die Stichprobenstreuung hinausgeht "
        "(Q-basiert); es ist **kein Qualitätswert**."
    )

    col_forest, col_side = st.columns([2.25, 1.0], gap="large")
    with col_forest:
        st.plotly_chart(forest_plot(frame, fit, title=f"Forest Plot · {version}",
                                    highlight_phase=int(version[-1]) if version in ("V1", "V2", "V3") else None),
                        width="stretch", key=f"forest_{version}")
        with st.expander("Lesehilfe: Was bedeuten Quadrate, Linien und Diamond?"):
            st.markdown(
                "- **Quadrat:** geschätzte Mittelwertdifferenz einer Studie; "
                "grössere Quadrate zeigen ein höheres *Random-Effects-Gewicht* an.\n"
                "- **Horizontale Linie:** 95%-Konfidenzintervall dieser Studie.\n"
                "- **Grüner Diamond unten:** gepoolte Mittelwertdifferenz; seine "
                "Breite steht für das 95%-KI des Gesamteffekts.\n"
                "- **Gestrichelte Linie bei 0:** kein durchschnittlicher Unterschied "
                "zwischen Intervention und Kontrolle.\n"
                "- **Orange:** in V1–V3 neu eingeschlossene Studien. V4 korrigiert "
                "dagegen die bereits vorhandene Studie S06.\n\n"
                "**Maus-Tipp:** Fahre über ein Quadrat, um die einzelne MD, "
                "ihr Konfidenzintervall und das Gewicht zu sehen."
            )
    with col_side:
        st.markdown("#### Was hat sich geändert?")
        st.info(meta["description"])
        if version != "V0":
            prev_name = VERSIONS[VERSIONS.index(version) - 1]
            prev = history.iloc[VERSIONS.index(version)-1]
            st.metric("Veränderung der MD", f"{fit.mean - prev['mean']:+.2f}",
                      help=f"Aktueller Effekt minus Effekt in {prev_name}.")
            st.metric("Veränderung der Studienzahl", f"{fit.k-int(prev['k']):+d}")
        st.write("**Was zeigt diese Version?**")
        st.write(guide["observation"])
        st.write("**Worauf solltest du achten?**")
        st.caption(guide["try_this"])
        st.write("**Statistische Einordnung**")
        if fit.ci_high < 0:
            st.success("Das 95%-KI liegt unter Null. Im gewählten Modell ist die mittlere MD statistisch von Null unterscheidbar.")
        else:
            st.warning("Das 95%-KI umfasst Null. Die Evidenz ist mit einem Null-Effekt vereinbar.")
        if fit.i2 > 75:
            st.warning("Sehr hohe Heterogenität: Die gepoolte MD allein beschreibt die unterschiedlichen Studienresultate nur eingeschränkt.")
        st.caption("Statistische Signifikanz beweist weder klinische Relevanz noch hohe Evidenzgewissheit.")

    st.markdown("#### Wie hat sich die Evidenz verändert?")
    st.plotly_chart(history_plot(history, selected=version, threshold=threshold),
                    width="stretch", key="history_plot")
    st.caption(
        "Jede Zeile zeigt den Gesamteffekt einer eigenen Review-Version mit "
        "95%-KI. Die gestrichelte Nulllinie steht für keinen Unterschied; "
        "die gepunktete Linie markiert die links festgelegte didaktische "
        "Relevanzschwelle. Die aktuell gewählte Version ist orange hervorgehoben."
    )
    st.info(
        "**Merke:** Eine spätere Version kann die frühere Schlussfolgerung "
        "bestätigen, abschwächen oder umkehren. Die wiederholten "
        "Konfidenzintervalle sind hier nicht für sequenzielles Testen adjustiert."
    )
    table_hist = history.copy()
    table_hist["95%-KI"] = table_hist.apply(lambda r:f"[{r.ci_low:.2f}; {r.ci_high:.2f}]", axis=1)
    table_hist["MD"] = table_hist["mean"].round(2)
    table_hist["I² (%)"] = table_hist["i2"].round(1)
    st.dataframe(table_hist[["version", "date", "k", "MD", "95%-KI", "I² (%)"]],
                 hide_index=True, width="stretch")

    st.markdown("#### Risk-of-Bias-Sensitivität")
    st.write(
        "Nicht jede Studie ist methodisch gleich vertrauenswürdig. Hier wird "
        "die Metaanalyse versuchsweise nur mit Studien gerechnet, die im "
        "**simulierten RoB-Label ‚Low‘** tragen. Vergleiche das Ergebnis mit "
        "der Gesamtanalyse. Dies ist eine vereinfachte Sensitivitätsanalyse, "
        "keine vollständige RoB-2- oder GRADE-Bewertung."
    )
    lowrisk = frame.loc[frame["rob"].eq("Low")].copy()
    if len(lowrisk) >= 2:
        lower_fit = fit_random_effects(lowrisk)
        a,b = st.columns(2)
        a.metric("Nur RoB = Low", f"{lower_fit.mean:+.2f}",
                 help=f"{len(lowrisk)} von {len(frame)} Studien.")
        b.metric("95%-KI dieser Teilmenge", f"[{lower_fit.ci_low:+.2f}; {lower_fit.ci_high:+.2f}]")
        st.caption("Explorative Sensitivitätsanalyse. Keine automatische Evidenzbewertung nach GRADE; einzelne RoB-2-Domänen werden nicht simuliert.")
    else:
        st.caption("Für diese Version sind weniger als zwei Studien mit niedrigem Risk of Bias vorhanden.")

with section_lab:
    st.subheader("Experimentierlabor · Wie beeinflussen einzelne Studien die Synthese?")
    st.write(
        "Hier kannst du die Metaanalyse selbst verändern. Du arbeitest mit den "
        f"**{fit.k} Studien aus {version}** als Startpunkt. Das Labor ist eine "
        "**Was-wäre-wenn-Simulation**: Es verändert keine publizierten "
        "Originaldaten und legt keine neuen Review-Versionen an."
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
        "unterschiedlich ausgedrückt. Wechsle links die Review-Version und "
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
        st.write(f"95%-Konfidenzintervall: **[{fit.ci_low:+.2f}; {fit.ci_high:+.2f}]**")
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
        "Die Kurve zeigt für jede Review-Version die bayesianische "
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
                      xaxis_title="Review-Version", yaxis_title=f"P(μ < {threshold:g}) in %",
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
    st.subheader("Vom Such-Alert zur neuen Review-Version")
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
    st.write("Die Simulation unterscheidet drei Entscheidungen: **(1)** Muss nach dem Monitoring neu gescreent werden? **(2)** Erfordern neue oder korrigierte Daten eine Reanalyse? **(3)** Rechtfertigt die Neubewertung eine neue publizierte Review-Version?")
    st.caption("Die im Ereignisprotokoll dokumentierten Updates und Veröffentlichungen sind Teil der Simulation. Die App führt keine echte PubMed-Suche durch und ermittelt keine reale klinische Evidenz.")
