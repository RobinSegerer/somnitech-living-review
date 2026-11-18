# SomniTech · Living Evidence Lab

An interactive **Streamlit teaching application** for demonstrating how a living systematic review changes when new trials appear, findings conflict, or previously extracted data are corrected. The app presents the same simulated evidence through frequentist and Bayesian random-effects meta-analysis.

> **Entirely fictional evidence.** The SomniTech app, study names, results, and publication dates (including dates in 2027) are invented for teaching. The findings are **not clinical evidence** and must not be used for medical decisions.

## Run the application locally

Requirements: Python 3.11–3.13, a terminal, and internet access for installing dependencies.

```bash
git clone https://github.com/RobinSegerer/somnitech-living-review.git
cd somnitech-living-review
python -m venv .venv
source .venv/bin/activate         # macOS / Linux
# Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit normally opens the app at `http://localhost:8501`. Stop the server with `Ctrl+C`.

## What you can do

### 01 · Living Review

Start with the fictional PICO question: does the SomniTech digital insomnia intervention lower the **Insomnia Severity Index (ISI)** at eight weeks compared with general sleep-hygiene advice?

Select **V0–V4** in the sidebar. Observe the pooled mean difference (MD), its 95% confidence interval (CI), between-study heterogeneity, and the forest plot. Newly included studies are highlighted. Each version offers a guiding question, an interpretation of the results, and a suggested activity. A low-risk-of-bias sensitivity analysis is available.

The effect is the **difference between follow-up means**, not a within-person change from baseline. A negative MD favours SomniTech.

### 02 · Experiment Lab

Edit sample sizes, means, standard deviations, or risk-of-bias labels, then select **Apply changes and recalculate**. Alternatively, add a synthetic RCT, inspect the revised forest plot, reset to the original version, or export the experimental dataset.

Three guided activities show how to:

1. Add a large null-effect trial.
2. Change the sample size without changing the study effect.
3. Correct the data in an existing trial.

These changes are session-local: they do not modify the canonical CSV files or the historical review versions.

### 03 · Frequentist vs. Bayesian

Use **the same study data** for both approaches and compare their interpretation of uncertainty.

- **Frequentist:** REML random-effects meta-analysis and modified Hartung–Knapp 95% confidence intervals.
- **Bayesian:** a hierarchical normal–normal model with adjustable priors for the pooled effect and heterogeneity, 95% credible intervals, and probabilities of a benefit or of exceeding an illustrative effect threshold.

The page explains priors, posterior distributions, the difference between confidence and credible intervals, and how results evolve over versions. Old studies are **not counted twice**: every Bayesian snapshot uses the original prior and the full valid evidence available at that time.

#### Interactive Bayesian updating (new)

A guided selector for **V0–V4** shows how the posterior from the preceding review version becomes the prior for the next evidence update. A Plotly chart includes:

- **Blue:** prior to this step (the original prior for V0, otherwise the previous version's posterior marginal for μ).
- **Orange dotted:** a peak-scaled **relative updating factor** for μ, indicating which values gain or lose support. This is **not a probability density** and uses its own right-hand vertical axis. It accounts for uncertainty about heterogeneity τ through the previous joint model.
- **Green:** posterior after including the new evidence.

A probability table compares **P(μ < 0)** and **P(μ < the chosen illustrative threshold)** before and after each step. V1–V3 incorporate newly added trials. V4 demonstrates that a correction is different: the incorrect S06 contribution must be **removed and replaced**, rather than adding the corrected data as another independent trial.

**Why full reanalysis is still correct:** With unchanged model and priors, recomputing each complete evidence snapshot under the same original joint prior is algebraically equivalent to sequentially updating the **joint** p(μ, τ) posterior. It is generally *not* sufficient to carry forward only the marginal distribution of μ when τ is also uncertain.


### 04 · Workflow & Data

Explore the steps of a living review—surveillance, screening, extraction and quality assurance, reanalysis, and publication. Review the synthetic event log and export versioned datasets, event records, and pooled estimates.

## Suggested guided tour

1. Open **Living Review**. Read the fictional case and start at **V0**. Move to V1–V4 and watch how effects, precision, and heterogeneity change, especially after the correction in V4.
2. Open **Experiment Lab**. Follow the three suggested experiments, starting with a large study reporting no effect.
3. Open **Frequentist vs. Bayesian**. Adjust one prior at a time. Compare the meaning of the 95% CI, 95% credible interval, and the probability `P(μ < −2)`.
4. Open **Workflow & Data**. Check which simulated events triggered reanalysis or a new published version.

## Fictional review versions

| Version | Year | Studies | Narrative |
| --- | --- | ---: | --- |
| V0 | 2024 | 4 | Initial review: promising but uncertain effect |
| V1 | 2025 | 6 | A large, favourable RCT strengthens the pooled result |
| V2 | 2026 | 9 | Three large trials report smaller or opposing effects |
| V3 | 2027 | 13 | Additional evidence, with substantial heterogeneity |
| V4 | 2027 | 13 | Correction of the eight-week endpoint of S06, with no new study |

`data/studies.csv` contains the original values and separately stored revised values. `src/data.py` applies the S06 correction **only from V4 onward**, preserving reproducible historical snapshots.

## Analysis methods

- **Outcome:** ISI at eight weeks, with lower scores indicating fewer insomnia symptoms.
- **Effect:** raw mean difference, `mean_int − mean_ctrl`, in ISI points.
- **Within-study variance:** `sd_int²/n_int + sd_ctrl²/n_ctrl`.
- **Frequentist model:** normal–normal random effects; between-study variance `τ²` estimated using REML. The modified Hartung–Knapp interval uses `max(1, Q_RE/(k−1))` for variance inflation and the `t(k−1)` quantile. `I²` is calculated from Cochran's Q.
- **Bayesian model:** `y_i | μ, τ ~ Normal(μ, v_i + τ²)`; `μ ~ Normal(0, σμ²)`, and `τ ~ HalfNormal(στ)`. The posterior integrates analytically over `μ` and numerically over `τ` on a fine grid. Defaults: `σμ = 5` and `στ = 2` ISI points.
- **Illustrative threshold:** `MD < −2` ISI points. **This is not a validated minimal clinically important difference (MCID).**

The statistical engine in `src/analysis.py` is independent of Streamlit and is covered by automated regression tests. An optional R script, `validation/metafor_crosscheck.R`, supports external verification of the frequentist estimates with `metafor`.

**Methodological caveats:** Repeated confidence intervals shown after successive updates are not automatically valid sequential frequentist inference. Bayesian updating does not remove publication bias, selection bias, or model misspecification. These simulations are not intended for making real clinical decisions.

## Repository structure

```text
app.py                       Streamlit entry point and navigation
data/studies.csv             Versioned fictional trial data
data/events.csv              Monitoring and review update event log
src/data.py                  Historical review snapshots
src/analysis.py              Frequentist and Bayesian models
src/plots.py                 Forest, history, and posterior charts
src/explanations.py          Educational guidance and laboratory exercises
tests/                       Automated regression and guidance tests
validation/                  Optional R/metafor comparison
requirements.txt             Application dependencies
requirements-dev.txt         Test dependencies
.github/workflows/tests.yml  Continuous integration
```

Run the tests:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## Deploy on Streamlit Community Cloud

Choose **Deploy an app**, select repository `RobinSegerer/somnitech-living-review`, branch `main`, and set **Main file path** to `app.py`. No secrets are required. Streamlit will install the packages listed in `requirements.txt`.

## License

MIT License. See [LICENSE](LICENSE).
