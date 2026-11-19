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

Select **V0–V4** in the sidebar. The introductory **Case & PICO** section is expanded for V0 and collapsed for later versions to reduce scrolling. Observe the pooled mean difference (MD), its 95% confidence interval (CI), between-study heterogeneity, and the forest plot. The study plots use a common x-axis across all review versions, and their summary metrics display changes since the previous version. New trials appear as orange circles; **S06 in V4 appears as a red diamond with a grey X showing the superseded estimate**. A dotted green line under the pooled diamond shows the approximate 95% prediction interval. Each version prompts students to **predict before revealing** the interpretation. An optional common-effect versus random-effects weight chart shows why large studies lose some relative dominance when heterogeneity rises. A separate toggle overlays **illustrative Bonferroni-adjusted intervals** across five hypothetical looks (0.05/5 per look): this is a teaching sensitivity analysis, not a prospective alpha-spending strategy. A low-risk-of-bias sensitivity analysis is also available.

The effect is the **difference between follow-up means**, not a within-person change from baseline. A negative MD favours SomniTech.

### 02 · Experiment Lab

Edit sample sizes, means, standard deviations, or risk-of-bias labels, then select **Apply changes and recalculate**. Alternatively, add a synthetic RCT, inspect the revised forest plot, reset to the original version, or export the experimental dataset.

Three guided activities show how to:

1. Add a large null-effect trial.
2. Change the sample size without changing the study effect.
3. Correct the data in an existing trial.

These changes are session-local: they do not modify the canonical CSV files or the historical review versions.

**New lab controls:** An **Include** checkbox supports leave-one-out and other exclusion-based sensitivity analyses. Group means are restricted to the valid ISI range (0–28), edited data are validated before reanalysis, and an optional toggle shows how the Bayesian probability P(μ below the threshold) changes. Exported experimental data retain each study's Include status.

**File-drawer exercise:** A button adds **two explicitly hypothetical, unpublished null-effect trials** (500 participants per arm, means 12 vs 12, SD 4). Compare the recalculated results, then Reset. The preset illustrates a potential consequence of missing null studies; it does **not** estimate publication bias or claim that such studies really exist. Clicking it twice does not duplicate trials.

### 03 · Frequentist vs. Bayesian

Use **the same study data** for both approaches and compare their interpretation of uncertainty.

- **Frequentist:** REML random-effects meta-analysis and modified Hartung–Knapp 95% confidence intervals.
- **Bayesian:** a hierarchical normal–normal model with adjustable priors for the pooled effect and heterogeneity, 95% credible intervals, and probabilities of a benefit or of exceeding an illustrative effect threshold.

The page explains priors, posterior distributions, the difference between confidence and credible intervals, and how results evolve over versions. Old studies are **not counted twice**: every Bayesian snapshot uses the original prior and the full valid evidence available at that time.

**Prediction versus estimation:** The frequentist panel also reports an approximate 95% prediction interval (when at least three studies are available). The Bayesian panel reports a 95% posterior-predictive interval for the **true effect in a new setting** θ_new, together with P(θ_new < 0) and P(θ_new below the illustrative threshold). These predictions incorporate between-setting heterogeneity; they are not uncertainty intervals for the pooled mean μ or predictions of a noisy future trial estimate.

#### Interactive Bayesian updating (new)

A guided selector for **V0–V4** shows how the posterior from the preceding review version becomes the prior for the next evidence update. A Plotly chart includes:

- **Blue:** prior to this step (the original prior for V0, otherwise the previous version's posterior marginal for μ).
- **Orange dotted:** a peak-scaled **relative updating factor** for μ, indicating which values gain or lose support. This is **not a probability density** and uses its own right-hand vertical axis. It accounts for uncertainty about heterogeneity τ through the previous joint model.
- **Green:** posterior after including the new evidence.

A probability table compares **P(μ < 0)** and **P(μ < the chosen illustrative threshold)** before and after each step. It now also displays the change in the posterior mean of the **between-study standard deviation τ**, making the joint updating process easier to understand. V1–V3 incorporate newly added trials. V4 demonstrates that a correction is different: the incorrect S06 contribution must be **removed and replaced**, rather than adding the corrected data as another independent trial.

**Why full reanalysis is still correct:** With unchanged model and priors, recomputing each complete evidence snapshot under the same original joint prior is algebraically equivalent to sequentially updating the **joint** p(μ, τ) posterior. It is generally *not* sufficient to carry forward only the marginal distribution of μ when τ is also uncertain.


### 04 · Workflow & Data

Explore the steps of a living review—surveillance, screening, extraction and quality assurance, reanalysis, and publication. Review the synthetic event log and export versioned datasets, event records, and pooled estimates.

**Monitoring workload and governance:** An explicitly fictional, internally consistent PRISMA-like activity table shows identified records, screened abstracts, assessed full texts, and included trials for each simulated search. V0 includes an update with zero eligible new trials; V4 is a data correction without a search. Neither constitutes a real search history or an official PRISMA 2020 flow. Version-specific examples explain why an editorial team might publish an update. A retirement checklist asks about evidence certainty, ongoing registered trials, and decision relevance; their default states are **not assessed/unknown**, never inferred from the meta-analysis. Learners can explore hypothetical editorial decisions, but no actual GRADE or registry checks are performed.

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
data/monitoring.csv          Invented screening and monitoring counts
src/data.py                  Historical review snapshots
src/analysis.py              Frequentist and Bayesian models
src/plots.py                 Forest, history, and posterior charts
src/explanations.py          Educational guidance and laboratory exercises
src/teaching.py              Weight comparisons and fictional didactic scenarios
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
