"""Version-specific reading guides and exercises for the fictional SomniTech study.

Teaching copy is kept separate from the statistical engine so that it can be
reviewed, tested, and translated without affecting the estimates.
"""
from __future__ import annotations

VERSION_GUIDE = {
    "V0": {
        "question": "How convincing is the initial indication of benefit?",
        "observation": (
            "Four small trials yield a negative pooled mean difference (MD), "
            "but the 95% confidence interval includes zero. The data and the "
            "selected model are compatible with no average effect."
        ),
        "try_this": (
            "Examine the confidence intervals of the individual studies in the forest plot. "
            "Why is the pooled result still uncertain although most estimates favour SomniTech?"
        ),
    },
    "V1": {
        "question": "What changes when a large new trial is added?",
        "observation": (
            "Gupta and SOMNI-TRIAL enter the review (highlighted in orange). "
            "The pooled estimate moves further in favour of the intervention, "
            "and the 95% confidence interval no longer includes zero."
        ),
        "try_this": (
            "Hover over SOMNI-TRIAL in the forest plot to inspect its random-effects weight. "
            "With substantial heterogeneity, a large sample size does not necessarily "
            "translate into a dominant weight."
        ),
    },
    "V2": {
        "question": "What happens when new, large trials contradict earlier findings?",
        "observation": (
            "Three large RCTs report small effects or effects in the opposite direction. "
            "The pooled estimate moves towards zero and I² rises sharply. "
            "Here, more data do not lead to a more decisive conclusion."
        ),
        "try_this": (
            "Compare V1 and V2 in the history chart. Note the pooled estimate, "
            "the confidence interval, and the increased heterogeneity."
        ),
    },
    "V3": {
        "question": "Does a larger evidence base always mean greater certainty?",
        "observation": (
            "Four additional studies are included. The estimated average benefit "
            "becomes smaller again, while substantial differences between studies remain."
        ),
        "try_this": (
            "Explore the low-risk-of-bias sensitivity analysis. "
            "Does the estimate change when studies with higher risk of bias are excluded?"
        ),
    },
    "V4": {
        "question": "Can a review change without adding any new studies?",
        "observation": (
            "No new study is added. Instead, SOMNI-TRIAL's incorrectly extracted "
            "four-week outcome is replaced by the correct eight-week outcome. "
            "The meta-analysis changes nevertheless."
        ),
        "try_this": (
            "Compare the number of studies in V3 and V4. Then open "
            "'Workflow & Data' and locate the S06 correction in the event log."
        ),
    },
}

LAB_EXPERIMENTS = [
    (
        "A · Add a large null-effect trial",
        "Use **Add a new synthetic study** to enter an RCT with "
        "**500 participants per group**, **intervention mean = 12**, "
        "**control mean = 12**, and **SD = 4** in both groups. "
        "Compare the pooled MD and its confidence interval with the original version.",
    ),
    (
        "B · Change precision, not the effect",
        "For the same new study, reduce the **sample size to 30 per group** "
        "without changing the means. What happens to its confidence interval "
        "and its weight in the forest plot?",
    ),
    (
        "C · Correct an existing study",
        "Change the **intervention mean** of an existing study and select "
        "**Apply changes and recalculate**. Observe whether the pooled estimate "
        "or heterogeneity changes. Use **Reset experiment to original version** "
        "to return to the starting point.",
    ),
]
