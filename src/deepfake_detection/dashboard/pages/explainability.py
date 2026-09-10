"""Explainability: where a result holds, and where it quietly fails.

Leads with the single most useful thing a breakdown gives a reader, the gap
between the best and worst manipulation method, because that is the number an
overall score hides. Everything else sits behind `Advanced`.

Of the three views this page was specified to carry, one has data and two do
not. The two are named as missing rather than filled with something that looks
like an explanation.
"""

import streamlit as st

from deepfake_detection.dashboard.lib import results

st.title("Explainability")
st.caption("Which fakes does the model catch, and which does it miss?")

breakdowns = results.load_branch_breakdowns()
fusion = results.load_fusion()

if not breakdowns and not fusion.available:
    st.info(
        "No evaluation reports found under "
        f"`{fusion.run_dir / 'evaluation'}`. Run `ddf evaluate branch` first."
    )
    st.stop()

if breakdowns:
    labels = {
        f"{item.branch} on {item.dataset}": item for item in breakdowns
    }
    chosen = st.selectbox("Which result to look at", list(labels), key="explain_eval")
    item = labels[chosen]

    methods = {
        name: values
        for name, values in item.per_method.items()
        if values.get("accuracy") is not None
    }
    if methods:
        best = max(methods.items(), key=lambda pair: pair[1]["accuracy"])
        worst = min(methods.items(), key=lambda pair: pair[1]["accuracy"])

        c1, c2, c3 = st.columns(3)
        c1.metric("Clips tested", f"{item.rows:,}")
        c2.metric(
            "Best method",
            f"{best[1]['accuracy']:.0%}",
            help=f"{best[0]}, {best[1].get('rows', 0)} clips",
        )
        c3.metric(
            "Worst method",
            f"{worst[1]['accuracy']:.0%}",
            help=f"{worst[0]}, {worst[1].get('rows', 0)} clips",
        )

        spread = best[1]["accuracy"] - worst[1]["accuracy"]
        if spread > 0.2:
            st.warning(
                f"**The model is uneven.** It catches `{best[0]}` "
                f"{best[1]['accuracy']:.0%} of the time and `{worst[0]}` only "
                f"{worst[1]['accuracy']:.0%}. One overall score would hide that."
            )
        else:
            st.success(
                "**The model is even across methods.** The best and worst "
                f"differ by {spread:.0%}, so no single generator is slipping past."
            )

    with st.expander("Advanced: every manipulation method"):
        st.dataframe(
            [
                {
                    "method": name,
                    "clips": values.get("rows"),
                    "accuracy": values.get("accuracy"),
                    "mean probability": values.get("mean_probability"),
                }
                for name, values in sorted(item.per_method.items())
            ],
            hide_index=True,
            width="stretch",
            column_config={
                "accuracy": st.column_config.NumberColumn("Accuracy", format="%.4f"),
                "mean probability": st.column_config.NumberColumn(
                    "Mean p(fake)", format="%.4f"
                ),
            },
        )
        st.caption(
            "Mean p(fake) sits beside accuracy because the two fail differently. "
            "A row of real clips with high accuracy and a mean probability near "
            "1.0 is a model that is right by luck at this threshold and wrong "
            "about the clips."
        )

    if item.per_manipulation_type:
        with st.expander("Advanced: by manipulation type"):
            st.dataframe(
                [
                    {
                        "type": name,
                        "clips": values.get("rows"),
                        "accuracy": values.get("accuracy"),
                        "mean probability": values.get("mean_probability"),
                    }
                    for name, values in sorted(item.per_manipulation_type.items())
                ],
                hide_index=True,
                width="stretch",
                column_config={
                    "accuracy": st.column_config.NumberColumn(
                        "Accuracy", format="%.4f"
                    ),
                    "mean probability": st.column_config.NumberColumn(
                        "Mean p(fake)", format="%.4f"
                    ),
                },
            )

    with st.expander("Advanced: provenance"):
        st.code(
            f"branch        {item.branch}\n"
            f"dataset       {item.dataset}\n"
            f"clips         {item.rows}\n"
            f"checkpoint    {item.checkpoint_sha256}",
            language="text",
        )

in_domain = fusion.partitions.get("in-domain") if fusion.available else None
if in_domain:
    with st.expander("Advanced: fairness by subgroup"):
        st.caption(
            "Coverage travels with every score. A subgroup the model abstained "
            "on half the time has its metric computed from the half it "
            "answered, and an accuracy alone would hide that."
        )
        for attribute in ("gender", "race"):
            rows = results.subgroup_rows(in_domain, attribute)
            if not rows:
                continue
            st.markdown(f"**{attribute.title()}**")
            st.dataframe(
                rows,
                hide_index=True,
                width="stretch",
                column_config={
                    "roc_auc": st.column_config.NumberColumn("ROC-AUC", format="%.4f"),
                    "balanced_accuracy": st.column_config.NumberColumn(
                        "Balanced accuracy", format="%.4f"
                    ),
                    "eer": st.column_config.NumberColumn("EER", format="%.4f"),
                    "coverage": st.column_config.NumberColumn(
                        "Coverage", format="%.2f"
                    ),
                },
            )
        st.caption(
            "These labels come from FakeAVCeleb's own annotations. They are "
            "coarse, and a difference between two groups is a prompt to "
            "investigate, not a fairness result on its own."
        )

with st.expander("Advanced: what this page does not have"):
    st.markdown(
        "**Grad-CAM.** Where a visual backbone looks when it calls a clip fake. "
        "Needs a backward pass over a selected clip, which this dashboard does "
        "not run: it reads recorded results and never fits or differentiates a "
        "model."
    )
    st.markdown(
        "**Embedding shift.** Which stream's embedding moves most under which "
        "manipulation. Needs stream embeddings on matched pairs, which the "
        "LAV-DF adapter can cut but no run has exported yet."
    )
