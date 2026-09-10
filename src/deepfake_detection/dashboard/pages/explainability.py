"""Explainability: where the recorded results break down, and where they hold.

Partly unlocked. Of the three views this page was specified to carry, one has
data behind it now and two do not, so one is built and the other two say plainly
that they are not. Filling them with something that looks like an explanation
would be worse than the lock was.

Built: per-method and per-manipulation-type accuracy, plus subgroup coverage.
These come from `ddf evaluate branch`, which already writes both breakdowns.

Not built: Grad-CAM, which needs a backward pass the dashboard does not run, and
embedding shift, which needs stream embeddings on matched manipulation pairs.

The breakdown is the useful half anyway. An aggregate ROC-AUC hides that a
detector can be perfect on one generator and near chance on another, and it was
a per-method table that first showed wav2lip going undetected while the same
model scored 1.0000 overall.
"""

import streamlit as st

from deepfake_detection.dashboard.lib import results

st.title("Explainability")
st.caption(
    "Where a recorded result holds and where it breaks, by manipulation method "
    "and by subgroup."
)

breakdowns = results.load_branch_breakdowns()
fusion = results.load_fusion()

if not breakdowns and not fusion.available:
    st.info(
        "No evaluation reports found under "
        f"`{fusion.run_dir / 'evaluation'}`. Run `ddf evaluate branch` first."
    )
    st.stop()

# ------------------------------------------------- per method, per branch

if breakdowns:
    st.subheader("Per-method accuracy")
    st.caption(
        "One evaluation per branch and dataset. A single overall number can be "
        "excellent while a whole generator goes undetected, which is exactly "
        "what happened here: a visual model at 1.0000 in-domain caught none of "
        "the wav2lip clips on the external set."
    )

    labels = {
        f"{item.branch} on {item.dataset} ({item.rows:,} clips)": item
        for item in breakdowns
    }
    chosen = st.selectbox("Evaluation", list(labels), key="explain_eval")
    item = labels[chosen]

    tabs = st.tabs(["By method", "By manipulation type"])
    for tab, (title, payload) in zip(
        tabs,
        (("method", item.per_method), ("manipulation type", item.per_manipulation_type)),
        strict=False,
    ):
        with tab:
            if not payload:
                st.info(f"This evaluation recorded no per-{title} breakdown.")
                continue
            st.dataframe(
                [
                    {
                        title: name,
                        "clips": values.get("rows"),
                        "accuracy": values.get("accuracy"),
                        "mean probability": values.get("mean_probability"),
                    }
                    for name, values in sorted(payload.items())
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
            st.caption(
                "Mean p(fake) sits beside accuracy because the two fail "
                "differently. A row of real clips with high accuracy and a mean "
                "probability near 1.0 is a model that is right by luck at this "
                "threshold and wrong about the clips."
            )
    st.caption(f"Checkpoint `{item.checkpoint_sha256[:16]}`.")

# ------------------------------------------------------------- subgroups

in_domain = fusion.partitions.get("in-domain") if fusion.available else None
if in_domain:
    st.subheader("By subgroup")
    st.caption(
        "Coverage travels with every score. A subgroup the model abstained on "
        "half the time has its metric computed from the half it answered, and "
        "an accuracy alone would hide that."
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
                "coverage": st.column_config.NumberColumn("Coverage", format="%.2f"),
            },
        )
    st.caption(
        "These labels come from FakeAVCeleb's own annotations. They are coarse, "
        "and a difference between two groups here is a prompt to investigate, "
        "not a fairness result on its own."
    )

# --------------------------------------------------------- what is missing

st.divider()
st.subheader("Not built")
st.markdown(
    "**Grad-CAM.** Where a visual backbone looks when it calls a clip fake. "
    "Needs a backward pass over a selected clip, which this dashboard does not "
    "run: it reads recorded results and never fits or differentiates a model."
)
st.markdown(
    "**Embedding shift.** Which stream's embedding moves most under which "
    "manipulation. Needs stream embeddings on matched pairs, which the LAV-DF "
    "adapter can cut but no run has exported yet."
)
