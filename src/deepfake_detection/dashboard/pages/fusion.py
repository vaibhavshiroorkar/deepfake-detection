"""Fusion: the trained model, what it weighs, and whether combining helps.

Unlocked. It was locked because the only fusion artifact was a software fixture.
`runs/program-20260906` now holds a fusion model fitted on genuine out-of-fold
branch features, scored on a held-out in-domain partition and on DFDC, with an
ablation over every branch subset.

Every number here is read from that run directory. The dashboard does not fit
models, so a figure on this page traces to a file and a checkpoint hash.

The ablation is the point of the page rather than the fused score. Objective 3
asks whether fusion beats every individual stream, and the answer depends on the
partition: in-domain it does, cross-corpus it does not. Showing only the
in-domain number would answer the easier question.
"""

import streamlit as st

from deepfake_detection.dashboard.lib import results

st.title("Fusion")
st.caption(
    "Calibrated branch scores combined into one probability, and the ablation "
    "that says whether combining was worth it."
)

fusion = results.load_fusion()

if not fusion.available:
    st.info(
        "No trained fusion model found. Expected these under "
        f"`{fusion.run_dir}`:\n\n"
        + "\n".join(f"- `{name}`" for name in fusion.missing)
        + "\n\nRun `ddf train fusion`, then `scripts/run_ablation.py`."
    )
    st.stop()

metadata = fusion.metadata

# ---------------------------------------------------------------- the model

with st.container(border=True):
    st.markdown("**The trained model**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Classifier", metadata.get("model", "unknown"))
    c2.metric("Branches", len(metadata.get("branches", [])))
    c3.metric("Out-of-fold clips", f"{metadata.get('samples', 0):,}")
    c4.metric("Fold checkpoints", len(metadata.get("oof_run_ids", [])))
    st.caption(
        "Fitted on out-of-fold features: every clip's branch scores come from a "
        "checkpoint trained on folds that excluded it, so fusion never reads a "
        "score from a model that saw the clip. Split hash "
        f"`{metadata.get('split_hash', '?')[:16]}`, preprocessing hash "
        f"`{metadata.get('preprocessing_hash', '?')[:16]}`."
    )

# ------------------------------------------------------------- the headline

if fusion.partitions:
    st.subheader("Fused performance")
    columns = st.columns(len(fusion.partitions))
    for column, (name, payload) in zip(
        columns, sorted(fusion.partitions.items()), strict=False
    ):
        overall = (payload.get("overall") or {}).get("metrics", {})
        with column:
            st.metric(name, f"{overall.get('roc_auc', float('nan')):.4f}", help="ROC-AUC")
            st.caption(
                f"balanced accuracy {overall.get('balanced_accuracy', float('nan')):.4f}"
                f"  ·  EER {overall.get('eer', float('nan')):.4f}"
            )

    delta = (fusion.partitions.get("in-domain") or {}).get("fusion_vs_visual_auc")
    if delta:
        low, high = delta.get("lower"), delta.get("upper")
        st.caption(
            f"In-domain, fusion beats the visual branch alone by "
            f"{delta.get('estimate', 0):+.4f} ROC-AUC "
            f"(95% CI {low:+.4f} to {high:+.4f})."
        )

# ---------------------------------------------------------------- ablation

st.subheader("Ablation: does combining beat the parts?")
st.caption(
    "One fusion model per branch subset, all fitted on the same out-of-fold "
    "features and scored on the same partitions, so the only thing changing is "
    "which branches are in the input."
)

rows = results.ablation_rows(fusion.ablation)
partitions = [name for name in ("in-domain", "dfdc") if any(name in row for row in rows)]
st.dataframe(
    rows,
    hide_index=True,
    width="stretch",
    column_config={
        "streams": st.column_config.TextColumn("Branches"),
        "count": st.column_config.NumberColumn("How many", width="small"),
        **{
            name: st.column_config.NumberColumn(name, format="%.4f")
            for name in partitions
        },
    },
)

verdicts = st.columns(max(len(partitions), 1))
for column, name in zip(verdicts, partitions, strict=False):
    beats = results.beats_every_single_stream(fusion.ablation, name)
    with column:
        if beats is None:
            st.info(f"**{name}**: not enough subsets scored to say.")
        elif beats:
            st.success(f"**{name}**: all branches beat every single branch.")
        else:
            st.warning(
                f"**{name}**: all branches does NOT beat every single branch."
            )

if results.beats_every_single_stream(fusion.ablation, "dfdc") is False:
    st.caption(
        "The two partitions disagree, and that is the result rather than a "
        "problem with it. Objective 3's success metric is met in-domain and not "
        "cross-corpus, so fusion as built here is not evidence of better "
        "generalization."
    )

raw = (fusion.ablation or {}).get("raw_branch_auc") or {}
if raw:
    with st.expander("Raw branch scores, before calibration"):
        st.caption(
            "A single-branch row above is that branch calibrated by the fusion "
            "model. These are the branches as they actually rank, which is the "
            "fairer comparison."
        )
        branches = sorted({branch for scores in raw.values() for branch in scores})
        st.dataframe(
            [
                {"branch": branch, **{name: raw[name].get(branch) for name in raw}}
                for branch in branches
            ],
            hide_index=True,
            width="stretch",
            column_config={
                name: st.column_config.NumberColumn(name, format="%.4f") for name in raw
            },
        )
        st.caption(
            "The sync branch ranks below 0.5, which is below chance. Calibration "
            "can invert a below-chance signal into an in-domain gain, and that "
            "gain did not survive a change of corpus."
        )

# ------------------------------------------------------------ per method

in_domain = fusion.partitions.get("in-domain")
if in_domain and in_domain.get("per_method"):
    st.subheader("By manipulation method")
    st.caption(
        "Averaged over methods rather than over clips, so a method with few "
        "clips is not drowned out: macro ROC-AUC "
        f"{in_domain.get('macro_method_roc_auc', float('nan')):.4f}."
    )
    st.dataframe(
        [
            {
                "method": method,
                "clips": payload.get("metrics", {}).get("rows")
                or payload.get("rows"),
                "coverage": payload.get("coverage"),
                "roc_auc": payload.get("metrics", {}).get("roc_auc"),
                "balanced_accuracy": payload.get("metrics", {}).get(
                    "balanced_accuracy"
                ),
            }
            for method, payload in sorted(in_domain["per_method"].items())
        ],
        hide_index=True,
        width="stretch",
        column_config={
            "roc_auc": st.column_config.NumberColumn("ROC-AUC", format="%.4f"),
            "balanced_accuracy": st.column_config.NumberColumn(
                "Balanced accuracy", format="%.4f"
            ),
            "coverage": st.column_config.NumberColumn("Coverage", format="%.2f"),
        },
    )

st.caption(
    f"Read from `{fusion.run_dir}`. The fusion design is described on the "
    "Documentation page under *Fusion & evaluation*."
)
