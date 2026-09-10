"""Fusion: does combining the branches beat using one on its own?

The page answers that in its first two lines, then puts everything else behind
`Advanced`. A first-time reader needs the verdict and the two numbers; the
provenance hashes, the full subset table and the per-method breakdown matter
only once they have a reason to look.

Every number is read from a run directory. The dashboard fits nothing, so each
figure traces to a file and a checkpoint hash.

The honest version of the verdict needs both partitions. In-domain, combining
wins. On DFDC it does not. Reporting only the first would answer the easier
question, so the summary states both.
"""

import streamlit as st

from deepfake_detection.dashboard.lib import results

st.title("Fusion")
st.caption("Do three branches together beat the best one alone?")

fusion = results.load_fusion()

if not fusion.available:
    st.info(
        "No trained fusion model yet. Expected under "
        f"`{fusion.run_dir}`:\n\n"
        + "\n".join(f"- `{name}`" for name in fusion.missing)
        + "\n\nRun `ddf train fusion`, then `scripts/run_ablation.py`."
    )
    st.stop()

metadata = fusion.metadata
ablation = fusion.ablation or {}

# ------------------------------------------------------------- the answer

in_domain_beats = results.beats_every_single_stream(ablation, "in-domain")
dfdc_beats = results.beats_every_single_stream(ablation, "dfdc")

if in_domain_beats and dfdc_beats is False:
    st.success(
        "**On the data it was trained on, yes.** Combining all three branches "
        "beats every branch used alone."
    )
    st.warning(
        "**On a different dataset, no.** Tested on DFDC, combining does not "
        "beat the best single branch. The gain does not transfer."
    )
elif in_domain_beats and dfdc_beats:
    st.success("**Yes, on both datasets.** Combining beats every single branch.")
else:
    st.info("Not enough of the ablation is scored to give a verdict yet.")

columns = st.columns(len(fusion.partitions) or 1)
LABELS = {
    "in-domain": ("Same dataset", "Held-out clips from the training dataset"),
    "dfdc": ("Different dataset", "DFDC, never seen during training"),
}
for column, (name, payload) in zip(
    columns, sorted(fusion.partitions.items()), strict=False
):
    title, explanation = LABELS.get(name, (name, ""))
    overall = (payload.get("overall") or {}).get("metrics", {})
    column.metric(title, f"{overall.get('roc_auc', float('nan')):.3f}", help=explanation)

st.caption(
    "Scores are ROC-AUC: 1.0 is perfect, 0.5 is a coin flip. Higher is better."
)

# ------------------------------------------------------------- the detail

with st.expander("Advanced: every combination of branches"):
    st.caption(
        "One fusion model per subset, all fitted on the same features and "
        "scored on the same clips, so the only thing changing is which "
        "branches are in the input."
    )
    rows = results.ablation_rows(ablation)
    partitions = [
        name for name in ("in-domain", "dfdc") if any(name in row for row in rows)
    ]
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
    if dfdc_beats is False:
        best = max(
            (row for row in rows if row.get("dfdc") is not None),
            key=lambda row: row["dfdc"],
            default=None,
        )
        if best:
            st.caption(
                f"On DFDC the best combination is {best['streams']} at "
                f"{best['dfdc']:.4f}, which is why all-three does not win there."
            )

with st.expander("Advanced: how each branch scores on its own"):
    raw = ablation.get("raw_branch_auc") or {}
    if not raw:
        st.info("This run recorded no uncalibrated branch scores.")
    else:
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
            "These are the branches before the fusion model recalibrates them. "
            "The sync branch scores below 0.5, which is worse than guessing. "
            "Calibration can flip a below-chance signal into an in-domain gain, "
            "and that gain did not survive the change of dataset."
        )

in_domain = fusion.partitions.get("in-domain")
if in_domain and in_domain.get("per_method"):
    with st.expander("Advanced: accuracy by manipulation method"):
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
                }
                for method, payload in sorted(in_domain["per_method"].items())
            ],
            hide_index=True,
            width="stretch",
            column_config={
                "roc_auc": st.column_config.NumberColumn("ROC-AUC", format="%.4f"),
                "coverage": st.column_config.NumberColumn("Coverage", format="%.2f"),
            },
        )

with st.expander("Advanced: provenance"):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Classifier", metadata.get("model", "unknown"))
    c2.metric("Branches", len(metadata.get("branches", [])))
    c3.metric("Training clips", f"{metadata.get('samples', 0):,}")
    c4.metric("Fold checkpoints", len(metadata.get("oof_run_ids", [])))
    st.caption(
        "Fitted on out-of-fold features: every clip's branch scores come from a "
        "checkpoint trained on folds that excluded it, so fusion never reads a "
        "score from a model that saw the clip."
    )
    st.code(
        f"run              {fusion.run_dir}\n"
        f"split hash       {metadata.get('split_hash', '?')}\n"
        f"preprocessing    {metadata.get('preprocessing_hash', '?')}",
        language="text",
    )
    delta = (in_domain or {}).get("fusion_vs_visual_auc")
    if delta:
        st.caption(
            "Fusion beats the visual branch alone by "
            f"{delta.get('estimate', 0):+.4f} ROC-AUC in-domain "
            f"(95% CI {delta.get('lower', 0):+.4f} to {delta.get('upper', 0):+.4f})."
        )
