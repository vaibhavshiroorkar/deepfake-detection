from pathlib import Path

import streamlit as st

from deepfake_detection.dashboard.components import render_page_header, render_status
from deepfake_detection.dashboard.evidence import load_validation_evidence
from deepfake_detection.dashboard.navigation import PageState

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_METRICS_PATH = (
    _PROJECT_ROOT / "runs" / "initial-20260902" / "visual-validation-metrics.json"
)
_HISTORY_PATH = (
    _PROJECT_ROOT / "runs" / "initial-20260902" / "visual-initial-history.json"
)

render_page_header(
    "Research record",
    "Experiments",
    "Review the saved development-validation record behind the visual baseline.",
)
render_status(PageState.READY)

try:
    evidence = load_validation_evidence(_METRICS_PATH, _HISTORY_PATH)
except (FileNotFoundError, OSError, ValueError) as exc:
    st.error(f"Evidence is unavailable: {exc}")
else:
    st.subheader("FakeAVCeleb development validation")
    st.write(
        "The frozen visual baseline was evaluated on 400 source-disjoint "
        "FakeAVCeleb validation rows at a fixed threshold of 0.5."
    )
    run_columns = st.columns(2)
    run_columns[0].metric("Training run", evidence.checkpoint_run_id)
    run_columns[1].metric("Evaluation run", evidence.evaluation_run_id)
    epoch_columns = st.columns(3)
    epoch_columns[0].metric("Epochs", len(evidence.epochs))
    epoch_columns[1].metric("Best epoch", evidence.best_epoch)
    epoch_columns[2].metric("Validation rows", evidence.rows)

    st.subheader("FakeAVCeleb development validation metrics")
    st.table(
        [
            {"Metric": "ROC AUC", "Value": f"{evidence.metrics['roc_auc']:.6f}"},
            {"Metric": "PR AUC", "Value": f"{evidence.metrics['pr_auc']:.6f}"},
            {
                "Metric": "Balanced accuracy",
                "Value": f"{evidence.metrics['balanced_accuracy']:.4f}",
            },
            {"Metric": "F1", "Value": f"{evidence.metrics['f1']:.6f}"},
        ]
    )
    st.subheader("FakeAVCeleb development validation confusion counts")
    st.table(
        [
            {"Count": "True positives", "Rows": evidence.confusion["true_positive"]},
            {"Count": "True negatives", "Rows": evidence.confusion["true_negative"]},
            {"Count": "False positives", "Rows": evidence.confusion["false_positive"]},
            {"Count": "False negatives", "Rows": evidence.confusion["false_negative"]},
        ]
    )
    st.subheader("Local MLflow record")
    st.markdown("[http://127.0.0.1:5000](http://127.0.0.1:5000)")
    st.write(
        "Open the initial-baseline-20260902 experiment, then select training "
        "run 4243b35e64c743b89cc33000cc9d3d3e or evaluation run "
        "56182266f70a424581f763b2d3b41989."
    )
