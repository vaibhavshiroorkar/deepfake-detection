"""Experiments: the frozen validation record, and every run beside it.

Two things, in that order. The top half is the validated evidence behind the
baseline the Evidence gate runs, loaded through `evidence.py` so a record that
fails a provenance check reports the failure instead of rendering. The bottom
half reads the local MLflow store and puts the runs next to each other, which is
what makes one run's numbers mean anything.

Read-only. Nothing on this page starts, edits or deletes a run.
"""

import streamlit as st

from deepfake_detection.dashboard.lib import mlflow_runs
from deepfake_detection.dashboard.sections.experiments import render_experiments

render_experiments()

st.divider()
st.subheader("Run comparison")
st.caption(
    "Every run in the local MLflow store, newest first. Training runs record a "
    "validation loss; evaluation runs record the threshold metrics."
)

try:
    names = mlflow_runs.experiments()
except (OSError, RuntimeError, ValueError) as error:
    st.error(f"The MLflow store could not be read: {error}")
else:
    if not names:
        st.info(
            "No experiments in the local store yet. Train one with "
            "`uv run ddf run --config <config>.yaml`."
        )
    else:
        chosen = st.selectbox("Experiment", names, key="mlflow_experiment")
        summaries = mlflow_runs.runs(chosen)
        rows = mlflow_runs.comparison_rows(summaries)
        if not rows:
            st.info(f"`{chosen}` holds no runs.")
        else:
            st.dataframe(rows, hide_index=True, width="stretch")
            st.caption(
                f"{len(rows)} runs. A blank cell means that run did not record "
                "that parameter or metric, not that its value was zero."
            )
    st.caption(f"Tracking store: `{mlflow_runs.tracking_uri()}`")
    st.markdown(
        "Serve the full UI with "
        "`uv run mlflow ui --backend-store-uri sqlite:///mlflow.db` and open "
        "[http://127.0.0.1:5000](http://127.0.0.1:5000)."
    )
