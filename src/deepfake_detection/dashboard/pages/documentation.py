import streamlit as st

from deepfake_detection.dashboard.components import render_page_header, render_status
from deepfake_detection.dashboard.navigation import PageState

DOCUMENTS = (
    (
        "Project handoff",
        "https://github.com/vaibhavshiroorkar/deepfake-detection/blob/main/docs/handoff.md",
    ),
    (
        "Research design",
        "https://github.com/vaibhavshiroorkar/deepfake-detection/blob/main/docs/research-design.md",
    ),
    (
        "Data card",
        "https://github.com/vaibhavshiroorkar/deepfake-detection/blob/main/docs/data-card.md",
    ),
    (
        "Reproducibility",
        "https://github.com/vaibhavshiroorkar/deepfake-detection/blob/main/docs/reproducibility.md",
    ),
    (
        "Model selection",
        "https://github.com/vaibhavshiroorkar/deepfake-detection/blob/main/docs/model-selection.md",
    ),
    (
        "CLI reference",
        "https://github.com/vaibhavshiroorkar/deepfake-detection/blob/main/docs/reference/cli.md",
    ),
)

render_page_header(
    "Project record",
    "Documentation",
    "Open the repository records that define scope, data, runs, and reproducibility.",
)
render_status(PageState.READY)

st.subheader("Repository records")
for label, target in DOCUMENTS:
    st.markdown(f"[{label}]({target})")
