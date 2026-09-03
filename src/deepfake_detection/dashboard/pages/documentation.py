from pathlib import Path

import streamlit as st

from deepfake_detection.dashboard.components import render_page_header, render_status
from deepfake_detection.dashboard.navigation import PageState

DOCUMENTS = (
    ("Project handoff", "docs/handoff.md"),
    ("Research design", "docs/research-design.md"),
    ("Data card", "docs/data-card.md"),
    ("Reproducibility", "docs/reproducibility.md"),
    ("Model selection", "docs/model-selection.md"),
    ("CLI reference", "docs/reference/cli.md"),
)
_PROJECT_ROOT = Path(__file__).resolve().parents[4]

render_page_header(
    "Project record",
    "Documentation",
    "Open the repository records that define scope, data, runs, and reproducibility.",
)
render_status(PageState.READY)

st.subheader("Repository records")
for label, relative in DOCUMENTS:
    if (_PROJECT_ROOT / relative).is_file():
        st.markdown(f"[{label}]({relative})")
    else:
        st.error(f"Documentation is unavailable: {relative}")
