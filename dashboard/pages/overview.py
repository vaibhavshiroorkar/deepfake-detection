"""Overview: the landing page.

States the problem, shows the architecture as one diagram, and says what is
built. How anything works belongs on the Documentation page; this page links
there instead of explaining it twice.

This is also the one place that states the "no training here" rule, so the other
pages do not each repeat it.

Static by construction: no model loads, no decoding, no data/ access, so it opens
instantly and is safe as the default page.
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

DIAGRAM = _REPO_ROOT / "assets" / "flow.png"

# app.py sets layout="wide" because the Preprocessing page needs the room for its
# 8-per-row frame grids, and Streamlit has no per-page override. This page is
# almost entirely prose, and prose set across 1200px is genuinely hard to read, so
# everything renders inside one bounded column. Left-aligned with a wide right
# margin rather than centred: it reads as an editorial column instead of a narrow
# box floating in the middle of the screen.
body, _ = st.columns([5, 3], gap="large")

with body:
    st.title("Audio-Visual Deepfake Detection")
    st.caption("Detecting lip-sync forgeries by checking whether a face and a voice belong to the "
               "same event.")

    st.header("The problem")
    st.markdown("""
A wav2lip forgery repaints the mouth and leaves the rest of the video alone. Almost every pixel is
genuine, so there is very little manufacturing residue for a vision-only detector to find, and
compression destroys most of what remains. Detectors that score well on full face swaps do poorly
here for a simple reason: there is barely anything to see.

What the forgery cannot repair is agreement between the two tracks. A real recording captures one
physical event twice, as light off a moving mouth and as the sound that mouth made. Synthesis breaks
the correspondence, and it stays broken after compression and resolution loss. So this system
measures disagreement between face and voice rather than asking whether a voice sounds synthetic,
which is why it has no standalone audio classifier.
""")

    st.header("Architecture")

    # Portrait diagram, inset again inside the already-bounded column so it does
    # not tower over the text it illustrates.
    _, diagram_col, _ = st.columns([1, 5, 1])
    with diagram_col:
        if DIAGRAM.exists():
            st.image(str(DIAGRAM), width="stretch")
        else:
            st.warning(f"Architecture diagram not found at `{DIAGRAM.relative_to(_REPO_ROOT)}`.")

    st.markdown("""
Preprocessing samples 16 timestamps per clip and runs a video path and an audio path over them,
producing three tensors:
""")
    st.code("""faces  [16, 3, 224, 224]   ->  visual streams, emotion stream
mouth  [16, 3,  96,  96]   ->  lip-sync stream
audio  [16, 5600]          ->  lip-sync stream, emotion stream""", language="text")
    st.markdown("""
Both paths are indexed by the **same 16 timestamps**, so frame *i* and audio window *i* describe the
same instant. Without that, every clip would look desynchronised and the cross-modal streams would
measure the pipeline rather than the forgery.

Each of the five streams emits a 256-dimensional embedding rather than a score. Fusion concatenates
them and learns from the combination, so it can represent a conjunction like "artifact evidence is
weak but lip-sync mismatch is strong". That is the signature of a lip-sync forgery, and no weighted
average of five scores can express it.
""")

    st.header("Status")
    st.markdown("""
| Component | State |
|---|---|
| Preprocessing | Built. Shared functions in `preprocessing/ops/`, called by both the batch pipeline and this dashboard. |
| Manifests and splits | Built. Identity-disjoint, verified by `verify_splits.py`. |
| Visual stream module | Built. EfficientNet-B0 and Xception wired; DINOv2 not yet. |
| Training | Not written. |
| Lip-sync and emotion streams | Designed. Stages 4 and 5. |
| Fusion, evaluation, explainability | Designed. Stages 6, 7 and 10. |
""")
    st.caption("An earlier build of the visual stream reached test accuracy 0.963 and AUC 0.994 "
               "in-distribution. Face alignment has since changed the cached pixels, so that is "
               "the bar to re-clear rather than a current result.")

    st.header("The pages")
    st.markdown("""
**Preprocessing** is the working page. Pick a clip or upload your own video, then step through the
visual and audio tabs. Every step is a toggle, applied cumulatively, ending in the exact tensor a
model would receive.

**Streams**, **Fusion** and **Explainability** are locked. Each says what will land there and what
unlocks it.

**Documentation** covers how every step works, what each model does, and how fusion, evaluation and
the splits are designed.
""")
    st.info("Nothing in this dashboard trains a model or writes to `data/processed/`. Training runs "
            "as a background script, and pages that offer to train hand you a command to run in a "
            "terminal.")
