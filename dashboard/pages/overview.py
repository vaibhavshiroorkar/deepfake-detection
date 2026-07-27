"""Overview — the landing page.

Says what the project detects, shows the architecture as one diagram, and states
honestly what is built and what is not. Everything about *how* a step or a model
works lives on the Documentation page; this page links there instead of
reproducing it.

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

st.title("Audio-Visual Deepfake Detection")
st.caption("Detecting lip-sync forgeries by measuring whether a face and a voice belong to the "
           "same event.")

st.header("The problem")
st.markdown("""
A wav2lip forgery repaints the mouth and leaves the rest of the video alone. Almost every pixel is
genuine, so there is very little manufacturing residue for a vision-only detector to find, and
compression destroys most of what there is. Detectors that score well on full face swaps do poorly
here for a straightforward reason: there is barely anything to see.

What the forgery cannot repair is agreement between the two tracks. A real recording captures one
physical event twice, as light reflected off a moving mouth and as the sound that mouth produced.
Synthesis breaks the correspondence between them, and it stays broken after compression and
resolution loss.

So this system is built to measure disagreement between face and voice. That is also why it has no
standalone audio classifier: the useful question is not whether a voice sounds synthetic, but
whether it matches the face it is paired with.
""")

st.header("Architecture")
st.caption("A clip becomes three tensors, five streams turn those into embeddings, and one fusion "
           "head turns the embeddings into a decision.")

# Portrait diagram, so it goes in a middle column — at full container width it
# renders absurdly tall on a wide screen.
_, diagram_col, _ = st.columns([1, 3, 1])
with diagram_col:
    if DIAGRAM.exists():
        st.image(str(DIAGRAM), width="stretch")
    else:
        st.warning(f"Architecture diagram not found at `{DIAGRAM.relative_to(_REPO_ROOT)}`.")

st.markdown("""
Preprocessing samples 16 timestamps per clip and runs two paths over them. The video path detects,
aligns and crops faces, and derives a mouth region from the same detection. The audio path decodes
to 16 kHz and cuts one window centred on each timestamp. Both paths are indexed by the **same 16
timestamps**, so frame *i* and audio window *i* describe the same instant. Without that, every clip
would look desynchronised and the cross-modal streams would be measuring the pipeline rather than
the forgery.
""")
st.code("""faces  [16, 3, 224, 224]   ->  visual streams, emotion stream
mouth  [16, 3,  96,  96]   ->  lip-sync stream
audio  [16, 5600]          ->  lip-sync stream, emotion stream""", language="text")
st.markdown("""
Each stream emits a 256-dimensional embedding and never a score. Fusion concatenates them and
learns from the combination, which lets it represent a conjunction like "artifact evidence is weak
but lip-sync mismatch is strong" — the signature of a wav2lip forgery, and something no weighted
average of five scores can express.
""")

st.header("Where the project stands")
st.markdown("""
| Component | State |
|---|---|
| Preprocessing | Built. Shared pure functions in `preprocessing/ops/`, called by both the batch pipeline and this dashboard, so there is no second implementation to drift. |
| Manifests and splits | Built. Identity-disjoint splits from `build_splits.py`, checked by `verify_splits.py`. |
| Visual stream module | Built. One config-driven module in `models/streams/common/`; EfficientNet-B0 and Xception are wired, DINOv2 is not yet. |
| Training | Not written. The streams are defined, but nothing has been trained since the preprocessing rebuild. |
| Lip-sync and emotion streams | Designed, not built. Stages 4 and 5. |
| Fusion, evaluation, explainability | Designed, not built. Stages 6, 7 and 10. |
""")
st.caption("An earlier build of the visual stream reached test accuracy 0.963 and AUC 0.994 "
           "in-distribution. Face alignment has since changed the cached pixels, so that is the "
           "bar to re-clear rather than a current result.")

st.header("Using this dashboard")
st.markdown("""
**Preprocessing** is the working page. Pick a dataset, split and clip under *Config*, then step
through the *Visual* and *Audio* tabs: every step is a toggle, applied cumulatively, ending in the
exact tensor a model would receive.

**Streams**, **Fusion** and **Explainability** are locked. Each lists what will land there and what
unlocks it, rather than showing controls that do not work.

**Documentation** is the reference: how every preprocessing step works, what each model does and
why it was chosen, and how fusion, evaluation and the splits are designed.
""")
st.info("This dashboard never trains and never writes `data/processed/`. Training runs as a "
        "background script; pages that offer to train hand you a command to run in a terminal.")
