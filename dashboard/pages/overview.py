"""Overview — the short landing page.

Answers three questions and nothing more: what the system detects, what it is
built out of, and what each page of this dashboard does. Every explanation of
*how* a step or a model works now lives on the Documentation page; this page
links there rather than reproducing it.

Static by construction: no model loads, no decoding, no data/ access, so it
opens instantly and is safe as the default page.
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

st.title("Overview")
st.caption("Detecting lip-sync deepfakes by measuring whether the face and the voice belong to "
           "the same event. Read-only page — nothing is loaded or decoded here.")

st.header("What this detects")
st.markdown("""
A wav2lip-style forgery repaints only the mouth region of a genuine video so the lips appear to
speak new words. Almost every pixel is real, so the manufacturing residue that vision-only
detectors hunt for is tiny and largely destroyed by compression.

What the forgery cannot hide is the relationship between the two modalities. In a real recording
the moving mouth and the sound it produces are two views of one physical event. Synthesis breaks
that lock, and the mismatch survives compression, resolution loss and a well-trained generator.

That is why the design has no standalone audio classifier: the signal is not "this voice sounds
synthetic", it is "this voice disagrees with this face".
""")

st.header("How the system is built")
st.markdown("""
Each clip becomes three tensors, which feed five streams, which fuse into one decision.

| Stage | What happens | Result |
|---|---|---|
| **Preprocessing** | 16 timestamps per clip; the video path detects, aligns and crops faces, the audio path decodes to 16 kHz and cuts one window per timestamp | `faces [16, 3, 224, 224]`, `mouth [16, 3, 96, 96]`, `audio [16, 5600]` |
| **Streams** | five models read those tensors — three visual, two cross-modal | one 256-d embedding each |
| **Fusion** | the embeddings are concatenated and passed through an MLP | P(fake) |

Both paths are indexed by the **same 16 timestamps**, so frame *i* and audio window *i* describe
the same instant. Every stream emits an embedding and never a score, which is what lets fusion
learn interactions between them instead of averaging opinions.

| Stream | Reads | Looks for | Status |
|---|---|---|---|
| **EfficientNet-B0** | faces | manipulation artifacts | built |
| **Xception** | faces | manipulation artifacts | built |
| **DINOv2 (ViT-S/14)** | faces | self-supervised features, for generalisation | Stage 3 |
| **Lip-sync** | mouth + audio | audio–video synchronisation mismatch | Stage 4 |
| **Emotion** | faces + audio | affect mismatch between face and voice | Stage 5 |
""")

st.header("The pages")
st.info("**This dashboard never trains and never writes `data/processed/`.** Training always runs "
        "as a background script; where a page offers to train, it hands you the command to run in "
        "a terminal.")
st.markdown("""
| Page | What it does |
|---|---|
| **Overview** | This page. |
| **Preprocessing** | Live compute. *Config* picks the dataset, split and clip and sets `N` frames and the audio window; *Visual* and *Audio* show each pipeline step cumulatively, ending in the exact tensor the model receives. Calls the same `preprocessing/ops/` functions as the batch pipeline, so there is no second implementation to drift. |
| **Streams** | Live inference. *Visual* gives EfficientNet-B0 and Xception real model boxes — temporal model, hidden size, embedding dim, freeze — with Train (emits a terminal command) and Run (forward-passes one clip on untrained weights, to check shapes, device and speed). *Lip-Sync* and *Emotions* are scaffolds until Stages 4 and 5. |
| **Fusion** 🔒 | Locked until Stage 6. |
| **Explainability** 🔒 | Locked until Stage 10. |
| **Documentation** | The in-depth reference: every preprocessing step, every model, the fusion and evaluation design, and the dataset and splits. |
""")
