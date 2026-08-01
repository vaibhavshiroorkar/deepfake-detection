"""Documentation: the long-form reference for the whole system.

Seven tabs, ordered the way a clip moves through the system: the problem, the
end-to-end flow, the two preprocessing paths in detail, the stream models that
consume them, fusion and evaluation, then the data and splits underneath it all.

Division of labour with the Overview page: Overview is the map, covering what is
detected, the architecture in one diagram, and what is built. This page is the
territory, and the only place a mechanism is explained in full. The first tab
repeats just enough of Overview to stand on its own.

It is documentation, not compute: nothing here loads a model, decodes a clip or
touches data/, so the page opens instantly.

Sources of truth it mirrors: docs/PROJECT_OVERVIEW.md (§2 architecture, §3
fusion, §5 data, §6 preprocessing contract) and docs/preprocessing.md (per-step
reference). When those change, this page changes with them.

Plain Streamlit components throughout: no custom CSS, no ASCII diagrams. Detail
lives inside expanders so each section reads as an outline first and a reference
second.
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

DIAGRAM = _REPO_ROOT / "assets" / "flow.png"

st.title("Documentation")
st.caption("The problem, every processing step, every model, and the reasoning behind the "
           "decisions that are easy to get wrong.")


# ====================== 1 · PROBLEM AND ARCHITECTURE ======================= #
def render_architecture():
    st.header("Three manipulation families, three places to look")
    st.markdown("""
Most deepfake detectors are trained to notice manufacturing residue: the seam where a generated
face was blended onto a real head, colour statistics that drift between the face and the neck,
warping around the jaw, upsampling patterns left by a GAN's decoder. Against a full-face swap
this works well, because the whole face is synthetic and the residue is everywhere. Nearly a
quarter of FakeAVCeleb's fakes are exactly that, `fsgan` and `faceswap` with no audio-visual
manipulation at all, and the three visual streams are the only part of this system that can
catch them.

A wav2lip-style forgery gives an artifact detector far less to work with. The generator is
handed a genuine video and a target audio track, and it repaints only the mouth region, frame by
frame, so the lips appear to speak the new words. Everything outside a small patch is the
original recording. The artifact budget is tiny, it sits in a region that moves and blurs
naturally, and video compression erases much of what is left. An `rtvc` voice clone is more
extreme still: the video track is untouched, so there is no visual evidence whatsoever.

Both are nonetheless catchable once you stop looking at the two tracks in isolation. A real
recording contains two views of the same physical event: light reflected off a moving mouth, and
pressure waves produced by that same mouth. The two views are causally locked together.
Synthesis breaks the lock. The generated mouth is plausible on its own and the audio is
plausible on its own, but their *joint* behaviour, the timing and the correspondence between
visible articulation and the sound produced, no longer belongs to one event.

Hence the design: artifact analysis for the swaps, cross-modal comparison for the rest, and no
standalone audio-only classifier anywhere. An audio model can report that a voice sounds
synthetic; it cannot report that the voice disagrees with the face, and only the disagreement
survives compression, resolution loss and a well-trained generator.
""")

    st.header("The signal chain")
    st.caption("Two parallel paths turn a clip into tensors, five streams turn those tensors into "
               "embeddings, one fusion head turns the embeddings into a decision.")

    # The diagram is the fastest way into the rest of this page; the tables below
    # give it the exact shapes it cannot carry.
    _, diagram_col, _ = st.columns([1, 3, 1])
    with diagram_col:
        if DIAGRAM.exists():
            st.image(str(DIAGRAM), width="stretch")
        else:
            st.warning(f"Architecture diagram not found at "
                       f"`{DIAGRAM.relative_to(_REPO_ROOT)}`.")

    st.markdown("""
| Path | Steps applied in order | Tensor produced |
|---|---|---|
| **Video** | 16 timestamps → MTCNN detect → margin-padded 224² crop → ImageNet normalise | `faces [16, 3, 224, 224]` |
| **Mouth** *(parallel, lip-sync only)* | mouth-corner landmarks from the same detect call → 96² ROI | `mouth [16, 3, 96, 96]` |
| **Audio** | PyAV decode → mono downmix → 16 kHz resample → leading-silence offset → 16 windows | `audio [16, 5600]` |

Each clip also carries a label: `0` real, `1` fake. Those tensors feed the five streams, every
one of which emits a 256-dimensional embedding; the five embeddings are concatenated into a
1280-dimensional vector, passed through an MLP, and squashed by a sigmoid into P(fake).

The two paths are not independent. They are indexed by the **same 16 timestamps**, so frame *i*
and audio window *i* describe the same instant of the same clip. Alignment by timestamp rather
than by sample index is what makes the cross-modal comparison meaningful at all: if the two
modalities drifted, every clip would look desynchronised and the streams would be measuring the
pipeline instead of the forgery.
""")

    st.header("The five streams")
    st.caption("What each stream is for. How each one works is the *Stream models* tab.")
    st.markdown("""
| Stream | Inputs | What it contributes | Stage |
|---|---|---|---|
| **Xception** | faces | Low-level blending and colour artifacts. The long-standing FaceForensics++ baseline, strongest of the classic CNNs on manipulation residue. | 3 |
| **EfficientNet-B0** | faces | A second artifact view from a different architecture family, and the lightest of the three, which is why it was built first and set the AUC 0.994 bar. | 2 |
| **DINOv2 (ViT-S/14)** | faces | Self-supervised features, learned with no fake/real labels at all. The bet is that generic visual structure transfers to manipulation methods never seen in training. | 3 |
| **Lip-sync** | mouth + audio | Cross-attention between AV-HuBERT (video) and Whisper (audio): a synchronisation-mismatch vector. | 4 |
| **Emotion** | faces + audio | Cross-attention between HSEmotions (face) and Wav2Vec2 (voice): an affect-consistency mismatch vector. | 5 |

Two things worth keeping straight while reading the rest of this page:

The three visual streams **never see audio**. They do not perform lip-sync checking or emotion
matching, and adding a fourth CNN would not give the system that ability. Cross-modal reasoning
exists only in the two cross-modal streams, which is precisely why they are worth the extra
engineering.

The cross-modal streams emit a fixed-size **feature vector, not a prediction**. Nothing in this
system transcribes speech or reads lips into text. Every comparison happens between learned
embeddings.
""")


# ===================== 2 · END-TO-END PREPROCESSING ======================== #
def render_pipeline():
    st.header("Input to tensors")
    st.caption("Every stage between an mp4 arriving and a batch reaching a model, in the order the "
               "code runs them, with the exact data representation at each handoff. The *Visual "
               "path* and *Audio path* tabs explain how the individual operations work; this tab is "
               "the control flow and the contract.")

    st.markdown("""
Two entry points share one implementation. `preprocessing/extract_clip.extract_clip` is the batch
path, called per clip by `dataset.ClipDataset` and warmed in bulk by `preprocessing/precache.py`.
The Preprocessing page calls the same `preprocessing/ops/` functions one step at a time so each
intermediate result is visible. There is no second implementation to drift.
""")

    st.subheader("Stage map")
    st.markdown("""
| # | Stage | Code | Result |
|---|---|---|---|
| 0 | Locate the dataset | `audit_dataset.find_dataset_root` | the directory holding `meta_data.csv` |
| 1 | Build the manifest | `manifest.manifest_from_meta`, `manifest.clip_label` | one row per clip |
| 2 | Split on identity | `build_splits.py`, `verify_splits.py` | `train/val/test.csv` |
| 3 | Decode audio | `ops.audio.decode` (PyAV) | `[channels, samples]` float32, native rate |
| 4 | Mono and 16 kHz | `ops.audio.downmix`, `ops.audio.resample` | `[samples]` float32 @ 16 kHz |
| 5 | Measure leading silence | `ops.audio.leading_silence_sec` | one float, seconds |
| 6 | Probe the video | `cv2.VideoCapture` properties | `fps`, `frame_count`, `duration` |
| 7 | Choose 16 timestamps | `ops.audio.sample_timestamps` | `[16]` float64, **shared by both paths** |
| 8 | Per frame: seek, detect, crop | `ops.faces.detect_crop` | `[16, 224, 224, 3]` uint8 (+ mouth) |
| 9 | Cut audio windows | `ops.audio.extract_windows` | `[16, 5600]` float32 |
| 10 | Write the cache | `extract_clip` | four files under `data/processed/<clip_id>/` |
| 11 | Normalise into a batch | `dataset.ClipDataset.__getitem__` | the tensors the model sees |

Fixed by `ops/constants.py`: `NUM_FRAMES=16`, `FRAME_SIZE=224`, `MOUTH_SIZE=96`, `AUDIO_SR=16000`,
`AUDIO_WINDOW_SEC=0.35`, `PIPELINE_VERSION=3`.
""")

    st.subheader("Representation at every boundary")
    st.caption("No image or audio file format is involved after decode. Frames never become PNGs "
               "and audio never becomes a WAV; everything is NumPy in memory and `.npy` on disk.")
    st.markdown("""
| Boundary | Type | Shape | dtype | Range |
|---|---|---|---|---|
| On disk (source) | mp4 container | 224x224 video + mono audio | h264 or mpeg4, aac or mp3 | n/a |
| After `cv2` decode | `ndarray` | `(224, 224, 3)` HWC, **RGB** | `uint8` | `[0, 255]` |
| After PyAV audio decode | `ndarray` | `(channels, samples)` | `float32` | about `[-1, 1]` |
| After downmix and resample | `ndarray` | `(samples,)` | `float32` | about `[-1, 1]` |
| `frames.npy` | `ndarray` | `(16, 224, 224, 3)` HWC | `uint8` | `[0, 255]` |
| `audio.npy` | `ndarray` | `(16, 5600)` | `float32` | source level, **not normalised** |
| `timestamps.npy` | `ndarray` | `(16,)` | `float64` | seconds |
| Model input, faces | `Tensor` | `(B, 16, 3, 224, 224)` CHW | `float32` | about `[-2.1, +2.6]` |
| Model input, audio | `Tensor` | `(B, 16, 5600)` | `float32` | source level |
| Model input, label | `Tensor` | `(B,)` | `int64` | `{0, 1}` |
""")

    with st.expander("**Stage 0 · What actually arrives**", expanded=True):
        st.markdown("""
21,544 clips, each a single mp4 holding both tracks. There are no loose frame images and no separate
audio files anywhere in the dataset.

Location is discovered, not configured: `find_dataset_root` globs `data/` three levels deep for a
`meta_data.csv` and takes its directory, which is the same rule the dashboard's picker uses, so the
two cannot disagree. `data/FakeAVCeleb_v1.2/` and `data/raw/FakeAVCeleb_v1.2/` both work, and clips
are nested `<category>/<race>/<gender>/<identity>/*.mp4`.
""")
        st.warning("""**The dataset is not uniformly encoded.** Verified with ffprobe:

| Category | Video codec | Audio codec |
|---|---|---|
| RealVideo-RealAudio | h264 | aac |
| RealVideo-FakeAudio | h264 | **mp3** |
| FakeVideo-RealAudio | h264, some **mpeg4** | aac |
| FakeVideo-FakeAudio | **mpeg4** | aac |

The wav2lip generator wrote its output with an MPEG-4 Part 2 encoder. OpenCV and PyAV decode all of
it identically, so the arrays are unaffected and the pipeline never notices. It matters only where
the file itself is handed to something pickier: no browser plays `mpeg4`, which is why the clip
player re-encodes (`dashboard/lib/media.playable_video_bytes`).""")

    with st.expander("**Stages 1-2 · Manifest and identity-disjoint splits**"):
        st.markdown("""
`manifest_from_meta` turns FakeAVCeleb's metadata into the table everything else reads:

```
clip_id, video_path, label, manipulation_type, method, source, target1, target2, race, gender
```

`video_path` is stored **relative to `data/`**, which is what lets a manifest attach to whichever
drop it points into rather than by filename.

The label comes from the category through `manifest.clip_label`, never from a filename: only
`RealVideo-RealAudio` is 0, the other three are 1.

`build_splits.py` splits on the `source` identity, never on the clip. The same person appears across
all four categories, so a clip-level split puts a real clip of someone in train and a forgery of
that same person in test, after which a model scores well by recognising the person.
`verify_splits.py` asserts zero identity overlap and zero file overlap, and is the gate that makes
the numbers mean anything. As configured: 1400 / 300 / 300 clips over 500 identities, undersampled
to a 1:3 real:fake ratio in every split.
""")
        st.warning("""**There are two labels, and they disagree on one category.** The manifest's
`label` is the *clip* label. A visual-only stream must be judged on the **video track**, so
`ClipDataset` takes `label_mode`:

- `'visual'` derives the label from `manipulation_type` (`FakeVideo-*` is fake), which makes
  `RealVideo-FakeAudio` count as **real**. Correct: its forgery is audio-only and leaves no trace in
  the frames.
- `'clip'` uses the manifest label, for the cross-modal streams and fusion, which do hear the audio.

Training a visual stream on the clip label asks it to detect a forgery absent from its input.""")

    with st.expander("**Stages 3-5 · Audio first, because the timestamps depend on it**",
                     expanded=True):
        st.markdown("""
The ordering looks backwards and is deliberate: the whole audio track is decoded and resampled
*before* a single frame is read, because the frame timestamps depend on a measurement taken from the
audio.

**Decode** goes through PyAV, which binds ffmpeg's libraries and bundles them, so no system ffmpeg
is needed. The result is `[channels, samples]` float32 at the file's native rate. This dataset
decodes as `fltp` (planar float), already in about `[-1, 1]`; the integer-PCM branch in
`ops.audio.decode` divides by `iinfo(dtype).max` and is simply not hit here.

**Downmix** is a channel mean. Stereo separation carries no forgery signal in this dataset and every
downstream audio encoder expects mono.

**Resample to 16 kHz** is filtered, not decimated: librosa low-pass filters with a windowed sinc
kernel before dropping samples, or everything above the new Nyquist frequency folds back as aliasing.
16 kHz is the rate Whisper and Wav2Vec2 were pretrained at, and speech carries little energy above
8 kHz.

**Leading silence** is measured with an energy gate at `top_db = 30`, meaning frames whose level sits
more than 30 dB below the clip's reference count as silence:
""")
        st.latex(r"\text{dB} = 20\,\log_{10}\!\left(\frac{a}{a_{\text{ref}}}\right)"
                 r"\quad\Longrightarrow\quad -30\,\text{dB} \approx \tfrac{1}{32}"
                 r"\ \text{of reference amplitude}")
        st.markdown("""
FakeAVCeleb's fake-audio clips carry extra near-silence at `t=0`. It is an artifact of assembly, not
of forgery, and a model left alone with it learns "silence at the start means fake", reports an
excellent AUC, and has learned nothing transferable. The measurement becomes the `start_offset` for
sampling:
""")
        st.code("""waveform   = resample(downmix(decode(video)), native_sr, 16_000)
silence    = leading_silence_sec(waveform, 16_000, top_db=30.0)
duration   = frame_count / fps                     # from cv2.VideoCapture
timestamps = sample_timestamps(duration, 16, 0.35, start_offset=silence)""", language="python")
        st.info("The waveform is never cut. Trimming shortens the audio while the frame timestamps "
                "stay put, which slides every window later in real time and manufactures exactly "
                "the desynchronisation the cross-modal streams exist to detect. "
                "`audit_dataset.py` also records the measurement per clip as `leading_silence_sec`, "
                "so later analysis can check the correction instead of trusting it.")

    with st.expander("**Stages 6-7 · The shared timestamp array**"):
        st.markdown("""
Video metadata comes from OpenCV: `CAP_PROP_FPS` and `CAP_PROP_FRAME_COUNT`, with
`duration = frame_count / fps`. `sample_timestamps` then places 16 evenly spaced instants, inset at
both ends by half an audio window so every window stays inside the recording:
""")
        st.latex(r"t_i = \mathrm{lo} + \frac{i}{N-1}\,(\mathrm{hi} - \mathrm{lo}), \qquad "
                 r"\mathrm{lo} = \max\!\left(\text{offset} + \tfrac{w}{2},\ \tfrac{w}{2}\right),"
                 r"\quad \mathrm{hi} = \max\!\left(T - \tfrac{w}{2},\ \mathrm{lo}\right)")
        st.markdown("""
This single array is the join between the modalities. Frames are seeked by *timestamp*, not by frame
index, which is what lets clips of differing frame rates share an index with an audio path that has
no concept of frames. The silence correction enters through the same array, so it moves both paths
together and cannot introduce the desynchronisation it exists to prevent.
""")
        st.warning("""**The duration is the video's, and the audio is often shorter.**
`frame_count / fps` measured 10.120 s on a clip whose audio track is 10.058 s. Windows are clamped
to the waveform, so the final window slides inward instead of staying centred on its frame: **62 ms
off** on that clip, while the other fifteen are exact. 62 ms is inside the range this project cares
about, since the 0.35 s window was chosen so that an error of order 100 ms is visible. The Audio tab
reports it as *Worst off-centre*. `extract_clip.py` behaves identically, so correcting it would
change every cached tensor.""")

    with st.expander("**Stage 8 · The visual path, per timestamp**"):
        st.markdown("""
For each of the 16 timestamps: `cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)`, read, convert BGR to RGB,
then one call to `detect_crop` does the whole front-end from a single detection.

| Step | What happens | Default |
|---|---|---|
| Detect | MTCNN, `keep_all=False`, so the highest-scoring face only | conf 0.90 |
| Gate | below threshold falls back to the **whole frame**, never a guess | counted in `num_faces_detected` |
| Crop | margin-padded bounding box, resized to 224x224 | `margin=0.20` |
| Mouth | 96x96 ROI centred between the two mouth-corner landmarks | derived from the same detection |

The crop clamps to the frame, so it never pads. Framing follows the detector, which means head
roll and off-centre framing survive into the tensor; five-point alignment used to remove them and is
parked in `docs/ideas.md`.

A clip that is quietly failing detection is visible rather than silently degraded, because the hit
rate is reported as "faces detected k/16".
""")
        st.warning("""**The mouth crop is computed and then discarded.** `detect_crop` returns
it, but `extract_clip` binds it to `_mouth` and never writes it, so the cache holds faces and audio
only. Nothing consumes it yet, since the lip-sync stream is Stage 4, and the dashboard derives it
live for inspection. Caching it will need a `PIPELINE_VERSION` bump.""")

    with st.expander("**Stage 9 · The audio path, per timestamp**"):
        st.markdown("""
One window of `AUDIO_WINDOW_SEC = 0.35` per timestamp, clamped to the waveform and zero-padded to a
fixed length so the result is rectangular: 5600 samples at 16 kHz, giving `[16, 5600]`.

Windows are centred per frame rather than concatenated into one track because the cross-modal
streams compare frame *i*'s mouth shape against the sound at that instant. The length is a
compromise: long enough to contain a phoneme and its transition, short enough that a synchronisation
error of order 100 ms shows as a mismatch rather than being averaged away.
""")

    with st.expander("**Stage 10 · The cache**"):
        st.markdown("""
Written per clip to `data/processed/<clip_id>/`, holding **un-normalised uint8** frames:
""")
        st.code("""frames.npy      (16, 224, 224, 3)  uint8      # HWC, RGB, NOT normalised
audio.npy       (16, 5600)         float32    # 16 kHz windows, source level
timestamps.npy  (16,)              float64    # seconds
version.txt     "3"                           # PIPELINE_VERSION""", language="text")
        st.markdown("""
Storing uint8 rather than normalised float32 is a deliberate four-fold saving, and it keeps the
expensive part (decode plus 16 MTCNN detections) separate from the cheap part (a divide and a
subtract). It also means the cached pixels are directly viewable.

`version.txt` carries `PIPELINE_VERSION`, currently **4**. Bumping it invalidates every stale cache,
so a change to the crop is safe: the pixels changed, the old cache must not be trusted, and nothing
has to be deleted by hand. The history in `ops/constants.py` is the record of what moved: v1 plain
MTCNN crop, v2 added alignment and silence-aware sampling, v3 padded aligned crops with black
instead of reflecting a second face into them, v4 removed alignment and returned to the bbox crop.

`precache.py` warms the whole cache in parallel worker processes, pinning MTCNN to CPU so several
workers do not contend over one GPU.
""")

    with st.expander("**Stage 11 · Normalisation, and the tensors the model sees**", expanded=True):
        st.markdown("""
Normalisation happens per batch in `ClipDataset.__getitem__`, not at cache time. Two operations,
then a transpose:
""")
        st.latex(r"x_{\text{norm}} = \frac{x/255 - \mu}{\sigma}, \qquad "
                 r"\mu = (0.485,\,0.456,\,0.406), \quad \sigma = (0.229,\,0.224,\,0.225)")
        st.markdown("""
The constants are the per-channel mean and standard deviation of the ImageNet training set, which is
why they differ between R, G and B: natural images carry more red and less blue. All three visual
backbones are ImageNet-pretrained through `timm`, and a pretrained network's early filters are
calibrated to the distribution they were trained on. Raw `[0, 255]` input shifts every activation in
the first layers, and the network spends its early budget correcting an offset that costs nothing to
remove here.

**The output is not `[0, 1]` or `[-1, 1]`.** Subtracting a mean puts everything darker than that mean
below zero, which is the point of centring:

| Input pixel | R | G | B |
|---|---|---|---|
| 0 (black) | −2.118 | −2.036 | −1.804 |
| 128 (mid grey) | +0.074 | +0.205 | +0.426 |
| 255 (white) | +2.249 | +2.429 | +2.640 |
| **maps to exactly 0.0** | **124** | **116** | **104** |

So on a dark indoor face crop most values are negative: one measured crop with a raw mean of 62 came
out 86% negative. That is expected, and it does not trouble the network, because the first operation
is a convolution with signed weights and ReLU comes after it, not before the input.
""")
        st.code("""faces  (B, 16, 3, 224, 224)  float32   # ImageNet-normalised, HWC -> CHW
audio  (B, 16, 5600)         float32   # 16 kHz windows, NOT normalised
label  (B,)                  int64     # per label_mode""", language="text")
        st.info("**The audio is never normalised in the contract.** Nothing in `ops/audio.py` or "
                "`extract_clip.py` touches amplitude, so `audio.npy` keeps whatever level the source "
                "had. `rms_normalize` exists but sits in the extras, off by default. Defensible, "
                "since Whisper and Wav2Vec2 both normalise internally, but worth being deliberate "
                "about: if real and fake audio differ systematically in loudness, that is an "
                "exploitable shortcut sitting in the input, exactly like the leading silence was.")

    st.subheader("Where the dashboard differs from the batch pipeline")
    st.caption("Worth knowing before reading a number off the Preprocessing page and assuming "
               "training saw the same thing.")
    st.markdown("""
| | Batch pipeline | Preprocessing page |
|---|---|---|
| Leading-silence offset | Applied, `start_offset=leading_silence` | Measured and shown, **not applied**; sampling starts at half a window |
| Frame count and window | Fixed at 16 and 0.35 s | Sliders, 4-32 frames and 0.10-1.00 s |
| Normalisation | Always | A toggle, so the un-normalised tensor is inspectable |
| Enhancement extras | Never | Available, all off by default |
| Mouth crop | Computed, discarded | Shown when enabled |
| Cache | Written to `data/processed/` | Never written; the page is read-only |

With every extra switched off, the page reproduces the batch pipeline's operations step for step.
The silence offset and the fixed 16 / 0.35 contract are the two differences that remain by design.
""")


# ========================== 3 · VISUAL PATH ================================ #
def render_visual():
    st.header("Visual path")
    st.caption("Five steps, applied per sampled frame, producing one [16, 3, 224, 224] tensor per "
               "clip plus a parallel mouth tensor.")

    with st.expander("**1 · Frame sampling**: choosing 16 moments", expanded=True):
        st.markdown("""
Sixteen evenly spaced timestamps across the clip, inset at both ends by half an audio window so
every frame's ±0.175 s of audio still lies inside the recording. Without the inset, the first and
last windows would be zero-padded on one side for every clip in the dataset, which is a constant
the model could learn from.

Sampling by timestamp rather than by frame index is deliberate. Clips vary in frame rate, and the
audio path has no notion of frames at all; a timestamp is the only index both paths can share.
The start of the range is also pushed past any measured leading silence, and pushed for both
modalities together, so the correction cannot introduce the desynchronisation it exists to avoid.

Sixteen frames is a compromise: enough for a temporal model to see articulation develop, few
enough that a backbone runs over the whole batch on a 6 GB GPU. It is a knob on the Config tab,
so the cost of changing it is visible immediately.
""")

    with st.expander("**2 · Face detection**: how MTCNN actually works"):
        st.markdown("""
MTCNN (Zhang et al., 2016) is a **cascade of three small convolutional networks** that get
progressively more expensive and progressively more selective. The design assumption is that the
overwhelming majority of image regions are obviously not faces, so they should be rejected by
something cheap, and only the survivors should be shown to something accurate.

The image is first resized repeatedly by a factor of about 0.709, one half-octave, down to the
minimum detectable face size, producing an **image pyramid**. This lets a single detector with a
fixed 12×12 receptive field find faces at any scale: rather than searching for faces of many
sizes, the network searches for one size in many images.

| Stage | Input | Job | Emits |
|---|---|---|---|
| **P-Net** | each pyramid level | Fully convolutional, so it sweeps a whole level in one pass and proposes candidate windows. Deliberately shallow and permissive, tuned for recall and expected to return many false positives. | face score, 4 box offsets |
| **R-Net** | 24×24 crops | A fully-connected refinement network. Rejects the bulk of P-Net's false positives and tightens the boxes that survive. | face score, 4 box offsets |
| **O-Net** | 48×48 crops | The deepest of the three, and the only one whose landmarks are used. Produces the final decision and the 5 facial landmarks alignment depends on. | face score, 4 box offsets, **5 landmarks** |

Between stages, two operations do the actual pruning. **Bounding-box regression** applies each
network's predicted offsets, so the box reaching the next stage is better centred than the one
that entered it. **Non-maximum suppression** then sorts surviving boxes by score and discards any
box whose intersection-over-union with a higher-scoring box exceeds a threshold, collapsing the
cluster of overlapping detections around each face into one.

The three networks are trained jointly with a multi-task loss: cross-entropy for the
face/not-face decision, squared error for the box offsets, and squared error for the landmark
coordinates, each weighted differently per stage. P-Net and R-Net weight detection heavily,
O-Net weights landmark accuracy heavily, because landmarks are its distinctive output. Training
also uses online hard example mining: within each mini-batch only the highest-loss samples
contribute gradients, so capacity is spent on cases the network is currently failing rather than
on background it already rejects.

The five landmarks are the left eye, right eye, nose tip, and the two mouth corners: ten
coordinates, and everything downstream depends on them.

**How this pipeline uses it.** The detector runs with `keep_all=False`, returning only the
highest-scoring face; these clips are single-speaker, so taking the most confident detection
avoids a policy decision about which face matters. A detection must clear a confidence
threshold, 0.90 by default. Frames below it fall back to the whole frame rather than to a guess,
and the Visual tab reports the hit rate as "faces detected k/16", so a clip that is quietly
failing detection is visible rather than silently degraded.

**Why MTCNN and not something newer.** It ships with the pinned environment via
`facenet-pytorch`; it returns the five landmarks the next step needs from the same forward pass
rather than requiring a second model; and it is fast enough to run CPU-side inside pre-caching
workers while the GPU is busy. The honest costs: it is a 2016 detector that struggles with
profile views and heavy occlusion, and its five landmarks are the minimum the mouth ROI can work
from. AV-HuBERT's own preprocessing would prefer 68.
""")

    with st.expander("**3 · Crop**: a margin-padded bounding box"):
        st.markdown("""
The detected box is padded by `margin` (0.20 by default) on each side, clamped to the frame, and
resized to 224x224 with cubic interpolation. Because it clamps, it can never introduce padding of
its own: every pixel in the crop came from the source frame.

Framing therefore follows the detector, so a rolled or off-centre head stays rolled and off-centre,
and the temporal model sees rigid head motion on top of expression and articulation.
""")
        st.info("**Five-point alignment used to sit here and was removed.** It warped the face onto "
                "a canonical ArcFace template so eyes and mouth landed on the same pixels in every "
                "frame. It is parked in `docs/ideas.md` along with the two traps it carried, the "
                "black-padding fix and the template inset, so bringing it back does not start from "
                "scratch. Removing it bumped `PIPELINE_VERSION` to 4, since the cached crop pixels "
                "changed.")

    with st.expander("**4 · Mouth ROI**: a parallel output, not a substitute"):
        st.markdown("""
A 96×96 crop centred between the two mouth-corner landmarks, produced from the *same* detect call
as the face, so it costs no additional detection. It is a parallel output: the visual and emotion
streams continue to consume the 224² face, and only the lip-sync stream reads the mouth.

The resolution is chosen for the consumer rather than for appearance. Lip-sync models operate on
small mouth crops because the informative content is the shape and motion of the aperture, not
skin texture, and a smaller crop keeps a video transformer's sequence length affordable.

A caveat worth carrying into Stage 4: AV-HuBERT's published preprocessing expects a grayscale,
68-landmark, mean-face-aligned mouth ROI. Matching that exactly requires a 68-landmark model and
therefore a new dependency. Today's landmark-centred RGB crop is the best available from MTCNN's
five points, and the gap is a known item rather than an oversight.
""")

    with st.expander("**5 · ImageNet normalisation**: once, here, for everyone"):
        st.latex(r"x_{\text{norm}} = \frac{x/255 - \mu}{\sigma}, \qquad "
                 r"\mu = (0.485,\,0.456,\,0.406), \quad \sigma = (0.229,\,0.224,\,0.225)")
        st.markdown("""
All three visual backbones are ImageNet-pretrained through `timm`, and a pretrained network's
early filters are calibrated to the input distribution they were trained on. Feeding raw [0, 255]
pixels into weights that expect roughly zero-mean, unit-variance channels shifts every activation
in the first layers, and the network spends its early training budget correcting an offset that
costs nothing to remove here.

The per-channel constants are the mean and standard deviation of the ImageNet training set, which
is why they differ slightly between R, G and B. Normalisation is applied once, in preprocessing,
rather than inside each stream, so it is impossible for two streams to disagree about the scaling
of their inputs.

On the Visual tab this step displays de-normalised frames, the transform inverted, purely so the
result is viewable. The tensor handed to the model is the normalised one, and the printed value
range underneath it is the honest check.
""")


# =========================== 3 · AUDIO PATH ================================ #
def render_audio():
    st.header("Audio path")
    st.caption("Short, but it contains the single most important correctness detail in the whole "
               "preprocessing stage.")

    with st.expander("**1 · Decode**: PyAV", expanded=True):
        st.markdown("""
Decoding goes through PyAV, which binds ffmpeg's libraries directly and bundles them, so no
system ffmpeg installation is required and the same code runs on every machine in the team. The
result is a `[channels, samples]` float32 array at the file's native sample rate, usually
44.1 kHz for this dataset.

The earlier implementation shelled out to an ffmpeg binary through `ffmpeg-python`. That
dependency was removed: a subprocess per clip is slow at dataset scale, and requiring a correctly
installed system binary is exactly the kind of environment drift that makes a pipeline
unreproducible.
""")

    with st.expander("**2 · Mono downmix and 16 kHz resample**"):
        st.markdown("""
The downmix is a channel mean. Stereo separation in this dataset carries no forgery signal, since
the fakes are generated as mono speech and mixed identically, so keeping two channels would double
the data for no information, and every downstream audio encoder expects mono anyway.

Resampling is a filtered operation, not a subsampling one. Dropping every other sample would fold
all content above the new Nyquist frequency back into the audible band as aliasing, so the signal
is low-pass filtered before decimation; librosa's resampler does this with a windowed sinc kernel.

16 kHz is not an arbitrary choice. It is the rate Whisper and Wav2Vec2 were pretrained at, and
their convolutional front-ends assume it. It also sits comfortably above speech content, which
carries very little energy above 8 kHz, so the higher native rate costs samples and memory
without adding usable signal.
""")

    with st.expander("**3 · Leading silence**: FakeAVCeleb's shortcut"):
        st.markdown("""
In FakeAVCeleb, clips with fake audio carry additional near-silence at t=0. It is an artifact of
how the dataset was assembled, not a property of forgery. Left alone, a model learns "silence at
the start implies fake", reports an excellent AUC, and has learned nothing that transfers to a
single real-world video.

The duration of the near-silent prefix is measured with an energy gate: frames whose RMS level
falls more than `top_db = 30` below the clip's reference level count as silence, where level is
measured in decibels.
""")
        st.latex(r"\text{dB} = 20\,\log_{10}\!\left(\frac{a}{a_{\text{ref}}}\right) "
                 r"\quad\Longrightarrow\quad -30\,\text{dB} \approx \tfrac{1}{32}"
                 r"\ \text{of reference amplitude}")
        st.markdown("""
The fix is **not** to trim the audio. Trimming shortens the waveform while the frames stay where
they were, which desynchronises the two modalities and corrupts the exact signal the cross-modal
streams measure. Instead the measurement is recorded per clip in the manifest as
`leading_silence_sec` by `audit_dataset.py`, and that offset is passed into `sample_timestamps`
so frame sampling and audio sampling both begin past the silence. The shortcut is removed and the
alignment is preserved.

This matters more than it looks. Dataset bias is the most common way a deepfake result turns out
to be worthless. The measurement stays in the manifest rather than being applied silently, so any
future analysis can check the correction rather than trust it.
""")

    with st.expander("**4 · Window extraction**"):
        st.markdown("""
One window of `window_sec`, 0.35 s by default, centred on each of the 16 frame timestamps,
clamped to the bounds of the clip and zero-padded to a fixed length so the result is rectangular.
At 16 kHz that is 5600 samples per window, giving `[16, 5600]`. A window that would run off either
end slides inward rather than being centred, so the last window is off-centre by however much the
video duration (`frame_count / fps`) exceeds the audio track: about 60 ms on a typical FakeAVCeleb
clip. The Audio tab reports this as *Worst off-centre*.

Windows are centred per frame rather than concatenated into one long track because the
cross-modal streams compare frame *i*'s mouth shape against the sound that occurred at that
moment. The length is a compromise: long enough to contain a complete phoneme and its transition,
short enough that a synchronisation error of order 100 ms shows up as a visible mismatch rather
than being averaged away.
""")

    st.subheader("Extras")
    st.caption("Off by default, and deliberately separate from the stored contract.")
    st.markdown("""
None of these are part of the stored contract. With every one disabled, the dashboard reproduces
the batch pipeline exactly. They sit together under the **Enhancement** step on the Preprocessing
page, and split into two kinds:

**Enhancement** (sharpen, denoise, CLAHE) asks whether cleaning an input helps or merely removes
the artifacts the detector was relying on. CLAHE in particular equalises contrast in local tiles,
which can amplify precisely the blending seams a visual stream is hunting for, or wash them out.

**Degradation** (Gaussian blur, JPEG re-compression, downscale-then-upscale, additive noise at a
target SNR, bandpass filtering) is the more important half. Real-world video has been re-encoded
by a phone, a messaging app and a social platform before anyone sees it. A detector that only
works on pristine academic clips has not been shown to work. These controls let you find the
quality level at which a stream stops functioning, which is a reportable result in its own right.

The mel-spectrogram control is a *view*, not a pipeline output: the short-time Fourier transform
is taken with a Hann window, the power spectrum is projected onto triangular filters spaced on
the mel scale, and the result is converted to decibels. It is there because the periodic
artifacts of a vocoder are far easier to see in a spectrogram than in a waveform.
""")
    st.latex(r"m = 2595\,\log_{10}\!\left(1 + \frac{f}{700}\right)")
    st.caption("The mel scale: roughly linear below 1 kHz, logarithmic above, matching how human "
               "pitch perception spaces frequencies.")


# ============================= 4 · STREAMS ================================= #
def render_streams():
    st.header("Visual streams")
    st.caption("All three are the same config-driven module with a different backbone name, so "
               "adding a fourth is a configuration change rather than an engineering project.")
    st.info("This tab is the design. The **Streams** section is the same thing running: pick a "
            "clip on Preprocessing, open Streams and run it, and every stage in the table below "
            "appears as the activations it actually produced on that clip.")
    st.markdown("""
| Stage in the module | Operation | Shape out |
|---|---|---|
| Input | face-crop sequence from preprocessing | `[B, 16, 3, 224, 224]` |
| Fold | frames folded into the batch dimension | `[B*16, 3, 224, 224]` |
| Backbone | timm model, optionally frozen | `[B*16, D]` |
| Unfold | back into per-clip sequences | `[B, 16, D]` |
| Temporal model | BiLSTM, GRU or mean-pool | `[B, H]` |
| Head | `Linear(H → 256)` + LayerNorm | `[B, 256]` |

Frames are folded into the batch dimension so the backbone sees a flat stack of images, then
unfolded back into sequences for the temporal model. The projection to a common 256 dimensions at
the end is what makes streams of different native widths interchangeable at the fusion stage.

**Why a temporal model at all.** A single frame of a good forgery can be flawless while the
sequence is not. Blending boundaries flicker between frames, generated texture fails to persist
the way real skin does, and synthetic mouth motion is often too smooth because the generator has
no reason to reproduce the small irregularities of real articulation. Mean-pooling the frame
embeddings discards ordering entirely and cannot represent any of this. A bidirectional LSTM
reads the sequence forwards and backwards, so the representation of frame 8 is informed by frames
1–7 and 9–16, and its gates let it retain or discard information across the sequence rather than
being overwritten at each step. With 256 hidden units per direction, the concatenated state is
512-dimensional before projection.

**Frozen or fine-tuned.** Freezing the backbone means the pretrained weights do not move and only
the temporal model and head are trained. It is dramatically cheaper, because per-frame embeddings
can be computed once and cached, after which an epoch costs almost nothing, and it is the realistic
default on a 6 GB GPU. Fine-tuning lets the backbone specialise on forgery artifacts and
generally scores better, at a cost that scales with five backbones. The decision is deliberately
left open until Stage 6 rather than assumed.
""")

    with st.expander("**Xception**: depthwise separable convolutions"):
        st.markdown("""
Xception replaces the standard convolution with a **depthwise separable** pair: first a single
k×k kernel per input channel, applied independently and mixing nothing across channels; then a
1×1 convolution that mixes channels and mixes nothing spatially. The hypothesis in the name,
"extreme Inception", is that spatial correlation and cross-channel correlation are separable
enough to be modelled by two dedicated operations rather than one entangled one.
""")
        st.latex(r"\frac{\text{separable cost}}{\text{standard cost}} = "
                 r"\frac{1}{C_{\text{out}}} + \frac{1}{k^{2}}")
        st.markdown("""
For k=3 and typical widths that is roughly 8–9× fewer multiply-accumulates. The saved parameters
are spent on depth: 36 convolutional layers in 14 modules, all but the first and last wrapped in
residual connections. It is the reference baseline in the FaceForensics++ benchmark, where it
outperformed the alternatives on manipulation detection, which is why it is in this system rather
than a newer general-purpose classifier.
""")

    with st.expander("**EfficientNet-B0**: compound scaling"):
        st.markdown("""
EfficientNet's contribution is the observation that depth, width and input resolution should not
be scaled independently. Deeper networks with narrow layers underuse their capacity; wider
networks without resolution have nothing fine-grained to look at. Compound scaling ties all three
to a single coefficient φ.
""")
        st.latex(r"\text{depth} = \alpha^{\varphi},\quad \text{width} = \beta^{\varphi},\quad "
                 r"\text{resolution} = \gamma^{\varphi} \qquad "
                 r"\text{subject to } \alpha\beta^{2}\gamma^{2} \approx 2")
        st.markdown("""
The constraint keeps FLOPs growing as roughly 2^φ, so each step up the family is a predictable
doubling. B0 is the base network found by neural architecture search, built from inverted-residual
MBConv blocks with squeeze-and-excitation, at about 5.3M parameters.

It was built first for a practical reason: it is the lightest of the three and it fit the 6 GB
laptop that produced the earlier results. That build reached test accuracy 0.963, AUC 0.994,
F1 0.974 and EER 0.018, in-distribution: the bar the rebuild has to clear.
""")

    with st.expander("**DINOv2 (ViT-S/14)**: self-supervised, and the generalisation bet"):
        st.markdown("""
DINOv2 is a vision transformer trained with **no labels of any kind**. Two networks are involved:
a student, and a teacher whose weights are an exponential moving average of the student's. Both
see different augmented crops of the same image, the teacher getting large global crops and the
student global and small local ones, and the student is trained to match the teacher's output
distribution. Learning to predict a global view from a local one forces the representation to
encode what the object *is*, not where the crop was taken.

Two mechanisms keep this from collapsing to a constant output: the teacher's distribution is
*centred* using a running mean, which prevents any one dimension from dominating, and *sharpened*
with a low temperature, which prevents it from flattening into a uniform distribution. DINOv2 adds
a masked-patch objective on top and trains on a large curated corpus.

The reason it is here is specific. Xception and EfficientNet were trained to recognise categories
and are then fine-tuned to recognise the artifacts of the forgeries they were shown, which is
exactly the kind of learning that fails on a generator it has never seen. DINOv2's features were
never fitted to any manipulation, so the expectation is that it degrades more gracefully on unseen
methods. Whether that expectation survives contact with the ablation is one of the results this
project is meant to produce.
""")

    st.header("Cross-modal streams")
    st.caption("Both streams share one mechanism: one modality asks the questions, the other "
               "supplies the evidence, and how cleanly the evidence answers is the signal.")
    st.latex(r"\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^{\top}}"
             r"{\sqrt{d_k}}\right)V")
    st.markdown("""
Read it as a soft lookup. Each row of *Q* is one moment of audio asking a question. Each row of
*K* is one moment of video offering itself as an answer. The dot product scores every audio moment
against every video moment, the softmax turns each row of scores into weights that sum to one, and
the weighted sum of *V* is "the video evidence that best explains this sound".

The √dₖ divisor is not cosmetic. The dot product of two d-dimensional vectors with
unit-variance components has variance d, so as dimension grows the scores spread out, the softmax
saturates into a near one-hot distribution, and its gradient approaches zero. Dividing by the
standard deviation keeps the distribution in a regime where learning is possible.

The forensic intuition: in a genuine recording the attention mass concentrates near the diagonal
with a small, consistent lag, because sound arrives at a fixed offset from the articulation that
produced it. In a re-synthesised clip the correspondence is smeared or drifts, because the mouth was
generated to match the audio only in aggregate. The stream does not decide anything about this; it
hands the resulting vector to fusion, which learns what to do with it.
""")

    with st.expander("**Lip-sync stream**: AV-HuBERT + Whisper (Stage 4)"):
        st.markdown("""
**AV-HuBERT** encodes the mouth-crop sequence and supplies Key and Value. It was pretrained by
masking parts of an audio-visual input and predicting cluster assignments of the hidden features,
with the clusters themselves refined over successive iterations. That objective forces its video
encoder to represent mouth motion in a space where audio is predictable, which is precisely the
property a synchronisation check needs.

**Whisper** encodes the audio and supplies Query. Only its encoder is used: log-mel input,
transformer stack, hidden states out. The decoder is never run, so no transcript is ever produced.
This is a deliberate divergence from Bohacek & Farid (2024), who lip-read the video, transcribe
the audio and compare the *words*. That approach is kept as a benchmark; it is not what is
implemented here, because word-level comparison discards the sub-word timing where a sync failure
actually lives.

The closest prior art is SyncNet (Chung & Zisserman, 2016), which learned an embedding space where
matching audio and video windows are close together and offset ones are far apart. This stream
generalises the same idea with cross-attention over a full clip.
""")

    with st.expander("**Emotion stream**: HSEmotions + Wav2Vec2 (Stage 5)"):
        st.markdown("""
**HSEmotions** reads facial expression from the face crops and supplies Key and Value.
**Wav2Vec2** reads vocal affect from the waveform and supplies Query; its convolutional feature
encoder consumes raw audio directly, and its transformer was pretrained contrastively to
distinguish true latent speech units from distractors, then adapted for emotion recognition.

In both cases the penultimate embedding is taken, never the emotion class probabilities.
Collapsing to eight labels would throw away the intensity and ambiguity the mismatch depends on,
and would make the stream's output depend on a taxonomy neither model agrees on. The premise, from
*Emotions Don't Lie* (Mittal et al., 2020), is that a forger who matches lip motion to audio
rarely also matches affect: the voice carries stress the face does not show.
""")


# ====================== 5 · FUSION AND EVALUATION ========================== #
def render_fusion():
    st.header("Fusion")
    st.caption("Feature-level, not score averaging.")
    st.markdown("""
Each stream produces a clip-level embedding. Each embedding is projected with a linear layer and
layer normalisation to a shared `common_dim` of 256, so streams with different native widths
become compatible. Every projected embedding is written to a feature store keyed by clip ID.
Fusion reads that table, concatenates the included streams into one 1280-dimensional vector,
passes it through an MLP, and applies a sigmoid to obtain the probability that the clip is fake.
Threshold at 0.5 for the label.

**Why not average scores.** Score averaging can only combine five opinions after each stream has
already discarded everything except a number. Feature fusion lets the MLP learn *interactions*.
"Artifact evidence is weak **but** lip-sync mismatch is strong" is the exact signature of a
wav2lip forgery, and it is a conjunction no weighted average of scores can express: averaging a
low score with a high one produces a middling score regardless of which stream said what.

The feature store also has a practical consequence. With frozen backbones, each stream's
embeddings are computed once and cached, after which retraining fusion or running an ablation is a
matter of minutes over a table rather than hours over video.
""")
    st.warning("""**The trade-off, stated honestly.** Feature-level fusion does not hand you an
interpretable "stream X contributed 30%" number, which per-stream scores plus a logistic
regression's weights would give for free. That interpretability has to be recovered a different
way: by running the ablation and comparing fused metrics across stream subsets, not by reading
fusion weights. This is a real cost of the design, not an oversight.""")

    st.header("Evaluation")
    st.caption("What gets measured, and what would hide a failure.")
    st.markdown("""
**Ablation.** Every stream is evaluated alone and in combination. Because fusion is feature-level,
removing a stream means changing the MLP's input dimension and retraining the head, not masking a
column. The expected finding is that Xception and EfficientNet overlap, since both are
artifact-focused CNNs likely to catch the same forgeries, in which case one is redundant and
should be dropped. Dropping a stream because the data said so is a result, not a failure.

**Per-category and per-method reporting.** Aggregate accuracy actively conceals the outcome that
matters here. A model can score 96% overall by handling face swaps well while failing on every
wav2lip clip, and wav2lip is the case this project exists to catch. Metrics are therefore always
broken down by FakeAVCeleb category and manipulation method, logged every epoch rather than
computed once at the end.

**Which metrics.** Accuracy alone is unreliable under class imbalance. AUC measures ranking
quality independently of any threshold choice. EER, the point where the false accept and false
reject rates are equal, is the standard single number in forensic biometrics and is directly
comparable with published work. F1 keeps precision and recall visible, which is where an
imbalanced test set does its damage.

**Generalisation.** Tested twice, because in-distribution numbers on a single dataset predict
almost nothing. A manipulation method is held out of training entirely, and the final system is
evaluated on Deepfake-Eval-2024, an in-the-wild collection never used for training at any point.
""")


# ========================== 6 · DATA AND SPLITS ============================ #
def render_data():
    st.header("Dataset: FakeAVCeleb v1.2")
    st.markdown("""
Chosen over FaceForensics++ and Celeb-DF for one decisive reason: it contains genuinely
manipulated audio. Visual-only datasets cannot test cross-modal mismatch at all, so they cannot
validate the central claim of this project.

| Category | Video | Audio | Why it is in the set |
|---|---|---|---|
| **RVRA** | real | real | The only genuine class, and the scarce one: roughly 500 clips against 19,500 fakes. |
| **RVFA** | real | fake | Real footage with a synthesised voice. Invisible to every visual-only stream by construction. |
| **FVRA** | fake | real | Manipulated face against the original audio. |
| **FVFA** | fake | fake | Both tracks manipulated. The largest category, and the one where all three detection principles have something to find. |

The manipulation *method* is a separate column from the category, and it is the axis that maps onto
the three detection principles: `faceswap` (730) and `fsgan` (3,964) are pure swaps, `wav2lip`
(9,602) plus `fsgan-wav2lip` (3,553) and `faceswap-wav2lip` (2,717) are lip-sync repaints, `rtvc`
(500) is a voice clone over genuine video, and `real` (500) is the genuine class.

Both breakdowns are reported. Aggregate accuracy is dominated by the 15,872 wav2lip-derived clips,
so a system that handles those well and pure face swaps badly, or the reverse, has failed at the
task even when its headline number looks acceptable.
""")

    st.header("Labels")
    st.caption("Authenticity is per stream, not per clip.")
    st.warning("""A visual-only stream is judged on the **video track's** authenticity, not the
clip's. `FakeVideo-*` is fake and `RealVideo-*` is real, **including RealVideo-FakeAudio**, whose
forgery is audio-only and therefore correctly invisible to a model that never hears it. This looks
like a labelling bug and is not one. It is the reason the cross-modal streams exist, and a visual
stream that appeared to catch RVFA would be reading a dataset artifact.""")

    st.header("Splits")
    st.caption("Identity-disjoint, built once, never re-split.")
    st.markdown("""
The same identities appear across all four categories. If a real clip of a person lands in train
while a forgery derived from that same person lands in test, the two share background, lighting,
framing, wardrobe and face. A model can then score spectacularly by recognising the *person*, and
the reported AUC measures memorisation rather than detection. This is the most common way deepfake
projects produce results that do not survive contact with new data.

Splits are therefore constructed once, on the `source` identity, by `build_splits.py`, checked by
`verify_splits.py` which asserts zero identity and zero file overlap, and never re-split randomly
downstream. As configured, that yields 1400 / 300 / 300 clips over 500 source identities.

**Class balance** is handled by two independent mechanisms that are easy to confuse. The natural
distribution is about 40:1 fake:real. `build_splits.py` undersamples fakes to a 1:3 real:fake ratio
in *every* split, including validation and test, so precision, recall and F1 remain meaningful on a
300-clip evaluation set and remain comparable to the earlier build's numbers, which were measured
the same way. Separately, training applies class weighting and a weighted sampler on top of that.
The first decision changes what the metrics mean; the second only changes how the model is
optimised.
""")

    st.header("On disk")
    st.markdown("""
| Path | Contents |
|---|---|
| `data/<drop>/` | A dataset, gitignored. FakeAVCeleb v1.2 is 21,544 clips plus `meta_data.csv`, nested `<category>/<race>/<gender>/<identity>/*.mp4`. |
| `data/processed/<clip_id>/` | Per-clip cache: `frames.npy`, `audio.npy`, `timestamps.npy`, `version.txt`. |
| `data/*.csv` | `full_manifest.csv` and the `train` / `val` / `test` splits. |

The dataset's location is **not** configured. Anything under `data/` holding a `meta_data.csv` is a
drop. `data/FakeAVCeleb_v1.2/` and `data/raw/FakeAVCeleb_v1.2/` both work, and two drops can sit
side by side. `preprocessing/audit_dataset.find_dataset_root` and the dashboard's picker apply the
same rule, so neither can disagree with the other about where the clips are.
""")
    st.warning("""**The clips are not all the same codec.** FakeAVCeleb's real footage is H.264, but
the wav2lip generator wrote its output with an MPEG-4 Part 2 encoder, so every `FakeVideo-FakeAudio`
clip and some `FakeVideo-RealAudio` clips are `mpeg4`.

This is invisible to the pipeline, since OpenCV and PyAV decode both without comment, but no browser can
play `mpeg4`, so `st.video` rendered a blank player for precisely the lip-sync forgeries this project
exists to detect. The Preprocessing page therefore re-encodes to H.264 on the way to the player
(`dashboard/lib/media.playable_video_bytes`), captions the clip when it has done so, and leaves the
file on disk untouched. Worth remembering before trusting any tool that assumes one codec per
dataset.""")
    st.markdown("""
The per-clip cache carries a `version.txt` stamped with `PIPELINE_VERSION`. Bumping that constant
invalidates every stale cache so it is transparently re-extracted on next access, which is what
makes a change to the crop safe: the pixels changed, so the cache must not be
trusted, and nothing has to be deleted by hand.

Two consequences of that discovery rule are worth knowing while using the picker. A drop's manifest
is derived in memory from its `meta_data.csv`, so a freshly unzipped dataset is selectable before
`audit_dataset.py` has ever run. And a manifest CSV is attached to whichever dataset its
`video_path` column points into rather than by filename, which is how the pipeline's flat
`data/train.csv` finds its way onto the right drop. The rescan button re-runs the whole discovery.
""")


def _measured(render):
    """Render a tab body inside a bounded column.

    app.py sets layout="wide" for the Preprocessing page's frame grids, and
    Streamlit has no per-page override, so a reference page of paragraphs would
    otherwise be set across 1200px. 7:2 is wider than the Overview page's column
    because several tabs carry four-column tables that need the room.
    """
    body, _ = st.columns([7, 2], gap="large")
    with body:
        render()


# Ordered the way a clip moves through the system. "Input → tensors" is the
# end-to-end control flow; "Visual path" and "Audio path" are the per-operation
# reference it calls into. "Stream models", not "Streams": the architecture tab
# already lists what the five streams are for, and two tabs with the same name is
# what made this page hard to navigate.
tabs = st.tabs(["Problem & architecture", "Input → tensors", "Visual path", "Audio path",
                "Stream models", "Fusion & evaluation", "Data & splits"])
for tab, render in zip(tabs, [render_architecture, render_pipeline, render_visual, render_audio,
                              render_streams, render_fusion, render_data]):
    with tab:
        _measured(render)
