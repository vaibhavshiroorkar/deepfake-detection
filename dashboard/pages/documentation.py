"""Documentation — the long-form reference for the whole system.

Everything that used to crowd the Overview page lives here: why lip-sync
forgeries defeat vision-only detectors, how every preprocessing step and every
model actually works, how fusion and evaluation are designed, and how the data
is split. Overview stays short and points here.

It is documentation, not compute: nothing here loads a model, decodes a clip or
touches data/, so the page opens instantly.

Sources of truth it mirrors: docs/PROJECT_OVERVIEW.md (§2 architecture, §3
fusion, §5 data, §6 preprocessing contract) and docs/preprocessing.md (per-step
reference). When those change, this page changes with them.

Plain Streamlit components throughout — no custom CSS, no ASCII diagrams. Detail
lives inside expanders so each section reads as an outline first and a reference
second.
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

st.title("Documentation")
st.caption("The in-depth reference: the problem, every processing step, every model, and the "
           "reasoning behind the decisions that are easy to get wrong. Read-only — this page "
           "loads no models and decodes nothing.")


# ====================== 1 · PROBLEM AND ARCHITECTURE ======================= #
def render_architecture():
    st.header("Why lip-sync forgeries defeat vision-only detectors")
    st.markdown("""
Most deepfake detectors are trained to notice manufacturing residue: the seam where a generated
face was blended onto a real head, colour statistics that drift between the face and the neck,
warping around the jaw, upsampling patterns left by a GAN's decoder. Against a full-face swap
this works well, because the whole face is synthetic and the residue is everywhere.

A wav2lip-style forgery gives them far less to work with. The generator is handed a genuine
video and a target audio track, and it repaints only the mouth region, frame by frame, so the
lips appear to speak the new words. Everything outside a small patch is the original recording.
The artifact budget is tiny, it sits in a region that moves and blurs naturally, and video
compression erases much of what is left.

The forgery is nonetheless obvious once you stop looking at the video in isolation. A real
recording contains two views of the same physical event: light reflected off a moving mouth, and
pressure waves produced by that same mouth. The two views are causally locked together.
Synthesis breaks the lock. The generated mouth is plausible on its own and the audio is
plausible on its own, but their *joint* behaviour — the timing, and the correspondence between
visible articulation and the sound produced — no longer belongs to one event.

That is the signal this project targets, and it is why there is deliberately no standalone
audio-only classifier anywhere in the design. An audio model can report that a voice sounds
synthetic; it cannot report that the voice disagrees with the face, and the disagreement is the
part that survives compression, resolution loss and a well-trained generator.
""")

    st.header("The signal chain")
    st.caption("Two parallel paths turn a clip into tensors, five streams turn those tensors into "
               "embeddings, one fusion head turns the embeddings into a decision.")
    st.markdown("""
| Path | Steps applied in order | Tensor produced |
|---|---|---|
| **Video** | 16 timestamps → MTCNN detect → 5-point align → 224² crop → ImageNet normalise | `faces [16, 3, 224, 224]` |
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
    st.caption("Every stream emits a 256-dimensional embedding and never a score. Keeping them at "
               "the feature level is what lets fusion learn interactions between them.")
    st.markdown("""
| Stream | Inputs | What it contributes | Stage |
|---|---|---|---|
| **Xception** | faces | Low-level blending and colour artifacts. The long-standing FaceForensics++ baseline, strongest of the classic CNNs on manipulation residue. | 3 |
| **EfficientNet-B0** | faces | A second artifact view from a different architecture family, and the lightest of the three — which is why it was built first and set the AUC 0.994 bar. | 2 |
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


# ========================== 2 · VISUAL PATH ================================ #
def render_visual():
    st.header("Visual path")
    st.caption("Five steps, applied per sampled frame, producing one [16, 3, 224, 224] tensor per "
               "clip plus a parallel mouth tensor.")

    with st.expander("**1 · Frame sampling** — choosing 16 moments", expanded=True):
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

    with st.expander("**2 · Face detection** — how MTCNN actually works"):
        st.markdown("""
MTCNN (Zhang et al., 2016) is a **cascade of three small convolutional networks** that get
progressively more expensive and progressively more selective. The design assumption is that the
overwhelming majority of image regions are obviously not faces, so they should be rejected by
something cheap, and only the survivors should be shown to something accurate.

The image is first resized repeatedly by a factor of about 0.709 — one half-octave — down to the
minimum detectable face size, producing an **image pyramid**. This lets a single detector with a
fixed 12×12 receptive field find faces at any scale: rather than searching for faces of many
sizes, the network searches for one size in many images.

| Stage | Input | Job | Emits |
|---|---|---|---|
| **P-Net** | each pyramid level | Fully convolutional, so it sweeps a whole level in one pass and proposes candidate windows. Deliberately shallow and permissive — tuned for recall, and expected to return many false positives. | face score, 4 box offsets |
| **R-Net** | 24×24 crops | A fully-connected refinement network. Rejects the bulk of P-Net's false positives and tightens the boxes that survive. | face score, 4 box offsets |
| **O-Net** | 48×48 crops | The deepest of the three, and the only one whose landmarks are used. Produces the final decision and the 5 facial landmarks alignment depends on. | face score, 4 box offsets, **5 landmarks** |

Between stages, two operations do the actual pruning. **Bounding-box regression** applies each
network's predicted offsets, so the box reaching the next stage is better centred than the one
that entered it. **Non-maximum suppression** then sorts surviving boxes by score and discards any
box whose intersection-over-union with a higher-scoring box exceeds a threshold, collapsing the
cluster of overlapping detections around each face into one.

The three networks are trained jointly with a multi-task loss: cross-entropy for the
face/not-face decision, squared error for the box offsets, and squared error for the landmark
coordinates, each weighted differently per stage — P-Net and R-Net weight detection heavily,
O-Net weights landmark accuracy heavily, because landmarks are its distinctive output. Training
also uses online hard example mining: within each mini-batch only the highest-loss samples
contribute gradients, so capacity is spent on cases the network is currently failing rather than
on background it already rejects.

The five landmarks are the left eye, right eye, nose tip, and the two mouth corners — ten
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
profile views and heavy occlusion, and five landmarks is the minimum alignment can work with —
AV-HuBERT's own preprocessing would prefer 68.
""")

    with st.expander("**3 · Five-point alignment** — the largest quality gain available"):
        st.markdown("""
The five detected landmarks are mapped onto a canonical ArcFace-style template with a
**similarity transform**: rotation, uniform scale and translation, and nothing else.
""")
        st.latex(r"\begin{bmatrix} x' \\ y' \end{bmatrix} = s\,R(\theta)\begin{bmatrix} x \\ y "
                 r"\end{bmatrix} + t \qquad \text{4 unknowns, 10 equations}")
        st.markdown("""
Five point correspondences give ten equations for four unknowns, so the system is
over-determined and solved in the least-squares sense; the residual is the extent to which this
face cannot be made to match the template by rotating and scaling alone. The resulting 2×3
matrix is applied with bilinear interpolation to produce the 224×224 crop.

**Why not a full affine transform.** Six degrees of freedom would add shear and non-uniform
scale, and would therefore fit the template more exactly — by deforming the face. Facial geometry
is evidence. A transform that can squash a wide face into a narrow template is destroying part of
what the detector should be looking at, so the extra two degrees of freedom are given up on
purpose.

**What alignment buys.** Without it, the temporal model receives a face that rolls, drifts and
rescales as the head moves, and must spend capacity undoing rigid motion before it can look at
anything else. With it, the eyes and mouth land on approximately the same pixels in every frame,
so what varies frame to frame is expression, articulation and artifacts. That is exactly the
signal the sequence model exists to read, and it is the single biggest quality gain available
here without adopting a heavier detector.
""")
        st.warning("""**Two traps, both already fixed.**

**Border padding.** An aligned canvas usually reaches past the edge of the source frame, and what
it reaches for has to come from somewhere. It is now filled **black**. It used to be *reflected*,
which pasted a mirrored, upside-down second face into essentially every FakeAVCeleb crop — these
clips are already tight face crops, so the canvas always overshoots, measured at 14–45% of it.
Padding must be inert, never synthesised, or the model learns the padding.

**Template inset.** `align_inset` shrinks the template to buy hairline and jaw context, and
defaults to **0**. Insetting asks the source frame for context further out than the face, and a
frame that is already a tight crop answers with more padding. It is a separate knob from the bbox
path's `margin`, which clamps to the frame and is therefore safe; feeding one value to both is
what made aligned crops a quarter empty. Raise it only for datasets whose frames are whole
scenes.""")

    with st.expander("**4 · Mouth ROI** — a parallel output, not a substitute"):
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

    with st.expander("**5 · ImageNet normalisation** — once, here, for everyone"):
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

On the Visual tab this step displays de-normalised frames — the transform inverted — purely so the
result is viewable. The tensor handed to the model is the normalised one, and the printed value
range underneath it is the honest check.
""")


# =========================== 3 · AUDIO PATH ================================ #
def render_audio():
    st.header("Audio path")
    st.caption("Short, but it contains the single most important correctness detail in the whole "
               "preprocessing stage.")

    with st.expander("**1 · Decode** — PyAV", expanded=True):
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
The downmix is a channel mean. Stereo separation in this dataset carries no forgery signal — the
fakes are generated as mono speech and mixed identically — so keeping two channels would double
the data for no information, and every downstream audio encoder expects mono anyway.

Resampling is a filtered operation, not a subsampling one. Dropping every other sample would fold
all content above the new Nyquist frequency back into the audible band as aliasing, so the signal
is low-pass filtered before decimation; librosa's resampler does this with a windowed sinc kernel.

16 kHz is not an arbitrary choice. It is the rate Whisper and Wav2Vec2 were pretrained at, and
their convolutional front-ends assume it. It also sits comfortably above speech content, which
carries very little energy above 8 kHz, so the higher native rate costs samples and memory
without adding usable signal.
""")

    with st.expander("**3 · Leading silence** — FakeAVCeleb's shortcut"):
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
One window of `window_sec` — 0.35 s by default — centred on each of the 16 frame timestamps,
clamped to the bounds of the clip and zero-padded to a fixed length so the result is rectangular.
At 16 kHz that is 5600 samples per window, giving `[16, 5600]`.

Windows are centred per frame rather than concatenated into one long track because the
cross-modal streams compare frame *i*'s mouth shape against the sound that occurred at that
moment. The length is a compromise: long enough to contain a complete phoneme and its transition,
short enough that a synchronisation error of order 100 ms shows up as a visible mismatch rather
than being averaged away.
""")

    st.subheader("Extras — off by default, and deliberately separate")
    st.markdown("""
None of these are part of the stored contract. With every one disabled, the dashboard reproduces
the batch pipeline exactly.

**Enhancement** (sharpen, denoise, CLAHE) asks whether cleaning an input helps or merely removes
the artifacts the detector was relying on. CLAHE in particular equalises contrast in local tiles,
which can amplify precisely the blending seams a visual stream is hunting for — or wash them out.

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
    st.header("Visual streams — one template, three backbones")
    st.caption("All three are the same config-driven module with a different backbone name, so "
               "adding a fourth is a configuration change rather than an engineering project.")
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
the temporal model and head are trained. It is dramatically cheaper — per-frame embeddings can be
computed once and cached, after which an epoch costs almost nothing — and it is the realistic
default on a 6 GB GPU. Fine-tuning lets the backbone specialise on forgery artifacts and
generally scores better, at a cost that scales with five backbones. The decision is deliberately
left open until Stage 6 rather than assumed.
""")

    with st.expander("**Xception** — depthwise separable convolutions"):
        st.markdown("""
Xception replaces the standard convolution with a **depthwise separable** pair: first a single
k×k kernel per input channel, applied independently and mixing nothing across channels; then a
1×1 convolution that mixes channels and mixes nothing spatially. The hypothesis in the name —
"extreme Inception" — is that spatial correlation and cross-channel correlation are separable
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

    with st.expander("**EfficientNet-B0** — compound scaling"):
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
F1 0.974 and EER 0.018, in-distribution — the bar the rebuild has to clear.
""")

    with st.expander("**DINOv2 (ViT-S/14)** — self-supervised, and the generalisation bet"):
        st.markdown("""
DINOv2 is a vision transformer trained with **no labels of any kind**. Two networks are involved:
a student, and a teacher whose weights are an exponential moving average of the student's. Both
see different augmented crops of the same image — the teacher gets large global crops, the student
gets global and small local ones — and the student is trained to match the teacher's output
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

    st.header("Cross-modal streams — measuring disagreement")
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
with a small, consistent lag — sound arrives at a fixed offset from the articulation that produced
it. In a re-synthesised clip the correspondence is smeared or drifts, because the mouth was
generated to match the audio only in aggregate. The stream does not decide anything about this; it
hands the resulting vector to fusion, which learns what to do with it.
""")

    with st.expander("**Lip-sync stream** — AV-HuBERT + Whisper (Stage 4)"):
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

    with st.expander("**Emotion stream** — HSEmotions + Wav2Vec2 (Stage 5)"):
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
    st.header("Fusion — feature-level, not score averaging")
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
wav2lip forgery, and it is a conjunction no weighted average of scores can express — averaging a
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

    st.header("Evaluation — what gets measured, and what would hide a failure")
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
quality independently of any threshold choice. EER — the point where the false accept and false
reject rates are equal — is the standard single number in forensic biometrics and is directly
comparable with published work. F1 keeps precision and recall visible, which is where an
imbalanced test set does its damage.

**Generalisation.** Tested twice, because in-distribution numbers on a single dataset predict
almost nothing. A manipulation method is held out of training entirely, and the final system is
evaluated on Deepfake-Eval-2024, an in-the-wild collection never used for training at any point.
""")


# ========================== 6 · DATA AND SPLITS ============================ #
def render_data():
    st.header("Dataset — FakeAVCeleb v1.2")
    st.markdown("""
Chosen over FaceForensics++ and Celeb-DF for one decisive reason: it contains genuinely
manipulated audio. Visual-only datasets cannot test cross-modal mismatch at all, so they cannot
validate the central claim of this project.

| Category | Video | Audio | Why it is in the set |
|---|---|---|---|
| **RVRA** | real | real | The only genuine class, and the scarce one: roughly 500 clips against 19,500 fakes. |
| **RVFA** | real | fake | Real footage with a synthesised voice. Invisible to every visual-only stream by construction. |
| **FVRA** | fake | real | Manipulated face against the original audio. |
| **FVFA** | fake | fake | Both tracks manipulated. Produced by wav2lip, this is the headline lip-sync case the system is built for. |

The manipulation *method* is a separate column from the category — `real`, `faceswap`, `fsgan`,
`wav2lip` — and both breakdowns are reported, because a system that handles face swaps well and
wav2lip badly has failed at the task even if its category-level numbers look acceptable.
""")

    st.header("Labels — authenticity is per stream, not per clip")
    st.warning("""A visual-only stream is judged on the **video track's** authenticity, not the
clip's. `FakeVideo-*` is fake and `RealVideo-*` is real, **including RealVideo-FakeAudio**, whose
forgery is audio-only and therefore correctly invisible to a model that never hears it. This looks
like a labelling bug and is not one. It is the reason the cross-modal streams exist, and a visual
stream that appeared to catch RVFA would be reading a dataset artifact.""")

    st.header("Splits — identity-disjoint, built once, never re-split")
    st.markdown("""
The same identities appear across all four categories. If a real clip of a person lands in train
while a forgery derived from that same person lands in test, the two share background, lighting,
framing, wardrobe and face. A model can then score spectacularly by recognising the *person*, and
the reported AUC measures memorisation rather than detection. This is the most common way deepfake
projects produce results that do not survive contact with new data.

Splits are therefore constructed once, on the `source` identity, by `build_splits.py`, checked by
`verify_splits.py` which asserts zero identity and zero file overlap, and never re-split randomly
downstream. The current split is 1400 / 300 / 300 clips over 500 source identities.

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
| `data/raw/FakeAVCeleb_v1.2/` | The dataset, gitignored — 21,544 clips, plus `meta_data.csv`. Clips are nested `<category>/<race>/<gender>/<identity>/*.mp4`. |
| `data/processed/<clip_id>/` | Per-clip cache: `frames.npy`, `audio.npy`, `timestamps.npy`, `version.txt`. |
| `data/*.csv` | `full_manifest.csv` and the `train` / `val` / `test` splits. |

The per-clip cache carries a `version.txt` stamped with `PIPELINE_VERSION`. Bumping that constant
invalidates every stale cache so it is transparently re-extracted on next access, which is what
makes a change like switching on face alignment safe: the pixels changed, so the cache must not be
trusted, and nothing has to be deleted by hand.

The dataset picker discovers what is available rather than being configured. A raw drop is any
directory containing a `meta_data.csv`, and its manifest is derived in memory, so a freshly
unzipped dataset is usable before `audit_dataset.py` has ever run. A manifest CSV attaches to
whichever dataset its `video_path` column points into. Nothing is hardcoded, and the rescan button
re-runs the whole discovery.
""")


tabs = st.tabs(["Problem & architecture", "Visual path", "Audio path", "Streams",
                "Fusion & evaluation", "Data & splits"])
with tabs[0]:
    render_architecture()
with tabs[1]:
    render_visual()
with tabs[2]:
    render_audio()
with tabs[3]:
    render_streams()
with tabs[4]:
    render_fusion()
with tabs[5]:
    render_data()
