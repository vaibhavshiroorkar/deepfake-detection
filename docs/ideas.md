# Parked ideas

Things deliberately not in the pipeline, kept here so the reasoning is not lost and
re-adding them does not start from scratch.

Each entry records what it was, why it is parked, and what a re-implementation has
to know. An idea leaves this file when it lands in the pipeline or is rejected
outright.

---

## 5-point face alignment

**Status:** removed 2026-07-28. Was the default from 2026-07-24. Implementation is
in git history at `996cdd6` (`preprocessing/ops/faces.py`, `align_face` and
`face_template`), with its tests in `tests/preprocessing/test_ops_faces.py`.

**What it did.** A partial-affine transform (rotation, uniform scale, translation,
no shear) mapped MTCNN's five landmarks onto a canonical ArcFace-style template,
then warped the frame onto a 224×224 canvas. Five point correspondences give ten
equations for four unknowns, solved least-squares, so the residual measured how far
the face was from the template under rigid motion alone.

The template lived in `ops/constants.py` as `ARCFACE_TEMPLATE_112`, the standard
insightface 112×112 five-point layout scaled to the crop size:

```python
ARCFACE_TEMPLATE_112 = np.array([
    [38.2946, 51.6963],   # left eye
    [73.5318, 51.5014],   # right eye
    [56.0252, 71.7366],   # nose tip
    [41.5493, 92.3655],   # left mouth corner
    [70.7299, 92.2041],   # right mouth corner
], dtype=np.float32)
```

That landmark order is exactly what `facenet-pytorch`'s MTCNN returns, so no
reordering was needed.

**Why it was worth having.** Without alignment the temporal model receives a face
that rolls, drifts and rescales as the head moves, and has to spend capacity
undoing rigid motion before it can look at anything else. With it, eyes and mouth
land on approximately the same pixels in every frame, so what varies frame to frame
is expression, articulation and artifacts, which is what the sequence model exists
to read. It was the largest quality gain available without adopting a heavier
detector.

**Why a similarity transform and not a full affine.** Six degrees of freedom would
add shear and non-uniform scale and would fit the template more exactly, by
deforming the face. Facial geometry is evidence; a transform that can squash a wide
face into a narrow template destroys part of what the detector should be reading.

### The two traps, both of which cost real time

**1. Border padding must be black, never synthesised.** An aligned canvas usually
reaches past the edge of the source frame. The original implementation reflected
those pixels, which pasted a mirrored, upside-down second face into essentially
every FakeAVCeleb crop, because these clips are already tight face crops and the
canvas always overshoots. Measured at 14-45% of the canvas. `cv2.warpAffine` with
`borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)` is correct, and matches
ArcFace's own `norm_crop`. The model can learn "no data" from constant padding; it
cannot learn anything good from a phantom face.

**2. `align_inset` is not the bbox `margin`, and must default to 0.** The inset
shrank the template toward the centre to buy hairline and jaw context. It asks the
source frame for context further out than the face, and a frame that is already a
tight crop answers with more padding: `inset=0.2` left 25% of the crop empty
against 14% at `inset=0`. The bbox `margin` is safe because it clamps to the frame;
the inset cannot clamp, it pads. Feeding one value to both knobs is what made
aligned crops a quarter empty. An inset is only appropriate for datasets whose
frames are whole scenes rather than pre-cropped faces.

### If it comes back

- Bump `PIPELINE_VERSION` (currently 4). Alignment changes cached pixels, so every
  `data/processed/` entry has to re-extract, and the visual stream needs
  re-measuring against the AUC 0.994 bar before the numbers are comparable.
- Keep the bbox crop as the fallback for the no-landmark and
  transform-failed cases. `estimateAffinePartial2D` can return `None`.
- Restore it as an option compared against the bbox crop on the *same* clips, not
  as a silent default. The point of parking it was that its benefit had never been
  measured on this dataset, only argued.

---

## Enhancement and robustness ops not yet built

The extras in `ops/extras_visual.py` and `ops/extras_audio.py` are all spatial or
spectral. Nothing perturbs **time**, which is the axis the lip-sync and emotion
streams are built to measure. In rough priority order:

**1. Audio/video offset, ±300 ms.** The positive control for the whole thesis: shift
audio against video by a known amount and confirm the lip-sync embedding moves. If
a deliberate 200 ms desync does not move it, the stream is broken and no amount of
training will reveal that. Cheap to build, because it only offsets the audio
timestamps relative to the frame timestamps in `extract_windows`. Two real desyncs
have already appeared by accident in this project (a trim that shifted windows by
64 ms, and a tail window off-centre by 62 ms); this control would have caught both
immediately.

**2. H.264/H.265 re-encode at a chosen CRF.** `jpeg_recompress` is intra-only and
reproduces none of the artifacts that matter. Real platforms use inter-frame
compression, where motion compensation allocates bits by motion, and the mouth is
the highest-motion region in the frame, which is exactly where wav2lip's evidence
lives. The machinery already exists: `dashboard/lib/media.transcode_to_h264`.

**3. Frame drop and fps reduction.** Dropped and duplicated frames are what
streaming actually does, and they perturb synchronisation directly. Nothing
currently touches temporal robustness.

Lower priority: audio codec round-trip (Opus or AMR-NB at low bitrate, closer to
what messaging apps do than a bandpass filter), pink rather than white noise (real
ambience is spectrally shaped), and reverberation, though that needs an impulse
response.

**The blocker for all of them.** The extras are imported by exactly one file,
`dashboard/pages/preprocess.py`. Nothing in `preprocessing/`, `training/` or
`evaluation/` applies them, so today they are an inspection tool: you can eyeball a
degraded frame but you cannot produce "AUC vs JPEG quality" or "EER vs SNR". Adding
more ops without an evaluation-time transform hook just adds sliders.

---

## Window centring at the tail

`extract_windows` clamps a window that would run off either end, so it slides
inward instead of staying centred on its frame. Frame timestamps come from the video
duration (`frame_count / fps`), which on FakeAVCeleb runs longer than the audio
track (10.120 s against 10.058 s on the clip measured), so the final window ends up
**62 ms** off-centre while the other fifteen are exact.

62 ms matters here: the 0.35 s window was chosen so that a synchronisation error of
order 100 ms is detectable, so a systematic 62 ms offset on frame 16 sits inside the
range the project cares about.

Not fixed because `preprocessing/extract_clip.py` has the identical behaviour, so
correcting it changes every cached tensor. The candidate fix is to derive the
duration from `min(video_duration, audio_duration)` in the shared op, which keeps
frame *i* and window *i* paired while guaranteeing every window fits. The Audio tab
reports the current error as *Worst off-centre*.

---

## assets/flow.png corrections

The architecture diagram has two typos: "AuHubert" should be AV-HuBERT, and
"Explanability" should be Explainability. It also still names the third backbone
"DinoV2"; the project moved to DINOv3 (2026-08-02), and the version in
`DeepFake_Detection_System_Presentation.pptx` is already ahead of this file on all
three counts.

More substantively, its "Stream Layer" box merges the video and audio paths *before*
the Visual Stream, which implies the visual streams see audio. They never do, and
the documentation is emphatic about it. The diagram should branch, not merge.

Parked because it is a binary asset. Rebuilding it as a Mermaid diagram would make
it reviewable in diffs and renderable natively by Streamlit.
