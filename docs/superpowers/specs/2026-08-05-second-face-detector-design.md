# A second face detector: YuNet alongside MTCNN

MTCNN was the only detector the pinned environment shipped, so `faces.detect`
called `facenet-pytorch`'s exact signature and every doc said MTCNN was the only
option. This adds OpenCV's YuNet as a second, selectable detector, and puts a
timing readout next to the choice so the two can be compared on a real clip
rather than argued about.

Both stay. Neither is removed and the default does not move.

## Why

YuNet is roughly 9x faster per frame and needs no GPU at all. Measured on one
FakeAVCeleb clip, 16 frames: MTCNN 357 ms (22.3 ms/frame), YuNet 39 ms
(2.4 ms/frame). That matters most in `precache.py`, where workers are pinned to
CPU precisely so they do not contend with training for the GPU.

It is not a straight upgrade. Every trained checkpoint so far used MTCNN's crops,
so switching detectors invalidates that work until it is re-validated. Hence a
choice rather than a replacement.

## The detector interface

`preprocessing/ops/detectors.py` holds two adapters behind one call:

```
name: str
detect(frame_rgb) -> (box_xyxy, landmarks5, prob) | (None, None, None)
```

`faces.detect` is the only place the confidence threshold is applied, so one
`conf_thresh` means the same thing whichever detector is loaded. YuNet is created
with a permissive internal score floor (0.05) so its own threshold cannot act as
a second, invisible gate that the dashboard slider is unable to reach.

`crop_and_resize`, `mouth_roi` and `detect_crop` are untouched.

### Landmark ordering

The design started from the assumption that YuNet's five points would need
reordering: its documentation lists the right eye first, MTCNN's the left. That
assumption was wrong. YuNet's "right eye" is the *subject's* right, which is the
image-left point `facenet-pytorch` calls the left eye. Both emit the image-left
eye first.

This was checked against 8 real clips before writing the adapter, and reordering
would have introduced the mirror bug it was meant to prevent. Nothing downstream
compares landmarks across detectors, so the failure would have been silent: same
shapes, same dtypes, mirrored faces. `tests/preprocessing/test_detectors.py`
pins the ordering by measuring both detectors on the same frame and asserting
that the as-is correspondence beats the swapped one.

The parked five-point alignment in `docs/ideas.md` is the reason this matters
beyond today: it reads landmarks by index, where the mouth ROI only takes a
midpoint of two.

## Cache

`version.txt` becomes `<PIPELINE_VERSION>:<detector>`, so `4:mtcnn`. A hit needs
both to match, which stops one detector reading the other's crops and a training
run learning from a mix nobody chose.

One slot per clip, not one per detector: extracting with the other detector
overwrites. Keeping both means keeping two copies of `data/processed/`.

`PIPELINE_VERSION` stays at 4. The detector is not a version: it varies per run
rather than moving forward, so it is stamped beside the number rather than
bumping it. A bare `4` is read as `4:mtcnn`, because every cache written before
the second detector existed can only have come from MTCNN. That saves
re-extracting the whole train+val set for what would otherwise be a rename.

## Surfaces

- `extract_clip(..., detector="mtcnn")`, with the lazy singleton keyed on
  `(name, device)`. Returns `detect_ms` on a fresh extraction.
- `precache.py --detector {mtcnn,yunet}`, printing mean ms/clip at the end. Its
  "pin to CPU so workers do not contend over the GPU" logic is moot for YuNet,
  which has no GPU path, and the startup line says so.
- Preprocessing page, Visual tab: a **Detector** dropdown beside the existing
  crop toggle, and a two-row timing readout. Each swap overwrites only its own
  row, so after one swap both numbers are on screen, one marked current and one
  marked previous run.

Streams pages are untouched; they keep reading the shared clip settings.

### Timing and the cache

`media.cached_face_mouth` returns `detect_ms` as a fourth value, measured
*inside* the memoized body so a cache hit replays the time detection really took
rather than the ~0 ms the cache hit itself cost. It covers the detect+crop calls
only, not frame decoding, which is identical work whichever detector is loaded
and would otherwise flatter the slow one.

## Weights

`checkpoints/yunet/face_detection_yunet_2023mar.onnx`, 232 KB, from
[opencv/opencv_zoo](https://github.com/opencv/opencv_zoo), committed alongside
the trained checkpoints. Vendored rather than fetched on first use: detection
then works offline, every run uses identical weights (which matters when the
output feeds a cache), and precache workers do not race for a download.

## Not in scope

- No per-detector confidence threshold. Both confidences are roughly 0-to-1 and
  YuNet's own default is also 0.90, so one threshold is defensible. The docs say
  they are not strictly comparable rather than inventing a second default.
- No re-precaching or re-training. Existing MTCNN caches stay valid and the
  default is unchanged, so this lands without invalidating prior work.
