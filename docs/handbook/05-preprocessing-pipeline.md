# Preprocessing pipeline

## Learning goals

After this chapter, you should be able to trace one `ClipRecord` into cached
visual, audio, mouth, and synchronization views. You should also be able to
calculate every default tensor shape, explain the quality blockers, and tell
implemented behavior from planned detector research.

## Required background

You need basic Python, arrays, paths, and arithmetic with seconds. Read
[Audio-video foundations](03-audio-video-foundations.md) for timestamps and
[Data and leakage](04-data-and-leakage.md) for the `ClipRecord` contract.

## Pipeline overview

The cache command follows this implemented path:

```text
ClipRecord
  -> FFmpegMediaDecoder.probe, read_frames, and read_audio
  -> ViewConfig and shared timestamps
  -> MTCNNFaceDetector.detect
  -> select_primary_track
  -> visual, audio, mouth, and sync views
  -> QualityReport.full_fusion_blockers
  -> preprocessing_config_hash and cache_fingerprint
  -> CacheStore.save
```

`FFmpegMediaDecoder` names its metadata inspection method `probe()`. The class
does not have an `inspect()` method. `Preprocessor.prepare()` coordinates the
rest of the calls and returns a `PreparedClip`. `build_cache()` then records
quality counts and saves the prepared clip.

The current defaults produce these unbatched `float32` arrays:

| Prepared value | Shape | Meaning |
|---|---|---|
| `visual_view` | `[16, 3, 224, 224]` | Sixteen normalized face crops. |
| `audio_view` | `[64000]` | Four seconds of mono audio at 16 kHz. |
| `sync_video_view` | `[50, 3, 112, 112]` | Two seconds of mouth crops at 25 fps. |
| `sync_audio_view` | `[32000]` | The aligned two-second audio window. |
| `sync_audio_context` | `[42240]` or absent | A raw 2.64-second window for real offsets. |

The first dimension of an image view is time, not batch. A data loader adds a
batch dimension. A batch of eight visual clips therefore has shape
`[8, 16, 3, 224, 224]`.

## Shared timeline

`ViewConfig` fixes the view definitions. Its current defaults are:

| Symbol | Field | Default |
|---|---|---|
| `N_v` | `visual_frames` | 16 frames |
| `H_v, W_v` | `visual_height`, `visual_width` | 224, 224 pixels |
| `T_a` | `audio_seconds` | 4.0 seconds |
| `T_s` | `sync_seconds` | 2.0 seconds |
| `f_s` | `sync_fps` | 25 frames per second |
| `H_s, W_s` | `sync_height`, `sync_width` | 112, 112 pixels |
| `O` | `sync_max_offset_seconds` | 0.32 seconds |
| `r` | `sample_rate` | 16000 samples per second |
| `e` | `eval_overlap` | 0.5, a unitless fraction |
| `m` | `crop_margin` | 0.20, a unitless fraction |
| `q` | `detector_confidence` | 0.80, a unitless probability |
| `R` | `remove_leading_silence` | `True`, a Boolean |

Let `D_v` be video duration, `D_a` be audio duration, and `l` be the manifest's
leading silence, all in seconds. `R` is the Boolean switch for silence removal.
With leading-silence removal enabled, content start `s_c`, in seconds, is:

```text
s_c = min(l, D_v)
```

If that value reaches the video end, the code resets `s_c` to zero. Disabling
removal also sets it to zero. The visual timestamp for index `i` is:

```text
t_i = s_c + (i + 1/2) * (D_v - s_c) / N_v
for i in {0, ..., N_v - 1}
```

Midpoints keep all requested timestamps inside the interval. `t_i` is time in
seconds, and `i` is the zero-based visual-frame index. Sync timestamps use a
fixed rate. Let `s_s` be sync start in seconds, `N_s` be the sync-frame count,
`u_j` be a sync timestamp in seconds, and `j` be its zero-based frame index:

```text
N_s = round(T_s * f_s) = 50
u_j = s_s + (j + 1/2) / f_s
for j in {0, ..., N_s - 1}
```

The code clamps each `u_j` to the final safe video timestamp
`max(0, D_v - 0.5 / f_s)`. For audio, let `T` be any window duration in
seconds and `L` be its sample count. Then `L = round(T * r)`. This gives 64000
main samples, 32000 sync samples, and 42240 context samples.

The initial sync start fits the two-second video window when possible:

```text
s_s = min(s_c, max(0, D_v - T_s))
```

Full offset context requires audio and the interval below. `lower` and `upper`
are candidate sync starts in seconds:

```text
lower = s_c + O
upper = min(D_v - T_s, D_a - T_s - O)
lower <= upper
```

When those conditions hold, the code replaces `s_s` with `lower`. Otherwise it
keeps the initial sync start and leaves `sync_audio_context` absent.

### Worked example

Take a five-second clip with five seconds of audio and 0.5 seconds of declared
leading silence. Then `s_c = 0.5`. The visual step is
`(5 - 0.5) / 16 = 0.28125` seconds. The first visual timestamp is `0.640625`
and the last is `4.859375`.

The earliest safe sync start is `s_c + O = 0.82`. The latest is:

```text
min(D_v - T_s, D_a - T_s - O) = min(3.0, 2.68) = 2.68
```

Real context is available because `0.82 <= 2.68`. The code sets `s_s = 0.82`.
The 50 mouth timestamps run from `0.84` through `2.80`. The context starts at
`s_s - O = 0.5` and lasts `T_s + 2O = 2.64` seconds. Its center two seconds
align with the mouth view. Cropping this real context at offsets from -320 to
320 ms avoids teaching the sync branch that zero padding means misalignment.

## Media decoding

`FFmpegMediaDecoder.probe()` opens the container with PyAV. It requires a video
stream. It reads duration, average video rate, audio presence, and audio
duration. Stream duration is converted to seconds with its time base. Container
duration is the fallback for video. If audio duration is unavailable, the code
uses video duration.

`read_frames()` uses OpenCV. For each shared timestamp it requests a seek in
milliseconds and decodes one BGR frame. It ignores the Boolean result from
`VideoCapture.set()`. It raises an error if the file cannot open or the
following `read()` fails. A failed or inaccurate seek that still returns a
frame is not detected. The preprocessor also rejects the wrong number of
returned frames.

`read_audio()` invokes the FFmpeg executable without a shell. It seeks to
`start_sec`, decodes `duration_sec`, removes video, mixes to one channel,
resamples to `sample_rate`, and returns little-endian `float32`. It pads a short
decode with zeros and trims a long decode to the requested sample count. A
missing FFmpeg executable raises `RuntimeError`; `build_cache()` catches it and
reports that clip as failed. A nonzero FFmpeg process raises
`subprocess.CalledProcessError` because `check=True`. `build_cache()` does not
catch that type, so one failed process can abort the whole cache build.

Seeking compressed media depends on container timestamps and keyframes. The
current frame adapter checks the decoded count, but it does not compare the
actual decoded presentation timestamp with the requested timestamp.

## Face detection and tracking

`MTCNNFaceDetector` is the only implemented detector. It converts each OpenCV
BGR frame to RGB and calls `facenet_pytorch.MTCNN.detect()`. The adapter keeps
boxes with probability at least 0.80 and sorts them by confidence. Its
`Detection` contract contains only a box and confidence. MTCNN can expose
facial landmarks, but this adapter does not request or retain them.

Tracking uses box intersection over union. Let `A` and `B` be boxes whose
coordinates are measured in pixels. Their areas use square pixels. IoU is a
unitless ratio:

```text
IoU(A, B) = area(A intersect B) / area(A union B)
```

For each frame, `select_primary_track()` visits longer existing tracks first.
It greedily appends the unused detection with maximum IoU when that IoU is at
least 0.30. Remaining detections start new tracks. Tracks are ranked by length,
then mean detection confidence.

Let `F` be the sampled-frame count, `L_1` be the primary-track detection count,
and `L_2` be the second-track detection count. Coverage and dominance are
unitless ratios, and `stable` is a Boolean. The quality terms are:

```text
coverage = L_1 / F
dominance = L_1 / L_2, or infinity when there is no second track
stable = coverage >= 0.80 and dominance >= 1.25
```

When a stable track misses a sampled frame, crop generation uses the known
track detection from the nearest sampled frame. It does not run interpolation.
When no stable track exists, both visual outputs are absent. The code never
uses the full frame as a fake face crop.

## Visual and mouth crops

For the visual view, the code centers a square on the face box. Let box width
`w`, box height `h`, and requested side `a` be measured in pixels. Margin
`m = 0.20` is unitless. The requested side length is:

```text
a = max(w, h) * (1 + 2m) = 1.4 * max(w, h)
```

The coordinates are rounded and clamped to the frame. Clamping at an image
edge can make the actual crop rectangular. OpenCV resizes every accepted crop
to 224 by 224 pixels.

The current mouth estimate is the lower 48 percent of the face box. Its top is
`y_top + 0.52 * h`, where `y_top` and `h` are in pixels. The other face-box
sides stay unchanged. That region gets a separate unitless 10 percent square
expansion and is resized to 112 by 112 pixels. This is a box-relative estimate,
not landmark alignment.

Both crop types use the same image normalization. Let `c` identify an RGB
channel. `p_c` is that channel's eight-bit pixel value from 0 through 255.
`mean_c` and `std_c` are the channel constants below on the zero-to-one scale.
The dimensionless normalized value is `x_c`:

```text
x_c = ((p_c / 255) - mean_c) / std_c
mean = (0.485, 0.456, 0.406)
std  = (0.229, 0.224, 0.225)
```

OpenCV BGR becomes RGB first. The visual crop is 224 by 224 pixels, and the
mouth crop is 112 by 112 pixels. The array then changes from height, width,
channels to channels, height, width. Stacking crops produces
`[time, channels, height, width]`.

## Audio views

Let `s_a` be main-audio start in seconds. The main audio start is:

```text
s_a = min(s_c, max(0, D_a - T_a))
```

This backs up the start when fewer than four seconds remain. The sync audio
uses `s_s`. The preprocessor estimates a valid sample count from the probed
audio duration and requested start. It does not receive the decoder's actual
unpadded sample count. Let `i` be a sample index and `x_i` be its returned
unitless `float32` amplitude in the estimated valid region. Let `mu` be that
region's mean, `sigma` be its population standard deviation, and `y_i` be its
normalized amplitude:

```text
y_i = (x_i - mu) / sigma, when sigma > 1e-7
y_i = 0, otherwise
```

Samples outside the estimated valid region stay zero. A constant or empty
estimated valid region also becomes zeros. However, `read_audio()` pads before
returning. If a successful decode is unexpectedly shorter than the probed
duration implies, its decoder-added zeros can fall inside the estimated valid
region and affect `mu` and `sigma`. The cache stores no actual-length mask to
detect that case. The main view is `[64000]`; the aligned sync view is
`[32000]`.

The 2.64-second `sync_audio_context` is different. The preprocessor only pads
or trims it to `[42240]`; it does not normalize it. The implemented
[`OFFSET_MILLISECONDS` and `crop_audio_context()`](../../src/deepfake_detection/branches/sync_objective.py)
define seven real-context offsets:

```text
{-320, -160, -80, 0, 80, 160, 320} milliseconds
```

[`CachedSyncDataset`](../../src/deepfake_detection/data/datasets.py) selects and
normalizes each cropped window. For the eighth sync class, it selects cached
sync audio from a different authentic source identity. Context is absent when
the video and audio do not contain the full margin around the sync window.

`audio_clipped` is true when the absolute value of any raw main-audio sample is
at least 0.999. It is reported as quality information, but it is not currently
a full-fusion blocker.

## Quality gates and abstention

`QualityReport.full_fusion_blockers()` returns named reasons that full fusion
is unavailable:

| Blocker | Current condition |
|---|---|
| `missing_audio` | No audio stream exists. |
| `unstable_face_track` | Either the face or mouth track is unstable. |
| `low_face_coverage` | Minimum face and mouth coverage is below 0.80. |
| `av_duration_mismatch` | Absolute audio-video duration difference exceeds 0.25 seconds. |
| `insufficient_sync_duration` | Video or present audio is shorter than 2.0 seconds. |

The report uses the lower coverage of the visual and sync tracks. Cache build
still saves a prepared clip with blockers and counts each reason in its audit.
Downstream full fusion must abstain rather than treat a missing view as real or
fake evidence. A branch can still be usable when its required view exists.

An empty blocker tuple means the implemented checks found all required full
fusion evidence. It does not prove correct identity, alignment, labels, or
deepfake content.

## Caching and hashes

The project uses two SHA-256 hashes for different questions.

`preprocessing_config_hash()` hashes canonical JSON containing all
`ViewConfig` fields and `code_version`. It identifies one pipeline definition
across clips. The complete field set is the defaults table under Shared
timeline, including `eval_overlap` and `remove_leading_silence`. Clip content
and dataset are excluded.

`cache_fingerprint()` hashes canonical JSON containing:

- the preprocessing configuration hash;
- the dataset name;
- declared leading silence;
- the SHA-256 digest of the complete media file.

Changing media bytes, dataset, silence metadata, configuration, or code version
therefore changes the clip fingerprint. A hash shows input equality under this
definition. It does not show that the views are correct.

`CacheStore` uses this namespace:

```text
cache root / safe dataset / safe clip ID / cache fingerprint.npz
```

Each readable path component also gets the first ten hexadecimal characters of
its own SHA-256 digest. This separates datasets and reduces collisions after
unsafe characters are replaced. The compressed NumPy archive stores present
view arrays plus JSON metadata for the clip ID, both preprocessing hashes, and
quality report. Absent arrays are omitted.

Save writes a temporary archive in the target directory and then replaces the
final path. This prevents readers from seeing a partly written archive. Cache
build also rejects media paths that escape the resolved dataset root. It stops
the build if successful clips contain more than one preprocessing config hash.

## Current limitations

These are current weaknesses, not measured findings:

| Weakness | Current evidence or planned check |
|---|---|
| MTCNN is the only implemented detector. | `test_face_detector.py` covers its adapter. The MTCNN versus YuNet comparison in `model-selection.md` is planned and has no result yet. |
| The adapter discards available facial landmarks. | The adapter test covers boxes and confidence only. The planned detector review will annotate five landmarks and compare landmark error. |
| The mouth crop uses the lower 48 percent of the face box. | `test_preprocessor.py` protects the output view contract. The planned landmark-aligned comparison will measure mouth jitter and downstream validation. |
| Greedy IoU tracking can switch identities after crossing or long occlusion. | `test_tracking.py` covers longest-track selection and equal-length ambiguity. A motion-aware tracker and identity-switch measurement are planned. |
| Multi-person clips may abstain when no track dominates by 1.25. | The equal-length two-face test exercises this conservative unstable result. The detector review plans to measure stable coverage and abstention. |
| YuNet and landmark-aligned crops are not implemented. | They remain Phase 2 work in the roadmap. No score, selected detector, or landmark result exists. |

The planned detector experiment uses only training identities. It will compare
target-face recall, false detections, landmark error, identity switches, track
coverage, abstention, mouth jitter, runtime, and memory under fixed rules. See
[Model selection](../model-selection.md). Published detector claims are
background only and cannot select the project detector.

### Design trade-offs

- Shared timestamps keep views comparable, but sparse visual sampling can miss
  brief artifacts.
- Leading-silence removal reduces a known shortcut, but depends on correct
  manifest metadata. The CLI keeps a no-removal ablation.
- Uniform face sampling covers the clip, but does not follow scene cuts or
  speech activity.
- A stable-track requirement avoids silent full-frame fallback, but lowers
  coverage for hard poses, occlusion, and multi-person scenes.
- Zero padding gives fixed shapes, but masks are not stored with the arrays.
- Content and config hashes catch stale inputs, but hashing the full media file
  adds I/O cost.

## Project code path

1. [`ClipRecord`](../../src/deepfake_detection/data/manifest.py) supplies clip
   ID, dataset, media path, cue labels, and leading-silence metadata.
2. [`_cache_build()`](../../src/deepfake_detection/cli.py) creates the default
   `ViewConfig`, decoder, MTCNN detector, preprocessor, and cache store.
3. [`FFmpegMediaDecoder`](../../src/deepfake_detection/views/media.py) probes
   streams and decodes requested frames and audio.
4. [`ViewConfig`, `uniform_timestamps()`, and `make_sync_window()`](../../src/deepfake_detection/views/timeline.py)
   define shared time and shape contracts.
5. [`MTCNNFaceDetector`](../../src/deepfake_detection/views/face_detector.py)
   converts BGR to RGB and returns filtered boxes.
6. [`select_primary_track()`](../../src/deepfake_detection/views/tracking.py)
   performs greedy IoU association and stability checks.
7. [`Preprocessor.prepare()`](../../src/deepfake_detection/views/preprocessor.py)
   creates all views, quality fields, and hashes.
8. [`QualityReport`](../../src/deepfake_detection/views/contracts.py) names
   full-fusion blockers.
9. [`preprocessing_config_hash()` and `cache_fingerprint()`](../../src/deepfake_detection/views/cache.py)
   fingerprint the pipeline and clip.
10. [`build_cache()`](../../src/deepfake_detection/data/cache_build.py) checks
    paths, counts failures and blockers, and calls `CacheStore.save()`.
11. [`CacheStore`](../../src/deepfake_detection/views/cache_store.py) writes and
    loads compressed view archives in dataset and clip namespaces.
12. [`OFFSET_MILLISECONDS` and `crop_audio_context()`](../../src/deepfake_detection/branches/sync_objective.py)
    define the real sync offsets and crop them from cached context.
13. [`CachedSyncDataset`](../../src/deepfake_detection/data/datasets.py) makes
    real-context offset and cross-identity mismatch examples from the cache.

## Failure cases

- A missing video stream or unavailable duration stops the clip.
- A nonpositive video duration stops preprocessing.
- An unreadable frame or wrong decoded frame count stops preprocessing. A
  failed or inaccurate seek that still returns a frame passes undetected.
- A missing FFmpeg executable becomes a reported clip failure. A nonzero
  FFmpeg process can abort the full cache build with `CalledProcessError`.
- A face box that becomes empty after frame clamping stops crop creation.
- No stable face track produces absent visual and mouth views, not full-frame
  substitutes.
- Missing audio produces absent audio views and a quality blocker.
- Short audio is zero padded; short video or audio blocks full sync fusion.
- Wrong leading-silence metadata moves all view starts and changes the cache
  fingerprint.
- A media path outside the dataset root or a missing file becomes a reported
  cache-build failure.
- A cache with a different preprocessing config hash is rejected by datasets
  when the caller supplies the expected hash.
- Greedy tracking can keep the wrong person while still meeting numeric
  coverage and dominance thresholds.

### Supporting tests

[`test_preprocessor.py`](../../tests/test_preprocessor.py) covers exact shapes,
normalization, missing-face behavior, short clips, silence removal, and real
sync context. [`test_tracking.py`](../../tests/test_tracking.py) covers primary
track selection and multi-face ambiguity.
[`test_face_detector.py`](../../tests/test_face_detector.py) covers confidence
filtering and box conversion. Its fixture ignores the input image, so it does
not prove the implementation's BGR-to-RGB conversion.
[`test_media_decoder.py`](../../tests/test_media_decoder.py) exercises real
video and audio decoding when FFmpeg is available.
[`test_views.py`](../../tests/test_views.py) covers timestamps, sync lengths,
and hashes. [`test_quality.py`](../../tests/test_quality.py) covers blockers.
[`test_clip_cache.py`](../../tests/test_clip_cache.py) covers archive round trips
and dataset namespaces. [`test_cache_build.py`](../../tests/test_cache_build.py)
covers missing media, blocker counts, and the shared preprocessing hash.
[`test_sync_objective.py`](../../tests/test_sync_objective.py) fixes the seven
offset constants and checks real-context cropping without padded edges.
[`test_sync_dataset.py`](../../tests/test_sync_dataset.py) covers authentic
offset variants and the cross-source mismatch class.

Run the focused pipeline set:

```powershell
uv run pytest tests\test_preprocessor.py tests\test_tracking.py tests\test_face_detector.py tests\test_media_decoder.py tests\test_views.py tests\test_quality.py tests\test_clip_cache.py tests\test_cache_build.py tests\test_sync_objective.py tests\test_sync_dataset.py -v
```

## Exercises

1. For a ten-second clip with two seconds of leading silence, calculate all 16
   visual timestamps.
2. Change `sync_seconds` to 1.5 in a local `ViewConfig`. Calculate the mouth and
   sync-audio shapes without running the code. Then verify them in a test.
3. Draw two equal-length tracks across three frames. Explain why the result is
   unstable even when both tracks have full coverage.
4. Use a constant waveform shorter than four seconds. Predict its normalized
   content and padded values.
5. List the exact inputs that change the config hash and clip fingerprint.
6. Design a training-only reviewed sample for MTCNN and YuNet. State which
   measurements would reject a faster but less reliable detector.

## Viva questions

1. Why do visual and sync sampling share one `ViewConfig`?
   Expected answer: it fixes time, shape, detector, and crop choices under one
   hash so cached views can be compared and reproduced.
2. Why are visual timestamps placed at interval midpoints?
   Expected answer: midpoints spread samples uniformly while keeping the first
   and last requests inside the valid content interval.
3. Why does the pipeline keep a 2.64-second sync context?
   Expected answer: it supplies real audio for a centered two-second window at
   every planned offset from -320 to 320 ms without padding cues.
4. What makes a track stable?
   Expected answer: primary coverage is at least 0.80 and its length is at
   least 1.25 times the second track's length.
5. What happens when no stable face exists?
   Expected answer: visual and mouth views are absent. Quality blocks full
   fusion. The pipeline does not substitute the full frame.
6. How is the current mouth crop found?
   Expected answer: it takes the lower 48 percent of the face box, expands it
   with a 10 percent square margin, and resizes it to 112 by 112.
7. Why normalize valid audio before leaving padding at zero?
   Expected answer: including padding in the statistics would make content
   values depend on clip length and could expose a padding shortcut.
8. Does `audio_clipped` block fusion?
   Expected answer: no. It is recorded, but it is not in
   `full_fusion_blockers()`.
9. What is the difference between the two hashes?
   Expected answer: the config hash identifies code version and all view
   settings. The clip fingerprint also includes dataset, leading silence, and
   complete media content.
10. Has YuNet or landmark alignment improved this project?
    Expected answer: no result exists. Both are planned controlled comparisons,
    and MTCNN with box-relative crops is the only implemented path.

## Sources

- [FFmpeg command documentation](https://ffmpeg.org/ffmpeg.html)
- [PyAV stream timing documentation](https://pyav.org/docs/stable/api/stream.html)
- [OpenCV color conversion reference](https://docs.opencv.org/4.12.0/d8/d01/group__imgproc__color__conversions.html)
- [OpenCV geometric image transformations](https://docs.opencv.org/4.13.0/da/d54/group__imgproc__transform.html)
- [facenet-pytorch MTCNN implementation](https://github.com/timesler/facenet-pytorch)
- [MTCNN paper](https://arxiv.org/abs/1604.02878)
- [OpenCV YuNet tutorial](https://docs.opencv.org/4.6.0/d0/dd4/tutorial_dnn_face.html)
