# Face and mouth view integrity implementation plan

**Goal:** Build the complete, reproducible MTCNN versus YuNet comparison path,
landmark-aligned mouth views, and a motion-aware tracker challenger without
using validation or test identities.

**Status:** Approved by the existing Phase 2 roadmap and model-selection
protocol. This plan implements the tooling and fixture evidence. It must not
select a detector until a human-reviewed training-only sample meets the frozen
sample-size and agreement gates.

**Architecture:** Normalize both detectors into one five-landmark contract.
Keep box and landmark mouth crops as explicit candidates. Build reviewed-frame
annotations, source-disjoint detector threshold calibration, held-out benchmark
evaluation, and MLflow evidence as separate layers. Raw frames and annotations
stay outside Git and MLflow.

**Primary official sources:**

- OpenCV 5 `FaceDetectorYN` and the OpenCV Zoo YuNet 2026 model.
- `facenet-pytorch.MTCNN.detect(..., landmarks=True)`.

## Global constraints

- Work directly on `main`, as requested.
- Use Python 3.11 through 3.13 and the pinned dependency sets.
- Use TDD for every behavior change.
- Keep direct commands usable without MLflow.
- Never commit raw video, extracted frames, annotations, crops, model binaries,
  local paths, or benchmark outputs.
- Never upload raw frames, annotations, or crops to MLflow. Log metrics, hashes,
  configuration, and non-sensitive aggregate artifacts only.
- Use training identities only for detector, tracker, and crop selection.
- Split reviewed training sources into threshold-calibration and comparison
  subsets before detector inference. Never tune score thresholds on comparison
  sources.
- The reviewed comparison gate is at least 500 frames from at least 100 clips.
- Double-review at least 10 percent of frames and report disagreement.
- Annotate every visible face box. Mark at most one intended speaking target.
  Annotate five target landmarks when a suitable target exists.
- Count unmatched detections as false detections. Do not call another valid
  visible face a detector false positive.
- Use fixture results only to validate software. Label them
  `software_fixture_only` and never copy them into research findings.
- Pin the YuNet asset by repository commit, URL, byte size, and SHA-256:
  commit `47534e27c9851bb1128ccc0102f1145e27f23f98`, model
  `face_detection_yunet_2026may.onnx`, size `229738`, SHA-256
  `ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0`.
- The primary runtime comparison uses the local AMD Ryzen 5 5600X CPU. Record
  the RTX 5070 Ti as available hardware, but do not compare MTCNN CUDA with a
  CPU-only OpenCV wheel as if the backends were equal.
- Preserve existing MTCNN plus box-crop behavior until measured evidence freezes
  a replacement.

## Frozen detector comparison rules

1. Deterministically assign reviewed source identities to a 20 percent
   threshold-calibration subset and an 80 percent comparison subset.
2. Run each detector at a low collection threshold and store all candidates.
3. On calibration sources only, select the highest-recall score threshold whose
   false detections per frame do not exceed `0.10`. Break ties with fewer false
   detections, then the higher threshold.
4. Apply each frozen threshold to comparison sources without further tuning.
5. Report target recall at IoU `0.50`, false detections per frame, non-target
   candidate count, five-point normalized mean error, landmark coverage, stable
   track coverage, target-track errors, abstention, mouth jitter, and latency.
6. Bootstrap comparison metrics by source identity with 1,000 fixed resamples.
7. Reject a detector whose target recall is more than `0.01` below the best.
8. Among eligible detectors, reject a candidate whose landmark NME is more than
   `0.01` worse or whose target-track errors exceed the best by more than one per
   1,000 tracked frames.
9. Choose the fastest remaining CPU candidate. Use downstream aligned-mouth
   validation only as a tie-breaker after this benchmark.

The numeric margins above must be committed before human annotation starts.
Changing them later requires a new decision record and invalidates selection.

---

### Task 1: Five-landmark detector contract and pinned YuNet asset

**Files:**

- Create: `src/deepfake_detection/views/model_assets.py`
- Create: `configs/assets/yunet-2026may.json`
- Create: `tests/test_model_assets.py`
- Modify: `src/deepfake_detection/views/tracking.py`
- Modify: `src/deepfake_detection/views/face_detector.py`
- Modify: `tests/test_face_detector.py`
- Modify: `CHANGELOG.md`

**Interfaces:**

```python
@dataclass(frozen=True, slots=True)
class Point:
    x: float
    y: float

@dataclass(frozen=True, slots=True)
class Landmarks5:
    eye_left: Point
    eye_right: Point
    nose: Point
    mouth_left: Point
    mouth_right: Point

@dataclass(frozen=True, slots=True)
class Detection:
    box: Box
    confidence: float
    landmarks: Landmarks5 | None = None

class YuNetFaceDetector: ...
def fetch_yunet_model(destination: Path, *, force: bool = False) -> Path: ...
```

- [ ] Validate finite boxes, points, and confidence values.
- [ ] Canonicalize eye and mouth pairs by image x-coordinate so provider naming
  conventions cannot swap the contract.
- [ ] Call MTCNN with `landmarks=True` and preserve its five points.
- [ ] Parse YuNet rows as box `x,y,w,h`, five point pairs, and score.
- [ ] Set YuNet input size from every frame and accept no malformed result row.
- [ ] Keep dependency and fake-model injection lazy for unit tests.
- [ ] Fetch only the fixed HTTPS URL. Download atomically, enforce size and
  SHA-256, and reject a wrong existing file unless `force=True`.
- [ ] Test empty outputs, thresholds, malformed shapes, BGR-to-RGB conversion,
  provider point order, asset idempotency, and bad hashes.

Verify:

```powershell
uv run pytest tests\test_face_detector.py tests\test_model_assets.py -v
uv run ruff check src tests
uv run ruff format --check src tests
```

Commit: `Add landmark-aware face detectors`

---

### Task 2: Deterministic landmark-aligned lower-face views

**Files:**

- Create: `src/deepfake_detection/views/alignment.py`
- Create: `tests/test_face_alignment.py`
- Modify: `src/deepfake_detection/views/timeline.py`
- Modify: `src/deepfake_detection/views/contracts.py`
- Modify: `src/deepfake_detection/views/preprocessor.py`
- Modify: `tests/test_preprocessor.py`
- Modify: `tests/test_quality.py`
- Modify: `CHANGELOG.md`

**Interfaces:**

```python
def similarity_transform(
    source: Landmarks5,
    *,
    output_size: tuple[int, int],
) -> np.ndarray: ...

def aligned_lower_face(
    frame: np.ndarray,
    landmarks: Landmarks5,
    *,
    height: int,
    width: int,
) -> np.ndarray: ...
```

- [ ] Fit one deterministic least-squares similarity transform from all five
  points to a versioned normalized face template.
- [ ] Reject degenerate, non-finite, or out-of-frame landmarks.
- [ ] Warp a canonical face, take the frozen lower-face region, resize, convert
  BGR to RGB, and apply the existing normalization.
- [ ] Add `mouth_crop_mode` with `box` and `landmark` choices to `ViewConfig`.
- [ ] Keep `box` as the default until benchmark evidence freezes a change.
- [ ] Add `landmark_coverage` to quality data with a backward-compatible default.
- [ ] In landmark mode, never fall back to a box crop. Missing landmark evidence
  makes the sync view unavailable and adds a clear blocker.
- [ ] Include crop mode and template revision in the preprocessing hash.
- [ ] Test roll, translation, scale, missing points, border cases, nearest-frame
  fill, cache round trips, and stable output shapes.

Verify:

```powershell
uv run pytest tests\test_face_alignment.py tests\test_preprocessor.py tests\test_quality.py tests\test_clip_cache.py -v
```

Commit: `Add landmark-aligned mouth views`

---

### Task 3: Motion-aware face association challenger

**Files:**

- Modify: `src/deepfake_detection/views/tracking.py`
- Modify: `src/deepfake_detection/views/timeline.py`
- Modify: `src/deepfake_detection/views/preprocessor.py`
- Modify: `tests/test_tracking.py`
- Modify: `tests/test_preprocessor.py`
- Modify: `CHANGELOG.md`

**Interfaces:**

```python
def select_primary_track(
    frames: tuple[tuple[Detection, ...], ...],
    *,
    min_iou: float,
    association: Literal["greedy_iou", "constant_velocity"] = "greedy_iou",
    max_gap: int = 0,
    ...,
) -> TrackSelection: ...
```

- [ ] Preserve current greedy IoU results exactly by default.
- [ ] Add deterministic constant-velocity box prediction using the last two
  matched frames and elapsed frame indices.
- [ ] Use predicted-box IoU for association and a bounded missing-frame gap.
- [ ] Resolve equal scores with stable track and detection indices.
- [ ] Never merge simultaneous detections into one track.
- [ ] Add tracker choice and gap to `ViewConfig` and its hash.
- [ ] Test crossing tracks, fast motion, one-frame misses, equal scores,
  multi-person ambiguity, and empty input.

Verify:

```powershell
uv run pytest tests\test_tracking.py tests\test_preprocessor.py -v
```

Commit: `Add motion-aware face association`

---

### Task 4: Training-only detector review sample and annotation contract

**Files:**

- Create: `src/deepfake_detection/benchmarks/__init__.py`
- Create: `src/deepfake_detection/benchmarks/detector_sample.py`
- Create: `src/deepfake_detection/benchmarks/detector_annotations.py`
- Create: `tests/test_detector_sample.py`
- Create: `tests/test_detector_annotations.py`
- Modify: `CHANGELOG.md`

**Interfaces:**

```python
@dataclass(frozen=True, slots=True)
class ReviewFrame: ...

@dataclass(frozen=True, slots=True)
class FaceAnnotation:
    box: Box
    target: bool
    landmarks: Landmarks5 | None

@dataclass(frozen=True, slots=True)
class FrameAnnotation:
    frame_id: str
    reviewer_id: str
    faces: tuple[FaceAnnotation, ...]
    no_suitable_target: bool
    pose: str
    lighting: str
    multi_person: bool

def build_review_sample(..., seed: int = 17) -> tuple[ReviewFrame, ...]: ...
def validate_annotations(...) -> AnnotationAudit: ...
```

- [ ] Accept only a manifest explicitly identified as the training partition.
- [ ] Sample source identities first, then clips, then timestamps.
- [ ] Stratify by manipulation type, method, race, and gender where available.
- [ ] Hash every source frame and record no private absolute path in shared
  aggregate outputs.
- [ ] Allocate a deterministic 20/80 source split for threshold calibration and
  comparison.
- [ ] Allocate a deterministic double-review subset of at least 10 percent.
- [ ] Use JSONL because a frame may contain multiple visible faces.
- [ ] Require exactly one target when a suitable target exists, target
  landmarks, valid boxes, valid points, and reviewer identity.
- [ ] Report missing strata and annotation disagreement. Do not silently shrink
  the 500-frame and 100-clip gates.
- [ ] Test determinism, group separation, balance limits, duplicate frames,
  invalid targets, multi-face frames, no-target frames, and double review.

Verify:

```powershell
uv run pytest tests\test_detector_sample.py tests\test_detector_annotations.py -v
```

Commit: `Add detector review data contracts`

---

### Task 5: Detector benchmark metrics, threshold calibration, and selection

**Files:**

- Create: `src/deepfake_detection/benchmarks/detector_metrics.py`
- Create: `src/deepfake_detection/benchmarks/detector_runner.py`
- Create: `tests/test_detector_metrics.py`
- Create: `tests/test_detector_runner.py`
- Modify: `CHANGELOG.md`

**Interfaces:**

```python
@dataclass(frozen=True, slots=True)
class DetectorBenchmarkReport: ...

def calibrate_detector_threshold(...) -> float: ...
def evaluate_detector(...) -> DetectorBenchmarkReport: ...
def compare_detectors(...) -> DetectorDecision: ...
def run_detector_benchmark(...) -> DetectorBenchmarkReport: ...
```

- [ ] Store raw candidate boxes, landmarks, scores, latency, detector revision,
  model hash, frame hash, source hash, and split role in deterministic JSONL.
- [ ] Warm up each backend before timing. Record every timed frame, median, p95,
  total throughput, device, thread count, and runtime snapshot.
- [ ] Match detections to all annotated faces once with deterministic maximum-IoU
  assignment. Match the target separately for recall and landmark error.
- [ ] Compute calibration thresholds only from calibration sources.
- [ ] Compute comparison metrics only from comparison sources.
- [ ] Normalize five-point error by annotated inter-eye distance and report
  missing-landmark coverage separately.
- [ ] Measure target-track errors and stable coverage for both association modes.
- [ ] Measure aligned mouth jitter after compensating with the annotated face
  transform.
- [ ] Produce 1,000 fixed source bootstrap intervals.
- [ ] Apply the frozen recall, NME, tracking, and runtime selection rules without
  hidden tie-breaking.
- [ ] Reject insufficient samples, source overlap, missing double review,
  non-finite values, inconsistent frame hashes, and post-hoc rule changes.
- [ ] Mark fixture reports as `software_fixture_only`.

Verify:

```powershell
uv run pytest tests\test_detector_metrics.py tests\test_detector_runner.py -v
```

Commit: `Add detector benchmark evaluation`

---

### Task 6: CLI, MLflow evidence, CI smoke, and phase handoff

**Files:**

- Create: `configs/detectors/mtcnn-landmark.yaml`
- Create: `configs/detectors/yunet-landmark.yaml`
- Create: `tests/test_detector_cli.py`
- Modify: `src/deepfake_detection/cli.py`
- Modify: `src/deepfake_detection/experiments/training_log.py`
- Modify: `src/deepfake_detection/inference/loading.py`
- Modify: `.github/workflows/ci.yml`
- Modify: `tests/test_ci.py`
- Modify: `README.md`
- Modify: `ROADMAP.md`
- Modify: `docs/model-selection.md`
- Modify: `docs/reproducibility.md`
- Modify: `docs/reference/cli.md`
- Modify: `CHANGELOG.md`

**Command tree:**

```text
ddf detector fetch-yunet
ddf detector sample
ddf detector validate-annotations
ddf detector run
ddf detector compare
```

- [ ] Add detector, tracker, crop mode, model path, and expected model hash to
  cache and prediction construction through one shared factory.
- [ ] Keep existing defaults unchanged.
- [ ] Make every detector command usable through `ddf run` and local MLflow.
- [ ] Log only aggregate reports, configs, hashes, and prediction JSONL with
  private paths removed. Never log review images or raw annotation files.
- [ ] Add a deterministic detector-evaluator fixture smoke to CI. It validates
  the comparison path but cannot select a real detector.
- [ ] Keep the real YuNet load test opt-in when the pinned model is present.
- [ ] Update only operational commands, frozen comparison rules, provenance,
  and truthful roadmap state.
- [ ] Mark landmark, YuNet, alignment, tracker, and benchmark tooling complete.
  Leave reviewed sample, measured comparison, and frozen selection unchecked
  until real human-reviewed evidence exists.

Verify:

```powershell
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
uv lock --check
uv run ddf-docs
uv run ddf detector --help
git diff --check
git status --short
```

Commit: `Complete detector benchmark tooling`

## Phase 2A exit gate

- Both adapters return the same validated five-landmark contract.
- Box and aligned mouth views have different versioned preprocessing hashes.
- Greedy IoU remains the default and constant-velocity association is tested.
- Review sampling and annotation validation enforce training-only source gates.
- Threshold calibration and comparison sources are disjoint.
- Fixture evidence proves metric, selection, hashing, MLflow, and CLI behavior.
- Raw media, frames, annotations, crops, and model binaries remain untracked.
- No detector, tracker, or crop winner is claimed without the reviewed sample.

## Human evidence gate after implementation

The code can prepare and evaluate the comparison without further design input.
Completing Phase 2 still requires:

1. The local training manifest and dataset root.
2. Human review of at least 500 sampled frames from at least 100 clips.
3. A second review of at least 10 percent of frames.
4. Execution of MTCNN and YuNet on the frozen sample.
5. Acceptance of the measured decision before changing preprocessing defaults.

