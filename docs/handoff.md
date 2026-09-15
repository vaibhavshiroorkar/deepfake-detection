# Project handoff

## Current state

Two tracks, and they answer different questions.

**Track one, the research pipeline**, is complete and measured. Design A trains
three cue-specific branches on FakeAVCeleb with source-grouped cross-fitting and
a calibrated late-fusion head. Design B replaces them with configurable streams
and a feature-level head. Both score through one evaluation path, every headline
number carries an identity-clustered interval, and every number resolves to a
row in [result traceability](research/result-traceability.md). Fifteen findings
are recorded in [findings](research/findings.md), including the five claims this
project made and later withdrew.

**Track two, open-world generated media**, is newer and is where the work is
pointed now. The system detects manipulation of a real recording; media
generated whole is a different problem, and the design for it is in
[architecture](research/architecture.md). DF26 is the benchmark: 2,691 clips,
271 real and 2,420 from seven modern generators including Veo 3.1, Kling 3.0,
Grok Imagine and Wan 2.6.

### Where the four stated targets stand

| Target | Status | Evidence |
| --- | --- | --- |
| 0.991 visual detection, in-domain | **met** | Design A fusion 0.9990 on the FakeAVCeleb test partition, visual branch 0.9988 |
| 0.844+ from audiovisual cues on genuine-frame fakes | **met on the number, failed on the mechanism** | Lip-sync stream 0.9969 on LAV-DF, which is exactly that case. Its `diagonal_mass` sat at chance in every epoch of three runs, so the audiovisual cue is not what produced the number |
| Fusion beats the strongest single stream | **met, and four times smaller than the cited 0.099** | Paired source bootstrap: +0.0247 [0.0220, 0.0280] over the visual baseline in-domain, separated from zero. Cross-corpus -0.0083 [-0.0185, 0.0024], not separated |
| 0.64+ on unseen generators without fine-tuning | **met** | DF26 leave-one-generator-out macro **0.6786**, container-controlled, range 0.5867 to 0.7622 |

### The protocol, which is the actual contribution

Leave-one-generator-out is the only headline. There is no in-domain column,
because a strong in-domain number has four times in this project turned out to
be a shortcut rather than skill: 0.9990 against 0.4920 cross-corpus, a 0.9969
matched-pair stream reading 0.5059 once capture conditions changed, a 0.9997 on
Veo 3 that was pure content confound, and a 1.0000 on every arm of a smoke test
that was a split leak.

Four shortcuts were caught before they produced a reported result, each by one
cheap check run before training rather than after:

- Veo 3 lab clips scored against talking-head reals, which would have given a
  perfect-looking content classifier.
- Real clips on both sides of a generator split, and in FaceForensics++ a fake
  can be matched to its own source video, so the split is on source as well as
  generator.
- Generation modes of one model treated as separate generators, which leaks the
  model through its sibling.
- Audio presence in DF26 predicting the label almost perfectly, which would have
  given a near-perfect audio branch measuring a container property.

### What is measured and settled

| Result | Number |
| --- | --- |
| DF26 leave-one-generator-out, container-controlled | macro 0.6786, spread 0.1755 |
| Frame-rate confound found by that control | Wan 2.6 fell 0.094, HunyuanVideo 0.045, others unchanged |
| Design A in-domain fusion | 0.9990 |
| Design A cross-corpus fusion on DFDC | 0.7500, paired difference against visual not separated from zero |
| Design B streams cross-corpus | four of five have intervals containing 0.5; only emotion separates, 0.5951 [0.5558, 0.6351] |
| Capture conditions versus manipulation type | an unseen manipulation family costs 0.11 and still works; an unseen corpus reads 0.49 |
| Full-frame probe on face manipulation | at chance, macro 0.5387, every interval containing 0.5 |
| Calibration cross-corpus | expected calibration error 0.0022 in-domain against 0.6366 on DFDC |

### What is running or blocked

Nothing is running. Blocked on other people: DeepSpeak v2 awaits author review,
AV-Deepfake1M needs a signed EULA, Deepfake-Eval-2024 was refused.

### The audiovisual gap, which is the open opportunity

DF26 cannot support audiovisual work: audio presence predicts its labels and
only 27 of 271 real clips carry audio at all. Its authors name audio, speech
quality and lip-sync as future work, so the gap is real and unclaimed. The
corpora that could fill it are in [corpora](research/datasets.md), and LAV-DF is
already on disk and hash-unified with the main cache.

## Dataset inventory

Counted from the working tree by `ddf handoff update`, not recorded by hand. An
earlier version of this file said `data/` was empty while both primary datasets
were on disk. That is the failure this generated block exists to prevent, so do
not edit the table by hand. Run the command again after any dataset change.

<!-- BEGIN GENERATED DATASETS -->
| Dataset directory | Role | State | Videos | Size |
|---|---|---|---:|---:|
| `data/FakeAVCeleb_v1.2` | Primary development set | present | 21,544 | 6.2 GB |
| `data/LAV-DF` | Cross-modal stream training, localized forgeries | present | 136,304 | 23.6 GB |
| `data/Celeb-DF-v2` | Cross-dataset generalization, no audio | present | 6,529 | 9.5 GB |
| `data/DFDC` | Cross-corpus test, the only one with audio not from VoxCeleb2 | present | 3,032 | 20.9 GB |
| `data/FaceForensics++` | Declared visual experiments | present | 7,000 | 16.7 GB |
| `data/MNW` | Locked external benchmark, evaluation only | present | 134 | 424.0 MB |
<!-- END GENERATED DATASETS -->

Raw data, checkpoints, run output, MLflow storage, and model artifacts remain
ignored by Git.

## Recorded evaluation reports

Every evaluation report found under `runs/`, regenerated with the inventory
above. A row reading `undefined` for ROC-AUC is a single-class evaluation set,
where no ranking metric exists; its detection rate is shown instead.

<!-- BEGIN GENERATED EVIDENCE -->
| Report | Dataset | Scope | Rows | ROC-AUC | Balanced accuracy |
|---|---|---|---:|---:|---:|
| `runs/comparison-20260904/visual-efficientnet-b0-ep8-seed17-validation-metrics.json` | FakeAVCeleb | development_comparison | 400 | 0.9992 | 0.9975 |
| `runs/comparison-20260904/visual-efficientnet-b0-initial-seed29-validation-metrics.json` | FakeAVCeleb | development_comparison | 400 | 1.0000 | 0.9975 |
| `runs/comparison-20260904/visual-efficientnet-b0-initial-seed43-validation-metrics.json` | FakeAVCeleb | development_comparison | 400 | 0.9998 | 0.9925 |
| `runs/comparison-20260904/visual-efficientnet-b0-lr3e-4-seed17-validation-metrics.json` | FakeAVCeleb | development_comparison | 400 | 1.0000 | 0.9950 |
| `runs/full-20260904/evaluation/baseline-visual-mnw-metrics.json` | MNW | external_mnw | 85 | 0.0357 | 0.0952 |
| `runs/full-20260904/evaluation/pilot-audio-validation-metrics.json` | FakeAVCeleb | development_validation | 598 | 1.0000 | 0.9984 |
| `runs/full-20260904/evaluation/pilot-visual-celebdf-metrics.json` | Celeb-DF-v2 | generalization_celebdf | 462 | 0.6476 | 0.5741 |
| `runs/full-20260904/evaluation/pilot-visual-mnw-metrics.json` | MNW | external_mnw | 85 | 0.0238 | 0.0952 |
| `runs/full-20260904/evaluation/pilot-visual-test-metrics.json` | FakeAVCeleb | development_test | 597 | 1.0000 | 1.0000 |
| `runs/full-20260904/evaluation/pilot-visual-validation-metrics.json` | FakeAVCeleb | development_validation | 598 | 1.0000 | 1.0000 |
| `runs/initial-20260902/visual-validation-metrics.json` | FakeAVCeleb | development_validation | 400 | 0.9992 | 0.9975 |
| `runs/program-20260906/evaluation/audio-in-domain-test-metrics.json` | FakeAVCeleb | development_test | 1,648 | 1.0000 | 1.0000 |
| `runs/program-20260906/evaluation/fusion-dfdc-metrics.json` | FakeAVCeleb | fusion | 2,351 | 0.7500 | 0.6203 |
| `runs/program-20260906/evaluation/fusion-test-metrics.json` | FakeAVCeleb | fusion | 1,648 | 0.9990 | 0.9990 |
| `runs/program-20260906/evaluation/visual-celebdf-metrics.json` | Celeb-DF-v2 | generalization_celebdf | 462 | 0.6682 | 0.5435 |
| `runs/program-20260906/evaluation/visual-dfdc-metrics.json` | DFDC | generalization_celebdf | 2,410 | 0.7611 | 0.5677 |
| `runs/program-20260906/evaluation/visual-in-domain-test-metrics.json` | FakeAVCeleb | development_test | 1,648 | 0.9988 | 0.9990 |
| `runs/program-20260906/evaluation/visual-mnw-metrics.json` | MNW | external_mnw | 85 | 0.0000 | 0.1429 |
<!-- END GENERATED EVIDENCE -->

## MLflow store

<!-- BEGIN GENERATED MLFLOW -->
| Experiment | Runs |
|---|---|
| `design-b-20260910` | 5 (5 finished) |
| `design-b-frozen-bn` | 4 (4 finished) |
| `design-b-live-bn` | 2 (2 finished) |
| `ffpp-20260913` | 1 (1 finished) |
| `full-20260904` | 5 (5 finished) |
| `initial-baseline-20260902` | 10 (10 finished) |
| `program-20260906` | 21 (21 finished) |
| `prototype-gpu-20260902` | 7 (2 failed, 5 finished) |
| `smoke-fixture-fusion` | 10 (10 finished) |
| `streams-20260905` | 3 (3 finished) |
<!-- END GENERATED MLFLOW -->

## Evidence record

The evidence below came from the current primary-checkout files
`runs\initial-20260902\visual-initial-history.json`,
`runs\initial-20260902\visual-validation-metrics.json`,
`runs\initial-20260902\cache-audit.json`, and `mlflow.db`.

| Field | Recorded value |
|---|---:|
| Architecture | EfficientNet-B0 plus GRU |
| Training rows | 1,595 |
| Validation rows | 400 |
| Source overlap | 0 |
| Epochs | 5 |
| Best epoch | 4 |
| Final training loss | 0.0403575 |
| Final validation loss | 0.0194314 |
| Training throughput | 11.3568 samples/sec |
| Peak allocated GPU memory | 11,013.91 MiB |
| Fixed threshold | 0.5 |
| ROC AUC | 0.999175 |
| PR AUC | 0.999292 |
| Balanced accuracy | 0.9975 |
| F1 | 0.997494 |
| True negatives | 200 |
| True positives | 199 |
| False positives | 0 |
| False negatives | 1 |

The checkpoint SHA-256 is
`ac9a085e1017cf2743a7f78f3b632051c18acda695496d2f434c7d968fd627b0`.
The training run ID is `4243b35e64c743b89cc33000cc9d3d3e`. The evaluation run
ID is `56182266f70a424581f763b2d3b41989`. Both records use preprocessing hash
`fd372dbe6bb64f359db4d57b05c3b5cd27ed6660f2bb8bdc50567224e0928c96` and
split hash
`3255ae334536336c73058941285925f3dd5b094c02b1037e19f379c6f45db30c`.

These figures apply only to FakeAVCeleb development validation. They do not
measure Celeb-DF-v2, FaceForensics++, MNW, or cross-dataset performance.

## Recorded chronology

On 2026-09-02, the initial model supervisor logged a cache-build failure with
exit code 2. It restarted later that day, used the completed initial-model
cache, reported 1,595 usable training rows and 400 validation rows, then
recorded completion at 18:50:21 +05:30. The cache audit records one failed
clip, 12 unstable-face-track blockers, four audio-video-duration blockers,
and two low-face-coverage blockers across the cache attempt.

The SQLite MLflow record has three named experiments: `smoke-fixture-fusion`,
`prototype-gpu-20260902`, and `initial-baseline-20260902`. The initial baseline
has the finished training run and the finished fixed-threshold evaluation run
listed above. The historical training runtime artifact records an NVIDIA
GeForce RTX 5070 Ti, 16,302 MiB GPU memory, and CUDA package versions for
torch and torchvision. This is evidence from the recorded training run. It
does not describe the current Python environment.

The prototype experiment has two failed runs. Run
`52da7def729f415fbb43eddbad77a1b1` saved a `FileNotFoundError` for a cache
path that repeated `runs\prototype-20260902`. Run
`a8aaf6cc31144b36997dd7c3e30e607a` saved a CUDA deterministic-algorithm error
for `upsample_linear1d_backward_out_cuda`. Commit
`268957796d366a81b5ab897dd1a4f523f1dc4b11` changed sync token resizing to
deterministic nearest timestamp selection. The SQLite record also shows the
later sync prototype run `73915f8d22fc4b3eb31bf303f307cbc4` as finished. The
evidence does not record a separate narrative cause for either failure beyond
the saved exception messages.

Prototype visual, audio, sync, and fusion runs are marked `prototype_only` in
MLflow. The fusion run `7b799a76d4a74305b02742ded2033118` has dataset tag
`software_fixture`. It is not a trained research fusion model.

On 2026-09-06 the `program-v1` run trained all three branches at full scale,
exported out-of-fold features, fitted fusion and scored it on both partitions.
Three failures inside it are worth carrying forward, and all three are recorded
in [docs/obstacles.md](obstacles.md): a hardcoded preprocessing hash copied from
an earlier `code_version` rejected every clip; DataLoader workers died on the
77 MB visual batches until a retry-with-zero-workers fallback caught it; and
`ddf train fusion` exited 1 because out-of-fold features were exported from the
visual-usable manifest while strict assembly needs a clip usable by all three
branches. Twenty-one runs predating MLflow logging were backfilled into the
tracking store.

On 2026-09-08 the pipeline state above was committed to main as `5611653` and
tagged `pipeline-v1`, so Design B work can be reverted to a measured baseline.

## Current execution environment

The primary checkout currently detects an NVIDIA GeForce RTX 5070 Ti through
`nvidia-smi`. The driver is 596.49 and the reported memory is 16,303 MiB. The
CUDA toolkit environment points to 13.2.

The primary checkout `.venv` is a CUDA runtime. It has `torch 2.12.1+cu130`
and `torch.cuda.is_available()` is `True`, reporting the RTX 5070 Ti. It also
carries `mlflow 3.15.1`, `librosa 0.11.0` and `matplotlib 3.11.0`, which the
tracking reader and the teaching pages need.

The ignored visual checkpoint exists in the primary checkout. Its SHA-256
matches
`ac9a085e1017cf2743a7f78f3b632051c18acda695496d2f434c7d968fd627b0`.

Reproduce it with:

```powershell
uv sync --extra cu130 --extra ml --extra media --extra dashboard --extra tracking --group dev
```

Do not install both the `cpu` and `cu130` extras together. `pyproject.toml`
declares them as a conflicting pair, and installing `cpu` gives a Torch build
where `torch.cuda.is_available()` is `False` on a working GPU.

## Dashboard flow

```powershell
uv run streamlit run src\deepfake_detection\dashboard\app.py --server.address 127.0.0.1
```

Open `http://127.0.0.1:8501`. Sidebar order: Overview, Evidence gate,
Preprocessing, Streams (with Visual, Lip-Sync, Emotion, Audio branch and Sync
branch indented under it), Experiments, Fusion, Explainability, Documentation.
Fusion and Explainability are dimmed and unclickable.

The dashboard reads the ignored `runs` artifacts of the active checkout, so
launch it from a checkout where they are present.

The Evidence gate is the only page that produces a verdict. Video input accepts
one local video, image or sound file and says which streams it can drive before
anything runs. Preprocessing builds the visual view after the user starts it.
Prediction loads the multimodal engine only after the server's own preprocessing
hash matches the one the fusion head was fitted under; when it falls back to the
frozen visual engine, that engine still checks its checkpoint hash, run ID, split
hash, commit, seed and preprocessing hash against the dashboard defaults. Derived state is keyed by the upload's SHA-256, so a new upload
invalidates preprocessing and the prediction rather than leaving a stale result
on screen.

Experiments reads the local history and metrics JSON, cross-checks shared
provenance, requires the FakeAVCeleb development-validation scope, and labels
every result with that scope. Below that it reads the local MLflow store
directly and puts every recorded run in one comparison table.

The teaching pages need no checkpoint and no data. Preprocessing shows every
step as a toggle applied cumulatively, ending in the exact tensor a model would
receive. The Streams pages build EfficientNet-B0 or DINOv3 from one
`StreamConfig` and show what each backbone stage responded to. They load
whatever checkpoint you pick, from `checkpoints/<stream>/` or from an MLflow
run, and report exactly which tensors did and did not fit.

Audio and sync are prototype pages. Full training is incomplete, so they load no
checkpoint and calculate no probability. Fusion is locked: its current artifact
is a software fixture, so the page does not load it or return a probability.

The dashboard reports missing local evidence as an error. It does not query a
remote service, load a substitute artifact, or create a metric when the local
record is absent. It never trains a model and never writes into `data/`.

## Local MLflow

Use the primary checkout when its database and artifact root are present:

```powershell
.\.venv\Scripts\mlflow.exe server `
  --backend-store-uri sqlite:///C:/Users/vaibh/Documents/GitHub/deepfake-generalization/mlflow.db `
  --default-artifact-root file:///C:/Users/vaibh/Documents/GitHub/deepfake-generalization/mlartifacts `
  --host 127.0.0.1 `
  --port 5000
```

Open `http://127.0.0.1:5000`. Select `initial-baseline-20260902`, then choose
training run `4243b35e64c743b89cc33000cc9d3d3e` or evaluation run
`56182266f70a424581f763b2d3b41989`. The dashboard's Experiments page shows the
same runs in one table without a server.

## Run comparison

The baseline had no variation to compare against: every recorded run used seed
17, one backbone and one learning rate. Four more runs were trained on the same
cached corpus to give it one. Every run shares the cache index, cache root,
split hash and preprocessing hash of the baseline, writes its own checkpoint and
history, and is evaluated on the same 400 source-disjoint validation rows at the
same fixed threshold of 0.5. Only the named field changes.

| Run | Seed | LR | Epochs | Best epoch | Validation loss | ROC AUC | PR AUC | Balanced accuracy | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `initial-seed17` (frozen baseline) | 17 | 1e-4 | 5 | 4 | 0.019431 | 0.999175 | 0.999292 | 0.9975 | 1 |
| `initial-seed29` | 29 | 1e-4 | 5 | 5 | 0.014301 | 0.999950 | 0.999950 | 0.9975 | 1 |
| `initial-seed43` | 43 | 1e-4 | 5 | 2 | 0.039375 | 0.999800 | 0.999808 | 0.9925 | 3 |
| `lr3e-4-seed17` | 17 | 3e-4 | 5 | 4 | 0.026085 | 1.000000 | 1.000000 | 0.9950 | 2 |
| `ep8-seed17` | 17 | 1e-4 | 8 | 4 | 0.020293 | 0.999175 | 0.999292 | 0.9975 | 1 |

Training and evaluation run IDs:

| Run | Training | Evaluation |
|---|---|---|
| `initial-seed17` | `4243b35e64c743b89cc33000cc9d3d3e` | `56182266f70a424581f763b2d3b41989` |
| `initial-seed29` | `9aebb7e32a9249c3a24562e11505706a` | `fca2d73320c842d8888729e87085766b` |
| `initial-seed43` | `0575b78b4b304c8dba3fdea75d5999bf` | `625967be05e54803800f51eb20bcee36` |
| `lr3e-4-seed17` | `ff1af7990cf1437abd8d0930ec226a16` | `84d7c7cac3684242b44f1e0ad450e6e2` |
| `ep8-seed17` | `eb628e801f664dadb6cdd349dfa9cfa1` | `0f8a10530b084adea04cb70e7b23cffb` |

What the comparison shows:

- Seed dominates. Validation loss ranges from 0.0143 to 0.0394 across three
  seeds at identical settings, a 2.8-fold spread. The baseline's 0.0194 sits in
  the middle of that range, so it is one draw rather than a tuned result.
- The error count moves with it: one misclassified row at seed 17 and 29, three
  at seed 43, out of 400.
- Neither the higher learning rate nor the longer schedule beats seed noise.
  `lr3e-4` gains on ROC AUC and loses on balanced accuracy; `ep8` matches the
  baseline exactly, stopping at the same best epoch 4.
- Every run scores above 0.99 on this split. That ceiling is a property of
  FakeAVCeleb development validation, not evidence of generalization, and it is
  why no single number here should be read as a result.

The comparison runs are tagged `evidence_scope: development_comparison` to keep
them apart from the frozen `development_baseline` and `development_validation`
records. They do not replace the baseline, and the dashboard still loads only
the frozen checkpoint.

The ignored `runs/comparison-20260904/` directory holds the configs, the
checkpoints, the per-run history and metrics JSON, the per-run prediction CSVs,
and the two shell runners that produced them.

## Verification limits and next work

### Limits on everything above

Single seed throughout, so every result is `provisional` rather than `accepted`
under the protocol in [research design](research-design.md), which asks for
three. That is the single largest gap and it is about eight hours of GPU.

The DF26 intervals are bootstrap over clips, not clustered on identity, because
generated clips have no speaker to cluster on. They are therefore narrower than
the identity-clustered intervals elsewhere here and should not be compared
against them directly.

The DF26 probe uses DINOv3 because it is cached locally and works offline.
Perception Encoder is about 15 AUROC points better on this task and needs
`open_clip_torch`, which is not installed and the venv has no pip.

The commercial generators still outscore the open-source ones after the
container control, 0.72 against 0.62, which inverts the DF26 paper. The
remaining explanation is training set rather than shortcut: their detectors
transfer from face manipulation, this probe transfers between generative models,
and the four commercial generators were all produced through one platform and
may simply be more similar to each other. That is untested.

### Next, in order

1. **Three seeds.** The only thing standing between every result and `accepted`.
   About eight hours of GPU, no new data.
2. **The face-crop path.** Measured as necessary: full frames are at chance on
   face manipulation, face crops are at chance on person-free generated scenes,
   so both paths must exist and be routed by content.
3. **Perception Encoder**, once `open_clip_torch` is installable.
4. **The audiovisual track**, which needs a corpus DF26 cannot provide. LAV-DF
   is on disk; DeepSpeak v2 and AV-Deepfake1M need their access granted.
5. **The paper.** [paper-source.md](research/paper-source.md) regenerates its
   tables from `runs/` and is the place to write from;
   [paper.md](research/paper.md) is the draft.

### Commands that reproduce the current state

```powershell
uv run python scripts/fetch_df26.py
uv run python scripts/normalize_media.py --source data/df26-scored --target data/df26-norm --fps 24
uv run python scripts/extract_probe_features.py --folder data/df26-norm --output runs/probe-20260914/df26-norm-dinov3.npz
uv run python scripts/train_logo_probe.py --features runs/probe-20260914/df26-norm-dinov3.npz
uv run python scripts/update_result_registry.py
uv run python scripts/update_paper_source.py
```
