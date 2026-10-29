# Project status

## Current state

The next experiments must use the corrected data and checkpoint contracts in
[reproducibility](reproducibility.md). Historical artifacts remain on disk.
The result registry marks affected rows `requires rerun`.

The PE/CLIP features used incorrect input normalization. The old PE macro score
of 0.7527 does not establish performance after the correction. Re-extract all
comparison arms into a new directory with `scripts/run_encoder_comparison.sh`.
It runs three source-split seeds and writes per-clip predictions and completion receipts.

The old OOF folds used the outer holdout for epoch selection. Run
`scripts/run_crossfit_seeds.ps1` to create inner validation folds and retrain.
The new runner writes separate evidence and checks training and selection sources
before exporting the outer holdout.

The LAV-DF cache must be rebuilt from manifests that retain explicit zero starts.
The old default shifted those windows, which can contaminate genuine labels.
Retrain and rescore the lip-sync stream before interpreting its old 0.9969 score.
New streams record temporal positions. Legacy checkpoints retain their original computation.

### Where the four stated targets stand

| Target | Status | Evidence |
| --- | --- | --- |
| 0.991 visual detection, in-domain | met by the historical visual branch | Single visual branch 0.9988; independent of the OOF head |
| 0.844+ from audiovisual cues on genuine-frame fakes | requires rerun | LAV-DF window contamination and an order-blind legacy stream |
| Fusion beats the strongest single stream | requires rerun | Original OOF selection used the holdout labels |
| 0.64+ on unseen generators without fine-tuning | requires rerun for the PE claim | PE normalization was wrong; corrected result is unknown |

The AV-HuBERT and HSEmotion checkpoints in `runs/encoders-20261022` are historical
artifacts. Their presence does not establish completed training. The corrected
runner writes into a new run and verifies completion before reuse.
DeepSpeak v2 needs author review. AV-Deepfake1M needs a signed EULA.
Deepfake-Eval-2024 access was refused.

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
| `runs/program-20260906/evaluation/audio-in-domain-test-seed29-metrics.json` | FakeAVCeleb | development_test | 1,648 | 1.0000 | 0.9962 |
| `runs/program-20260906/evaluation/audio-in-domain-test-seed43-metrics.json` | FakeAVCeleb | development_test | 1,648 | 0.9986 | 0.9906 |
| `runs/program-20260906/evaluation/audio-wavlm-in-domain-test-metrics.json` | FakeAVCeleb | development_test | 1,648 | 1.0000 | 0.9981 |
| `runs/program-20260906/evaluation/fusion-dfdc-metrics.json` | FakeAVCeleb | fusion | 2,351 | 0.7500 | 0.6203 |
| `runs/program-20260906/evaluation/fusion-test-metrics.json` | FakeAVCeleb | fusion | 1,648 | 0.9990 | 0.9990 |
| `runs/program-20260906/evaluation/sync-final-sync-seed17-metrics.json` | FakeAVCeleb | ? | 0 | 0.4364 | 0.5000 |
| `runs/program-20260906/evaluation/sync-final-sync-seed29-metrics.json` | FakeAVCeleb | ? | 0 | 0.4570 | 0.5000 |
| `runs/program-20260906/evaluation/sync-final-sync-seed43-metrics.json` | FakeAVCeleb | ? | 0 | 0.4931 | 0.5000 |
| `runs/program-20260906/evaluation/visual-celebdf-metrics.json` | Celeb-DF-v2 | generalization_celebdf | 462 | 0.6682 | 0.5435 |
| `runs/program-20260906/evaluation/visual-convnext-dfdc-metrics.json` | DFDC | generalization_celebdf | 2,410 | 0.7272 | 0.5066 |
| `runs/program-20260906/evaluation/visual-convnext-dfdc-seed29-metrics.json` | DFDC | generalization_celebdf | 2,410 | 0.6776 | 0.5358 |
| `runs/program-20260906/evaluation/visual-convnext-dfdc-seed43-metrics.json` | DFDC | generalization_celebdf | 2,410 | 0.6301 | 0.5720 |
| `runs/program-20260906/evaluation/visual-convnext-in-domain-test-metrics.json` | FakeAVCeleb | development_test | 1,648 | 0.9992 | 0.9973 |
| `runs/program-20260906/evaluation/visual-convnext-in-domain-test-seed29-metrics.json` | FakeAVCeleb | development_test | 1,648 | 0.9986 | 0.9957 |
| `runs/program-20260906/evaluation/visual-convnext-in-domain-test-seed43-metrics.json` | FakeAVCeleb | development_test | 1,648 | 0.9032 | 0.7919 |
| `runs/program-20260906/evaluation/visual-dfdc-metrics.json` | DFDC | generalization_celebdf | 2,410 | 0.7611 | 0.5677 |
| `runs/program-20260906/evaluation/visual-dfdc-seed17-metrics.json` | DFDC | generalization_celebdf | 2,410 | 0.7611 | 0.5677 |
| `runs/program-20260906/evaluation/visual-dfdc-seed29-metrics.json` | DFDC | generalization_celebdf | 2,410 | 0.6963 | 0.5354 |
| `runs/program-20260906/evaluation/visual-dfdc-seed43-metrics.json` | DFDC | generalization_celebdf | 2,410 | 0.6878 | 0.5216 |
| `runs/program-20260906/evaluation/visual-in-domain-test-metrics.json` | FakeAVCeleb | development_test | 1,648 | 0.9988 | 0.9990 |
| `runs/program-20260906/evaluation/visual-in-domain-test-seed17-metrics.json` | FakeAVCeleb | development_test | 1,648 | 0.9988 | 0.9990 |
| `runs/program-20260906/evaluation/visual-in-domain-test-seed29-metrics.json` | FakeAVCeleb | development_test | 1,648 | 0.9991 | 0.9990 |
| `runs/program-20260906/evaluation/visual-in-domain-test-seed43-metrics.json` | FakeAVCeleb | development_test | 1,648 | 0.9993 | 0.9987 |
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

## Dashboard and web app

The web app (`ddf-web`, see the README) is the only path that issues a verdict.
The dashboard is a teaching tool; [dashboard.md](dashboard.md) describes its
pages. Both read the ignored `runs` artifacts of the active checkout, so launch
them from a checkout where those are present.

## Local MLflow

Run from the checkout that holds `mlflow.db` and `mlartifacts`:

```powershell
$researchRoot = (Get-Location).Path.Replace('\', '/')
.\.venv\Scripts\mlflow.exe server `
  --backend-store-uri "sqlite:///$researchRoot/mlflow.db" `
  --default-artifact-root "file:///$researchRoot/mlartifacts" `
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

Single seed for the audio and sync branches, so nearly every result is
`provisional` rather than `accepted` under the protocol in
[research design](research-design.md), which asks for three. That is still the
single largest gap.

The visual branch now has all three, and what they show is the reason the gap
matters: cross-corpus, seed alone moves DFDC by 0.0733, which is wider than most
effects reported here, and the reported seed is the most favourable of the
three. Read every single-seed cross-corpus number in this document with that
spread in mind.

The DF26 intervals are bootstrap over clips, not clustered on identity, because
generated clips have no speaker to cluster on. They are therefore narrower than
the identity-clustered intervals elsewhere here and should not be compared
against them directly.

The DF26 probe uses Perception Encoder. The claim that it beat DINOv3 by about
15 points was withdrawn for weeks because no artifact existed. The re-run in
`A-encoder-comparison` confirms the direction at half the size: +0.0741
[+0.0121, +0.1317], paired and resampled by source video. CLIP is level with PE.
That also retires the mechanism argument for DINOv3, that a vision-language
objective trades away forensic detail; on generated video it did not.

The commercial generators still outscore the open-source ones after the
container control, 0.81 against 0.68 under PE (0.72 against 0.62 under DINOv3),
which inverts the DF26 paper. The
remaining explanation is training set rather than shortcut: their detectors
transfer from face manipulation, this probe transfers between generative models,
and the four commercial generators were all produced through one platform and
may simply be more similar to each other. That is untested.

### Next, in order

0. **Finish the encoder comparison, then re-measure DF26.** Score the AV-HuBERT
   and HSEmotion streams beside the earlier ones (`scripts/score_streams.py
   --run-dir runs/encoders-20261022`), compare five-stream fusion with old and
   new encoders (`scripts/run_deep_ablation.py --streams ... --sizes 1 5`), and
   score lip-sync with no training (`scripts/avhubert_sync_score.py`). Then
   re-extract the DF26 Perception Encoder and CLIP features with the corrected
   normalisation (F33) and run the frozen face-crop probe
   (`scripts/face_crop_probe.py`), the face-crop cell F13 left empty. Before
   another seed of the F37 recipes, find the loader-worker memory growth
   (obstacles).

1. **Report seeds, not a seed.** All three branches and the fusion head now have
   seeds 17, 29 and 43; each seed's head is fitted on its own cross-fitted fold
   models. What that shows:

   | System | In-domain spread | DFDC spread |
   | --- | ---: | ---: |
   | visual branch (F21) | 0.0005 | 0.0733 |
   | fused system (F27) | 0.0000 | **0.1078** |
   | sync branch (F26) | 0.0567 | not scored |

   The fused DFDC headline of 0.7500 is the **best of three**; the mean is
   0.6873. Combining branches compounded the noise rather than averaging it.
   The paired fusion-against-visual comparison now has all three seeds (F19).
   It never favours fusion on DFDC, and at seeds 29 and 43 fusion is measurably
   worse. Most other Design A rows still quote seed 17 alone; rerunning them per
   seed is the remaining seed work.

2. **Three seeds per challenger, before any new challenger.** VIS-01 and AUD-01
   are both run and both undecided, and neither is undecided because the
   architectures are close. ConvNeXt differs from EfficientNet by 0.0340 on
   DFDC against a 0.0733 seed spread (F29), and WavLM and Wav2Vec2 both sit at
   the in-domain ceiling with no cross-corpus partition to separate them (F28).
   A third candidate cannot resolve a comparison that cannot resolve two.
   AASIST and the SyncNet-style baseline are the scope controls' "one or two
   challengers" already spent.

3. **A corpus with manipulated audio that is not VoxCeleb2.** The audio branch
   is at chance cross-corpus, 0.5054 [0.4747, 0.5371] on DFDC against 0.7739
   in-domain (F30), so two of the three branches do not transfer and only the
   visual one does.

   DFDC can only answer this at clip level: it does not publish which modality
   each forgery manipulated, so a cue-specific audio test needs a corpus that
   does. LAV-DF has audio manipulations but is built on VoxCeleb2 exactly as
   FakeAVCeleb is, which by this project's own rule is a different-generator
   test rather than a cross-corpus one. DeepSpeak v2 and AV-Deepfake1M are the
   candidates and both are blocked on access.

4. **The face-crop path.** Measured as necessary: full frames are at chance on
   face manipulation, face crops are at chance on person-free generated scenes,
   so both paths must exist and be routed by content.

5. **The audiovisual track**, which needs a corpus DF26 cannot provide. LAV-DF
   is on disk; DeepSpeak v2 and AV-Deepfake1M need their access granted.

6. **The paper.** [paper-source.md](research/paper-source.md) regenerates its
   tables from `runs/` and is the place to write from;
   [paper.md](research/paper.md) is the draft.

### What a run actually costs

Measured on 2026-09-23 and 2026-09-24, because the estimates this file carried
were wrong by two orders of magnitude and deterred the replication for weeks:

| Job | Measured |
| --- | --- |
| audio branch, full scale | 8 to 16 minutes |
| sync branch, full scale | 12 minutes |
| visual branch, full scale | about 40 minutes |
| feature export, one partition | 3m22s |
| BatchNorm sweep, three arms | about 75 minutes |

Visual training must run with `--workers 0` on this machine. A visual batch is
77 MB and prefetching workers have now taken the host out of memory twice.

### Commands that reproduce the current state

```powershell
uv run python scripts/fetch_df26.py
uv run python scripts/normalize_media.py --source data/df26-scored --target data/df26-norm --fps 24
uv sync --extra probe
uv run python scripts/extract_probe_features.py --folder data/df26-norm --encoder pe --output runs/probe-20260914/df26-norm-pe.npz
uv run python scripts/train_logo_probe.py --features runs/probe-20260914/df26-norm-pe.npz
uv run python scripts/compare_probe_encoders.py --features runs/probe-20260914/df26-norm-dinov3.npz runs/probe-20260914/df26-norm-pe.npz runs/probe-20260914/df26-norm-clip.npz --output runs/probe-20260914/df26-norm-encoder-comparison.json
uv run python scripts/update_result_registry.py
uv run python scripts/update_paper_source.py
```
