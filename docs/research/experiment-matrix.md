# Experiment matrix

## Controls

All comparisons use one frozen source split, preprocessing hash, cache index,
training budget, early-stopping rule, and evaluation implementation. MNW is
evaluation-only. It cannot select a model or threshold.

## Fixed seeds

Research branch comparisons use seeds 17, 29, and 43. A candidate remains
incomplete until all three runs finish or retain an explained failure.

## Experiment stages

| ID | Stage | Candidates | Selection evidence | Status |
| --- | --- | --- | --- | --- |
| DET-01 | Detector | MTCNN, YuNet | Reviewed training-only benchmark | planned |
| VIS-01 | Visual | EfficientNet-B0 plus GRU, ConvNeXt-Tiny | Validation and method-holdout metrics | planned |
| AUD-01 | Audio | Wav2Vec2 Base, WavLM, AASIST | Validation and method-holdout metrics | planned |
| SYN-01 | Sync | Current temporal branch, SyncNet-style baseline | Offset and mismatch metrics | planned |
| FUS-01 | Fusion | Logistic regression, small MLP | Out-of-fold validation metrics | planned |
| EXT-01 | External | Frozen selected system on MNW | Locked zero-shot metrics | planned |

## Status rules

Use only `planned`, `running`, `failed`, `accepted`, or `superseded`. Add MLflow
run IDs only after runs exist. Never convert smoke fixture metrics into a
research result.
