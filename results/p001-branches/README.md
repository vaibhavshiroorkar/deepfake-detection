# P-001 visual and audio paths

The two supervised paths that H2 fuses with the sync path (amendment A8). Both train on the neutral FakeAVCeleb training partition at seeds 17, 29 and 43, pick their epoch and threshold on validation, and are then frozen. The targets are scored by the frozen models; nothing is retrained, selected or thresholded on a target.

- **Visual:** EfficientNet-B0 with a GRU over face frames, trained on the video label.
- **Audio:** Wav2Vec2 Base with attentive pooling, trained on the audio label.

| Path | Set | Clips | Seed 17 | Seed 29 | Seed 43 |
|---|---|---|---|---|---|
| Visual | FakeAVCeleb test, identity-strict | 495 | 0.989 [0.979, 0.997] | 0.988 [0.979, 0.995] | 0.990 [0.982, 0.996] |
| Visual | DFDC | 2,858 | 0.726 [0.693, 0.758] | 0.671 [0.635, 0.708] | 0.693 [0.654, 0.731] |
| Visual | LAV-DF test, file label | 3,815 | 0.352 [0.343, 0.362] | 0.391 [0.381, 0.399] | 0.389 [0.380, 0.399] |
| Visual | HiFi-AVDF | 3,537 | 0.549 [0.534, 0.562] | 0.544 [0.529, 0.558] | 0.550 [0.535, 0.563] |
| Audio | FakeAVCeleb test, identity-strict | 495 | 1.000 [1.000, 1.000] | 0.952 [0.931, 0.971] | 1.000 [1.000, 1.000] |
| Audio | DFDC | 3,032 | one class | one class | one class |
| Audio | LAV-DF test, file label | 3,823 | 0.653 [0.638, 0.671] | 0.576 [0.566, 0.586] | 0.648 [0.634, 0.662] |
| Audio | HiFi-AVDF | 3,784 | 0.750 [0.733, 0.767] | 0.719 [0.704, 0.736] | 0.757 [0.742, 0.773] |

ROC-AUC on each path's own label, with 95% identity-bootstrap intervals (1,000 resamples). A reading aid: the deliverables take every number from the files here through `results/numbers.json`. One HiFi-AVDF clip (`sora2/1289`) could not be decoded and was not cached; 271 more have no usable face window and 24 no usable audio.

## Reading

- **The visual path is below chance on LAV-DF** in every seed, on every fake type, including clips whose video was never edited (0.175 to 0.243 against the reals). So it reads some difference between LAV-DF's real and fake files that is not the face edit, and reads it backwards. What that difference is, is unknown; the file-cue audit reads metadata and cannot see pixels.
- **The visual path is near chance on HiFi-AVDF**, whose fakes come from text-to-video generators rather than face swaps.
- **The audio path scores 0.72 to 0.76 on HiFi-AVDF**, above that corpus's neutral audio-label file-cue ceiling of 0.636. Whether it detects generated speech or a cue the ceiling did not measure is unknown.
- **The audio path has no DFDC result.** Every clip in our DFDC subset has real audio, so the audio-label set has one class and no AUC.
- **The audio path scores 1.000 on FakeAVCeleb in two seeds.** The neutral audio-label ceiling is 0.793 ([`shortcut-audit/fakeavceleb-neutral`](../shortcut-audit/fakeavceleb-neutral/summary.json)), so part of this may be a residual audio cue rather than voice cloning. Assumed, not tested.

## Files

- `*-predictions.csv`: one row per clip (`probability`, `predicted`, `label`, `method`).
- `*-metrics.json`: `ddf evaluate predictions` on those predictions, with identity-bootstrap intervals. These feed `numbers.json`.
- `*-branch-report.json`: the report `ddf evaluate branch` wrote while scoring, kept for its per-method and calibration detail. It has no intervals.
- `*-threshold.json`: the validation threshold of each model.
- `receipts/`: content-bound receipts for each training and evaluation run.
