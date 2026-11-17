# P-001 H2: fusion against visual-only

The fusion test fixed in amendment A8 before any visual or audio model existed (`scripts/run_h2.py`, `src/deepfake_detection/fusion/h2.py`). A logistic regression on the three standardized branch logits (visual, audio, sync) is fit on FakeAVCeleb validation. Its threshold comes from 5-fold cross-fitting grouped by identity. It is then frozen and applied to the test set and the targets. A clip counts only if all three paths score it, and the visual path is compared on exactly those clips.

The statistic is AUC(fusion) minus AUC(visual) on the clip label (fake if either stream is fake), averaged over seeds 17, 29 and 43. One identity bootstrap of 1,000 resamples is shared by the seeds. Holm correction at 0.05 runs over DFDC, LAV-DF (window label, A9) and the identity-strict FakeAVCeleb test. H2 holds if at least two survive.

| Target | Clips | Seed 17 | Seed 29 | Seed 43 | Mean [95%] | p | Holm |
|---|---|---|---|---|---|---|---|
| FakeAVCeleb test, strict | 495 | +0.093 | +0.080 | +0.092 | +0.088 [+0.078, +0.100] | 0.000 | yes |
| DFDC | 2,786 | -0.007 | +0.008 | +0.008 | +0.003 [-0.008, +0.013] | 0.305 | no |
| LAV-DF test, window label | 3,686 | +0.205 | +0.094 | +0.212 | +0.170 [+0.156, +0.183] | 0.000 | yes |
| HiFi-AVDF (reported) | 3,485 | +0.056 | +0.046 | +0.066 | +0.056 [+0.049, +0.063] | 0.000 | n/a |

**H2 holds** (two of three survive). HiFi-AVDF is reported outside the rule. One HiFi-AVDF clip (`sora2/1289`) could not be decoded and was not cached. A reading aid: the deliverables take every number from `h2.json` and the prediction files here through `results/numbers.json`.

## Reading

The differences hide their levels. Each path's AUC on the clip label, over the same clips (lowest and highest of three seeds):

| Target | Visual | Audio | Sync | Fusion |
|---|---|---|---|---|
| FakeAVCeleb test, strict | 0.898, 0.900 | 0.703, 0.708 | 0.780, 0.810 | 0.979, 0.992 |
| DFDC | 0.679, 0.729 | 0.513, 0.547 | 0.557, 0.579 | 0.687, 0.723 |
| LAV-DF test, window label | 0.369, 0.410 | 0.632, 0.848 | 0.635, 0.658 | 0.503, 0.622 |
| HiFi-AVDF (reported) | 0.541, 0.548 | 0.682, 0.709 | 0.561, 0.574 | 0.587, 0.614 |

- **FakeAVCeleb:** the gain is mostly the cloned-voice clips. The visual path scores 0.99 on its own video label but 0.90 on the clip label, and fusion recovers the fakes it was never trained to see.
- **LAV-DF:** the visual path is below chance, so fusion beats a reversed baseline. Fusion still trails the audio path and the sync path alone in every seed. Its weights come from FakeAVCeleb, where the visual path is strong, which is the likely reason (assumed, not tested).
- **DFDC:** every clip in our subset has real audio. The audio path is near chance and fusion adds nothing.
- **HiFi-AVDF:** the LAV-DF pattern again. Fusion beats a visual path near chance but trails the audio path alone in every seed.
- **Stability:** the audio weight is 6.19, 0.48 and 2.62 across seeds, and the LAV-DF gain ranges from +0.09 to +0.21.

So H2 supports adding the audio and sync paths to a visual detector. It does not show that this fusion transfers to a new corpus.

## Files

- `h2.json`: the fusion weights per seed, the per-target differences, the bootstrap and the Holm decision. `not_scored_yet` lists a reported-only target that was not scored when the file was written.
- `fusion-{target}-seed{S}-predictions.csv`: per clip, the label, the fusion probability and the three branch logits. `test` and `lavdf-test` (file label) are reported but are not in the Holm family.
- `receipts/h2-family.json`: the receipt of the run on the Holm family (11-16, 22:56), which made the H2 decision. Its outputs stay in `runs/p001/h2/`.
- `receipts/h2-full.json`: the receipt of the run with HiFi-AVDF (11-17), whose outputs are the files here. It wrote to `runs/p001/h2-full/` because the receipt tool refused to overwrite the family run. Its family targets, Holm decision, weights and the 15 shared prediction files match the family run byte for byte.
