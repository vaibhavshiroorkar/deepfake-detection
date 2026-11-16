# P-001 sync path on the locked targets

The three sync models of [`results/p001-sync`](../p001-sync/README.md), frozen, scored on neutral copies of three locked corpora (`scripts/score_sync_targets.py`). Nothing was retrained, selected or thresholded on a target. Each target was neutralized with v4, and these files hold the results after amendment A10. The results from before A10 are in [`results/p001-sync-targets-pre-a10`](../p001-sync-targets-pre-a10/).

| Target | Clips scored | Seed 17 | Seed 29 | Seed 43 | Neutral ceiling |
|---|---|---|---|---|---|
| LAV-DF test, file label | 3,693 of 3,823 | 0.597 [0.581, 0.615] | 0.622 [0.606, 0.638] | 0.622 [0.606, 0.638] | 0.557 |
| LAV-DF test, window label (A9) | 3,693 of 3,823 | 0.634 [0.605, 0.662] | 0.657 [0.628, 0.685] | 0.655 [0.625, 0.682] | see `shortcut-audit/lavdf-window-neutral` |
| DFDC | 2,908 of 3,032 | 0.553 [0.517, 0.587] | 0.575 [0.542, 0.607] | 0.565 [0.531, 0.598] | 0.526 |
| HiFi-AVDF | 3,739 of 3,808 | 0.559 [0.539, 0.577] | 0.570 [0.551, 0.587] | 0.572 [0.554, 0.590] | 0.629 |

ROC-AUC with 95% identity-bootstrap intervals (1,000 resamples). The numbers in the deliverables come from the metrics files here through `results/numbers.json`; this table is a reading aid.

## What changed the inputs, and when

- **A9, LAV-DF window label.** The neutral copy keeps 3.00 s from the first sound. In 1,307 of the 2,823 LAV-DF fake clips, that window holds no edited time. The window label counts a clip as fake only if its window overlaps edited time by at least one frame (`scripts/lavdf_window_labels.py`). Found after the file-label result was read; both are reported.
- **A10, face locator.** The face landmarker missed small faces, so DFDC first scored 921 clips with 67 real ones. Windows below the 0.8 face share were redone with a YuNet locator. Coverage then rose to 2,908 clips on DFDC and 3,739 on HiFi-AVDF; LAV-DF did not change. Found after the DFDC result was read; both are reported.
- **Short windows.** Windows with fewer than 50 frames (video shorter than the audio) are left out and counted: 4 on HiFi-AVDF before A10 (handbook mistake 67).
- **No audio.** 23 HiFi-AVDF source clips have no audio track (14 real, 9 fake); a sync path cannot score them.

## Reading

- The sync path stays above chance on every target but well below its in-domain 0.78 to 0.81.
- On HiFi-AVDF it is below the neutral file-cue ceiling. Per generator it ranges from about 0.70 to 0.78 (PixVerse V5) down to below chance on Sora 2. Generators that make sound and picture together leave little lip mismatch to find.
- On DFDC, whose fakes keep their original voice, it is just above the neutral ceiling.

Each evaluation has a content-bound receipt in `receipts/`. The receipts of the pre-A10 results record the paths they had before they were moved.
