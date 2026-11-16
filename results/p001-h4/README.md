# P-001 H4: false alarms set from real target clips

For each target and seed, `scripts/run_h4.py` draws 200 real target clips as a calibration set (seed 20261113, fixed in P-001) and reads no fake label. Three thresholds are then scored on the remaining clips:

- **source validation:** the FakeAVCeleb validation threshold, unchanged;
- **conformal, real only:** the k-th smallest real calibration score, k = ceil((n + 1)(1 - alpha)), alpha = 0.05;
- **umbrella, real only:** the Neyman-Pearson umbrella threshold, alpha = delta = 0.05.

The preregistered bar is a realized false-positive rate of at most 0.07 with the conformal rule on every target.

| Target | Seed | Source FPR | Conformal FPR | Conformal TPR | Met |
|---|---|---|---|---|---|
| HiFi-AVDF | 17 / 29 / 43 | 0.48 / 0.47 / 0.40 | 0.057 / 0.055 / 0.047 | 0.073 / 0.085 / 0.074 | yes / yes / yes |
| LAV-DF, file label | 17 / 29 / 43 | 0.43 / 0.37 / 0.31 | 0.056 / 0.069 / 0.063 | 0.051 / 0.079 / 0.062 | yes / yes / yes |
| LAV-DF, window label (A9) | 17 / 29 / 43 | 0.44 / 0.38 / 0.33 | 0.086 / 0.056 / 0.075 | 0.110 / 0.087 / 0.114 | no / yes / no |

A reading aid; the deliverables take every number from the JSON files here through `results/numbers.json`.

- **The real-only threshold brings false alarms back to about 5%** where the carried-over threshold flags 31 to 48% of real clips. The cost is recall: at that threshold the sync path catches 5 to 11% of fakes, because its AUC on these targets is low.
- **Under the window label, two of three LAV-DF seeds miss the bar.** The window label turns 1,307 clips from fake files into real ones, and those clips are cut from edited recordings, so calibration and test reals may no longer come from one distribution, which the conformal guarantee assumes. This is reported, not explained away.
- **DFDC is not run.** Our DFDC subset has 190 scored real clips (194 in all), fewer than the 200 that calibration needs.
- **One calibration draw per row.** The conformal guarantee holds on average over draws, so a single draw can land above alpha by chance.

The H4 results from before amendment A10 are in [`results/p001-h4-pre-a10`](../p001-h4-pre-a10/). A10 changed only HiFi-AVDF here, and it is met under both.
