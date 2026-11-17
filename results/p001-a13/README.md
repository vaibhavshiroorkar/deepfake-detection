# P-001 amendment A13: what does the visual path read on LAV-DF?

**Exploratory.** Fixed in [amendment A13](../preregistration/P-001-shortcut-neutral-av-study.md) before this result existed. No P-001 verdict changes.

The P-001 visual path scores LAV-DF fakes whose video was never edited (RealVideo-FakeAudio) below the real clips, so something in the frames separates LAV-DF's real and fake files besides the edit. A13 measures six statistics of the cached face frames the path sees (`scripts/run_a13.py`), on the 1,936 scored LAV-DF real-video clips (998 real, 938 unedited-video fakes). FakeAVCeleb (75 real against 352 video fakes, identity-strict test) is measured the same way, to show what the path learned.

| Statistic | LAV-DF AUC [95%] | rho with visual score | FakeAVCeleb AUC [95%] |
|---|---|---|---|
| Sharpness | 0.454 [0.442, 0.464] | +0.01 | 0.448 [0.406, 0.480] |
| High-frequency share | 0.500 [0.488, 0.511] | -0.04 | 0.484 [0.453, 0.513] |
| 8x8 blockiness | 0.520 [0.503, 0.536] | -0.05 | 0.594 [0.543, 0.645] |
| Brightness | 0.505 [0.495, 0.517] | +0.02 | 0.464 [0.425, 0.494] |
| Contrast | 0.492 [0.481, 0.502] | +0.08 | 0.479 [0.444, 0.511] |
| Temporal change | 0.450 [0.438, 0.462] | +0.22 | 0.509 [0.472, 0.546] |
| **All six, cross-fit** | **0.564** | | **0.589** |
| **The visual path itself** | **0.196** | | **0.989** |

AUC for the fake group against the reals, with identity-bootstrap intervals (1,000 resamples). rho is the Spearman correlation with the visual path's probability (mean of three seeds) on the LAV-DF clips. A reading aid: the deliverables take every number from `a13.json` through `results/numbers.json`.

## Reading, by the rule fixed in A13

- **No candidate.** A statistic qualifies only if its AUC lies outside 0.4 to 0.6 and its correlation with the visual score is at least 0.3 in size. None does: every LAV-DF AUC lies between 0.450 and 0.520, and the largest correlation is +0.22.
- **So the difference stays unexplained.** The visual path separates the two LAV-DF groups at 0.196 (0.804 inverted); these six statistics together reach 0.564. Whatever it reads is not sharpness, compression blocks, brightness, contrast or motion as measured here.
- **The same holds on FakeAVCeleb:** the path reaches 0.989 while the six statistics reach 0.589, so it learned something these measures do not capture on its own training data too.
- A11 adds one clue: frozen CLIP features sit near chance on LAV-DF (0.536) rather than inverted, so the difference is something the P-001 visual path learned, not something any face model sees.

## Method notes

- Frames are converted back from ImageNet normalization to luminance before measuring.
- Blockiness is the mean jump across 8-pixel boundaries over the mean jump elsewhere, with 1e-6 added to both. The guard was added before any A13 result existed, after a unit test showed that a flat block interior otherwise read as "not blocky". The face crops are resized to 224 pixels, so the original 8x8 grid of the video is not aligned with the crop; the statistic was fixed in A13 as specified and is reported as such.
- `statistics.csv` holds every clip's six values and visual score. `receipts/a13.json` is the content-bound receipt.
