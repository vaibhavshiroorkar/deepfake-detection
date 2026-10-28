# Rehearsal script for the 14 main slides

Spoken length: about 40 seconds per slide, 9 to 10 minutes for the main slides. For each slide: the one sentence to say, the number to say aloud, and the question an examiner is most likely to ask, with the short answer. Backup slides B1 to B10 follow the divider (slide 15) and are for questions only.

| # | Slide | Say (one sentence) | Number to say aloud | Likely question and short answer |
|---|---|---|---|---|
| 1 | Title and authors | We built one local system that checks images, video and the speech in videos, and recorded every step. | Three authors, one guide. | What is the project in one sentence? A documented path from a simple first version to a frozen system, judged on data split by source identity. |
| 2 | Problem, scope, objectives | A fake can be a swapped face, a generated image or a cloned voice, and we claim only what the sealed tests support. | 0.9450 within the training domain against 0.7718 across datasets for a plain Xception in one published benchmark. | What do you not claim? No overall score, no audio-visual, pulse or generated-video detection, no detection of modern generated images, no Gemini capability. |
| 3 | Protocol and data | No person, speaker or source video is on both sides of a split, sealed sets are read once, and three tests were declared first. | 7,650 videos in 344 groups (44 identities and 300 single YouTube videos). | Why group-disjoint? A source video or speaker in two splits inflates AUC; detectors can learn identities. |
| 4 | Method: file to score | A file becomes a score through a frozen encoder, a linear head, a temperature and a threshold chosen on validation. | Intervals: 2,000 group-bootstrap resamples, 95 percent. | Is the score a probability? No: only for content like the validation split; otherwise it ranks files. |
| 5 | Working version, lessons | The first version showed two faults: video underfits and speech collapses outside its corpus. | Training AUC 0.8367; speech 0.9942 on validation and 0.6100 on the development slice. | Why frozen encoders first? Cheapest end-to-end version, and its diagnostics told us what to change. |
| 6 | Iteration ladder | A change entered only when it met its written success line. | Fine-tuning adds +0.1700 [+0.1228, +0.2114] over the frozen probe of the same encoder (exploratory). | What can 15 identities and 12 speakers show? They can rank candidates and reject a change; every verdict on them is exploratory. |
| 7 | Sealed tests T1 to T3 | Video and speech improved by the declared rule, and against the re-run Xception the video system is non-inferior. | 0.6238 to 0.8218 (+0.1980); 0.7516 to 0.9148 (+0.1633); margin 0.0100. | Is it better than the baseline? Not shown: the 95 percent superiority rule was met but does not survive Bonferroni, and the baseline ran at 224 px. |
| 8 | Speech: gain and shortcut | Part of the speech gain removes a silence and level shortcut, so it is not all better detection. | 0.061 of the 0.163 gain; a level cue remains (RMS AUC 0.135). | Did you fix a bug or build a better detector? Partly a removed shortcut; the test shows better ranking, not how much is better detection of synthesis. |
| 9 | Image analysis | The ProGAN-trained probe is fine on older generators and misses most images from recent ones. | Misses 0.7395 of images from 21 recent generators. | Why not retrain? A retrained head ranks better (0.8839) but flags 0.5213 of real images, so it is not installed. |
| 10 | Platform and its limits | Eight modules each report a finding with its scope and no overall score, and the tests after the freeze limit what each may claim. | FakeAVCeleb codec metadata alone gives 0.9985; at chance within LAV-DF produced videos (0.5032); DF26 0.4957. | Why no overall verdict? The modules examine different evidence with different calibrations; nothing measured supports a joint error model. |
| 11 | Dashboard | The page shows one card per check; the face-video flag reads Possible manipulation with high uncertainty. | 713 tests passed, 2 skipped (opt-in GPU tests), 13 deselected (data-dependent tests). | What does a flag mean? A possible sign, not a determination: the false-alarm rate differed between test sets. |
| 12 | Gemini-generated images | No Gemini data, model or experiment exists; this is an open question, not a capability. | Imagen 3, the nearest generator we measured: AUC 0.7691, missed 0.8719 (frozen head). | Can it detect Gemini images? Not tested; the terms, provenance and matched real controls are unresolved. |
| 13 | Comparison and limitations | Published numbers are context; thresholds do not transfer. | Speech EER 6.94 percent here against 1.13 percent quoted for AASIST (not re-run, no ranking); false alarms 0.1098 and 0.1405 (post hoc). | Are you comparable with published detectors? Only against the one baseline we re-ran, where we are non-inferior. |
| 14 | Conclusion and demo | The protocol worked: two confirmed improvements, one non-inferiority, and negative results kept with their reasons. | +0.1980, +0.1633, margin 0.0100. | What would you do next? Official Celeb-DF v2 test list beside the baseline, a larger encoder, a threshold calibrated on diverse sources. |

## Demo order (3 minutes)

1. Open the dashboard, upload an image: show the card, the label and the bar that says model output, not a probability.
2. Upload a video with sound: show the checks by outcome and that nothing says media is authentic.
3. Open the Evaluation Lab page: say it re-analyses stored scores, runs no model and cannot approve a detector.

## Three sentences to have ready

- Non-inferior, not superior: the declared 95 percent rule was met (+0.0176 [+0.0025, +0.0327]) but the 98.33 percent interval [-0.0011, +0.0355] includes zero.
- The speech gain is partly the removal of a silence and level shortcut: 0.061 of 0.163.
- Comparable means non-inferior against one baseline we re-ran with the same data, split and recipe.
