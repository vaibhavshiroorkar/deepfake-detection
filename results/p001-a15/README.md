# P-001 amendment A15: H3 in one direction, FakeAVCeleb to LAV-DF

**Exploratory.** Fixed in [amendment A15](../preregistration/P-001-shortcut-neutral-av-study.md) before this result existed. H3 needs both directions, and the reverse (LAV-DF to FakeAVCeleb) needs models trained on LAV-DF, so **H3 stays not run** as preregistered.

The two binary heads are the P-001 visual path (video label) and audio path (audio label), each at its FakeAVCeleb validation threshold. The baseline marks both streams fake whenever the clip is fake, using the true clip label. `scripts/run_a15.py` writes `a15.json`; the deliverables take their numbers from it through `results/numbers.json`.

| LAV-DF label | Clips | Macro-F1, heads (seeds 17, 29, 43) | Macro-F1, baseline | Heads minus baseline [95%] |
|---|---|---|---|---|
| Window (A9), primary | 3,815 | 0.200, 0.353, 0.249 | 0.800 | -0.532 [-0.549, -0.516] |
| File | 3,815 | 0.185, 0.403, 0.282 | 0.799 | -0.509 [-0.523, -0.496] |

Intervals: 1,000 bootstrap resamples of the 999 LAV-DF original recordings (seed 17). The one-sided share at or below zero is 1.000 under both labels.

## Reading, by the rule fixed in A15

- **The direction does not transfer.** The heads do far worse than a predictor that only knows the clip label.
- **The video head is worse than guessing.** Its precision on the video label is 0.19 to 0.24, below the share of clips whose video was edited. This matches the inverted visual path on LAV-DF (A13).
- **The audio head ranks but its threshold does not carry over.** At seeds 17 and 43 it is right 86 to 92% of the times it says "audio fake", but it catches only 9 to 11% of audio fakes. The same audio path reaches an AUC of 0.632 to 0.848 on LAV-DF (H2), so the loss is in the FakeAVCeleb threshold, as with the sync path's threshold in H4.

Per-stream F1, precision and recall per seed are in `a15.json`.
