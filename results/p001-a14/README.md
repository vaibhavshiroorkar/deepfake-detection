# P-001 amendment A14: does the audio path rely on a leftover file cue?

**Exploratory.** Fixed in [amendment A14](../preregistration/P-001-shortcut-neutral-av-study.md) before this result existed. No P-001 verdict changes.

The P-001 audio path scores 0.952 to 1.000 on the FakeAVCeleb test, while a classifier on file details alone still reaches 0.793 on the audio label after cleaning. The shortcut audit names two remaining audio cues: loudness after AAC encoding, and small differences at the clip edges. A14 rescored the frozen P-001 audio models with both removed from the cached audio (`scripts/run_a14.py`):

- **loudness:** every clip's valid samples scaled to -23 dBFS RMS;
- **edges:** 250 ms cut from both ends, then zero-padded back, with the model told the new valid length (so the padding is not read as digital silence);
- **both** together; and **none**, the unperturbed audio through the same loop, as a check.

| Set | Seed | None | Loudness | Edges | Both |
|---|---|---|---|---|---|
| Identity-strict test (495 clips) | 17 / 29 / 43 | 1.000 / 0.952 / 1.000 | 1.000 / 0.946 / 1.000 | 1.000 / 0.933 / 1.000 | 1.000 / 0.940 / 1.000 |
| Full test (2,993 clips) | 17 / 29 / 43 | 1.000 / 0.964 / 1.000 | 1.000 / 0.959 / 1.000 | 1.000 / 0.949 / 1.000 | 1.000 / 0.949 / 1.000 |

ROC-AUC on the audio label; each metrics file holds its identity-bootstrap interval. "None" reproduces the official P-001 results exactly, so the loop scores as the original evaluation did. No clip was too short to cut. A reading aid: the deliverables take every number from the metrics files here through `results/numbers.json`.

## Reading, by the rule fixed in A14

- **The two named cues do not drive the score.** Seeds 17 and 43 stay at 1.000 under every perturbation. Seed 29 falls by at most 0.019, to 0.933, which is just under the 0.95 line in the rule but nowhere near the 0.793 ceiling.
- **Not covered:** cues the audit did not name. The audio path may still use something else in the file; A14 cannot rule that out.

## Files

- `audio-seed{S}-{part}-{kind}-predictions.csv` and `-metrics.json` per seed, set and perturbation; `summary.json` (clips too short to cut: none); `receipts/a14.json`.
