# P-001 amendment A16: does the sync score move when the audio is moved?

**Exploratory.** Fixed in [amendment A16](../preregistration/P-001-shortcut-neutral-av-study.md) before this result existed.

All 75 real clips of the neutral FakeAVCeleb test partition (75 people) had a usable 0 to 2 s window. Lips stay at 0 to 2 s; only the audio moves. Frozen AV-HuBERT Base and the three P-001 sync heads (`scripts/run_a16.py`).

| Audio | AUC, moved against original [95%] | Seeds 17, 29, 43 | Share of clips whose misfit rose |
|---|---|---|---|
| 0.04 s later (1 frame) | 0.559 [0.524, 0.600] | 0.556, 0.561, 0.562 | 0.72 to 0.79 |
| 0.08 s later (2 frames) | 0.666 [0.613, 0.719] | 0.679, 0.659, 0.661 | 0.80 |
| 0.20 s later | 0.899 [0.848, 0.947] | 0.898, 0.895, 0.905 | 0.93 to 0.95 |
| 0.40 s later | 0.983 [0.968, 0.994] | 0.977, 0.984, 0.986 | 1.00 |
| 0.80 s later | 0.984 [0.967, 0.995] | 0.977, 0.988, 0.987 | 0.95 to 0.96 |
| Another person's audio | 0.989 [0.978, 0.997] | 0.984, 0.990, 0.992 | 0.99 to 1.00 |

Intervals: mean of three seeds, 1,000 bootstrap resamples of people (seed 17). `windows.csv` holds every window's misfit per seed.

## Reading, by the rule fixed in A16

- **The score moves with the mechanism.** Every interval for the 0.20, 0.40 and 0.80 s shifts and the swap lies above 0.5, and the AUC rises from 0.559 at 0.04 s to 0.899 at 0.20 s.
- **It also sees shifts it was never trained on.** Training negatives were shifts of 5 to 20 frames and other clips' audio. Shifts of 1 and 2 frames (40 and 80 ms) still raise the misfit, with intervals above 0.5, though far less.
- **What this does not show.** The larger shifts and the swap resemble the training negatives, so they confirm the trained objective on people the head never saw. A16 does not test whether the voice-cloning row of the per-method table also carries the residual audio cue (paper, Table "per method"): a cloned voice is new audio, not moved audio.
