# P-001 amendment A12: does a fusion fit on other corpora transfer?

**Exploratory.** Fixed in [amendment A12](../preregistration/P-001-shortcut-neutral-av-study.md) after the H2 results were read and before this result existed. No P-001 verdict changes.

The A8 fusion trails single paths on LAV-DF and HiFi-AVDF. The paper guessed this was because its weights were fit on FakeAVCeleb. A12 refits the same fusion (`fit_fusion`, unchanged) in two ways, from the branch logits of [`results/p001-h2`](../p001-h2/README.md) (`scripts/run_a12.py`):

- **Leave one corpus out:** fit on the pooled clips of the other two targets, score the held-out one. A detector that never saw the target.
- **Upper bound:** 5-fold cross-fit by identity on the target itself. It uses the target's labels, so it is not a detector; it shows whether the three paths hold any complementary information there.

| Target | Clips | Visual | Audio | Sync | A8 fusion | Leave one out | Upper bound |
|---|---|---|---|---|---|---|---|
| DFDC | 2,786 | 0.702 | 0.532 | 0.569 | 0.706 | 0.477 | 0.706 |
| LAV-DF, window label | 3,686 | 0.396 | 0.762 | 0.649 | 0.566 | 0.690 | 0.796 |
| HiFi-AVDF | 3,485 | 0.545 | 0.698 | 0.569 | 0.601 | 0.573 | 0.701 |

ROC-AUC on the clip label, mean of three seeds. A reading aid: the deliverables take every number from `a12.json` through `results/numbers.json`, which also holds each paired difference with its identity-bootstrap interval.

## Reading, by the rules fixed in A12

- **DFDC and HiFi-AVDF: no complementary information.** The upper bound does not beat the best single path: +0.004 [-0.009, +0.016] over visual on DFDC, +0.003 [-0.002, +0.009] over audio on HiFi-AVDF. No weighting of these three paths helps there.
- **LAV-DF: a little, and it does not transfer.** The upper bound beats the audio path by +0.034 [+0.019, +0.049]. The leave-one-out fusion beats the A8 fusion by +0.123 [+0.105, +0.139], so weights from other corpora help, but it still trails the audio path by 0.073 [0.050, 0.097].
- **The best path changes with the corpus:** visual on DFDC, audio on LAV-DF and HiFi-AVDF. So weights fit on some corpora point the wrong way on the next. On DFDC the leave-one-out fusion falls below chance.

So the guess in the paper holds only in part. Fitting on FakeAVCeleb hurts on LAV-DF, but the larger problem is that no fixed fusion of these paths transfers between these corpora.

## Files

- `a12.json`: AUC of every score per seed, and each paired difference (mean over seeds, per seed, 95% identity-bootstrap interval, one-sided p).
- `receipts/a12.json`: the content-bound receipt of the run.
