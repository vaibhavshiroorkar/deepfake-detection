# FREEZE scoring harness (written 2026-10-25 by the freeze fork)

Purpose: produce, at the freeze, one score file per iteration of the staircase (PROTOCOL.md addendum of D-041) and evaluate each once with
`python -m dfd.eval.run --final --iteration <tag>`. Nothing here has read a sealed split. Everything below was tested only on val, dev_x and the
LAV-DF fusion-fit split, except where "untested" is stated.

## Files (all new, under scripts/freeze/ unless stated)

| File | What it does |
|---|---|
| common.py | datasets and split files, record selection behind the final-only guard, feature and score caches, frozen-feature and fine-tuned scorers |
| prepare_heads.py | CPU. Retrains and saves the heads of I-0 (T-V IMG-07, T-I IMG-13, T-A AU-8), I-2 and I-3 (AU-8) with the logged recipe and seeds 0, 1, 2 and checks them against the logged results (done, see below) |
| prepare_crops.py | CPU. Face crops for the sealed test videos (Celeb-DF v2 test_x, DFD, Celeb-DF++, LAV-DF test_in). Data preparation only |
| score_final.py | the entry script: `--iteration <tags|all>` with exactly one of `--final` and `--dry-run` (`--limit N` for dry runs) |
| run_all.sh | the whole chain (heads, crops, every tag, the check), `--final` or `--dry-run`, logs in data/_logs/freeze_*.log, marker data/_final/DONE |
| check_score_files.py | validates every score file with dfd.eval.scorefile.read_scores, counts per (iteration, modality, dataset, split), seeds, sample sets, parity with logged scores |
| run_eval.sh | runs the final evaluations of all score files (written, not run) |
| dev_prime_lavdf_cache.py | test helper only (dry root): fills the LAV-DF score cache from the E-010 to E-012 files so the fusion code can be tested on the CPU |
| docs/_agents/FREEZE.md | this file |

One edit outside the owned files, needed for the Xception baseline: `src/dfd/train/finetune.py` now also saves `buffers` (BatchNorm running statistics)
in `weights.pt`. Before, only parameters were saved, so a reloaded Xception would have used the ImageNet statistics instead of the fine-tuned ones and its
scores would not have matched E-013. For the ViT encoders (E-005, E-011, E-012) the edit changes nothing. The edit was made before the baseline run started
(the baseline began after E-012 finished); `common.load_finetuned` refuses an Xception weights file without `buffers`. Check once that
`checkpoints/I-1/IMG-04_lr*_s0/weights.pt` has the key `buffers` after the first baseline run finishes.

## Iterations and score files

Every score file has exactly the columns run_id, iteration, seed, modality, dataset, split, sample_id, group_id, label, score (raw probabilities, clip level
for videos as the mean of frame probabilities per D-024, image level for images), seeds 0, 1, 2, run_id = `<tag>-<modality>-s<seed>`.

| Tag | Configuration | Modalities | Datasets (manifest keys) and splits | Source experiment |
|---|---|---|---|---|
| I-0 | T-V IMG-07 frozen + head; T-I IMG-13 + head; T-A AU-8 first-4-s window + head | visual, image, audio | ffpp val, test_in; celebdf_v2 dev_x, test_x; dfd test_x; celebdf_pp test_x (+ derived below); forensynths_progan_val val; forensynths_progan_test test_in; forensynths_cnn_subset dev_x, test_x; diffusion_test test_x; asvspoof2019_la val, test_in; in_the_wild dev_x, test_x | E-004 |
| I-1 | T-V IMG-17 fine-tuned (lr 1e-5, seeds 0 to 2); T-I, T-A as I-0 | same | same | E-005 |
| I-2 | T-A AU-8 on the trim and peak-normalise window; T-V, T-I as I-1 | same | same | E-007 |
| I-3 | T-A head trained with three augmented copies; rest as I-1 | same | same | E-008 |
| I-4 | T-A mean logit of AU-8, AU-4, AU-6; rest as I-1 | same | same | E-009 |
| BASE | T-V Xception (IMG-04) fine-tuned, learning rate from data/_results/baseline/selection.json | visual | T-V datasets only | E-013, D-060 |
| I-5, I-5o, I-5p | LAV-DF subset, frozen heads of E-005 (visual) and E-009 (audio) | visual, audio | lavdf (fused label) / lavdf_videolabel, lavdf_audiolabel / lavdf_produced_videolabel, lavdf_produced_audiolabel; splits val, test_in | E-010 |
| I-6, I-6o, I-6p | LAV-DF in-domain heads (E-011) and the logistic-regression fusion | visual, audio, fused (fused only in I-6) | same | E-011 |
| I-6b, I-6bo, I-6bp | LAV-DF visual head trained on produced videos only (E-012) | visual | same | E-012 |
| I-7 | no model (S-1 failed, D-052) | none | nothing to score | E-006 |

Why the suffixes o and p: `dfd.eval.run` fits the threshold and temperature of a run on all val rows of that run_id. LAV-DF has three label schemes (fused label;
own modality label; own label inside the produced videos), so one run_id would mix labels in one fit. Each scheme is its own iteration tag. For the guard the tag is the
iteration column of the file, so each tag is evaluated with `--iteration <tag>`.

Derived datasets (T-V, test_x only), made from the Celeb-DF v2 and Celeb-DF++ rows (D-045 items 3 and 4): `celebdf_pp_with_v2_real` = the 2,849 Celeb-DF++ fakes plus the 744 Celeb-DF v2 reals,
and `celebdf_family` = Celeb-DF v2 test_x plus Celeb-DF++ test_x (744 real, 6,906 fake); the group id is the identity (`celebdf:idN`) in both. The raw `celebdf_pp` block
(fakes only, one class) is scored but not written as its own dataset, because `dfd.eval.run` cannot compute an AUC on one class. The split file used for Celeb-DF++ is
`celebdf_pp-3f9df077b9eb` (2,849 fake, 0 real, 44 groups); the other file, `celebdf_pp-497a9caa5fcc`, counts the 150 real videos twice and is not used.

Seeds: the heads and fine-tuned models of seeds 0, 1, 2 are scored separately; the evaluation averages them as in PROTOCOL section 6. The I-0 checkpoint of the web page ships seed 0.

## Order of work at the freeze

1. `CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=src .venv/Scripts/python.exe scripts/freeze/prepare_heads.py` (CPU; already run on 2026-10-25, the heads are in data/_final/heads and reproduce the logged results exactly, see "Tested").
2. `PYTHONPATH=src .venv/Scripts/python.exe scripts/freeze/prepare_crops.py --workers 8` (CPU; was started on 2026-10-25 in the background, log data/_logs/freeze_crops.log; resumable).
3. `bash scripts/freeze/run_all.sh --final` (GPU, refuses if the GPU holds more than 2000 MiB, needs the baseline to be DONE for BASE; resumable; ends with the check).
4. `PYTHONPATH=src .venv/Scripts/python.exe scripts/freeze/check_score_files.py` must print "all checks passed" (items missing because a clip has no face or no crop are listed as MISSING and count as a problem: look at the provenance JSON, decide, and write the decision down before evaluating).
5. `bash scripts/freeze/run_eval.sh` (the commands below).

Exact evaluation commands (one per tag; the guard accepts a first final evaluation of an iteration and dataset, refuses a second without `--repeat "<reason>"`, and the ledger is written only after success):

```
PYTHONPATH=src .venv/Scripts/python.exe -m dfd.eval.run --scores data/_final/scores/I-0.csv --final --iteration I-0 --out data/_final/eval/I-0.json
... the same for I-1, I-2, I-3, I-4, BASE, I-5, I-5o, I-5p, I-6, I-6o, I-6p, I-6b, I-6bo, I-6bp
```

The two confirmatory comparisons (PROTOCOL addendum) use `--compare` on these files after the regular evaluations are in the ledger, for example the final system against I-0 on visual,
celebdf_family, test_x:

```
PYTHONPATH=src .venv/Scripts/python.exe -m dfd.eval.run --compare --scores data/_final/scores/I-4.csv --scores-b data/_final/scores/I-0.csv --modality visual --dataset celebdf_family --split test_x --metric auc --final --iteration I-4 --iteration-b I-0
```
and against the baseline with `--scores-b data/_final/scores/BASE.csv --iteration-b BASE` (delta of D-025 from the baseline seed spread, rule of D-060).
Which iteration is "the final system" is a decision for the freeze: T-V comes from I-1 on, T-A from I-4, the audio-visual fusion from I-6 (see open items).

## Tested (CPU, 2026-10-25, val and dev_x and LAV-DF train and val only)

- prepare_heads.py ran to the end. The I-0 heads reproduce the logged scores in data/_results/I-0/<task>/scores.npz with maximum absolute difference 0.0 (T-V, T-I, T-A, validation and dev_x clip scores); the I-2 and I-3 heads reproduce the AUC estimates of data/_results/I-2/ha1.json and I-3/ha2.json exactly (validation 0.9889 and 0.9869, dev_x 0.6723 and 0.7455).
- score_final.py --dry-run for I-0 to I-4, I-5, I-6, I-6b wrote valid files that `dfd.eval.scorefile.read_scores` and `dfd.eval.run` (non-final, `--splits val,dev_x`) accept. The I-0 file reproduces the logged validation scores to 2e-7. The LAV-DF files evaluated through `dfd.eval.run` give the E-010 and E-011 numbers (I-6 fused AUC 0.992 with the per-seed fusion against 0.9935 in E-011, which averages the seeds before fusing; I-6o visual 0.748 and audio 0.860; I-6p visual 0.507 and audio 0.857; I-5 visual 0.291 and audio 0.636).
- check_score_files.py passes on those files; `bash scripts/freeze/run_all.sh --dry-run` ran end to end with the GPU hidden (`CUDA_VISIBLE_DEVICES=-1`), skipping the GPU blocks with a note.
- The fine-tuned scoring path (`common.load_finetuned`, `ft_scores`, IMG-17 weights of E-005) ran on 300 FF++ val and 300 Celeb-DF dev_x clips per seed and reproduces the stored E-005 validation scores to 4.4e-4 (fp16 inference is not bit-exact; the checker's tolerance is 1e-3). See the incident note below: this ran on the GPU by mistake.
- The sealed-split guards: `--final` and `--dry-run` are mutually exclusive and one is required; `--limit` is refused with `--final`; `common.select` raises for a sealed split unless final=True and `L.select_records` refuses it a second time.

## Untested (written by careful reading only)

- The extraction branches for the sealed sets: `video_features`, `image_features`, `audio_features` when no sweep feature file exists (all test sets), and the `--final` path as a whole.
- `prepare_crops.py` was started (Celeb-DF v2 test_x is processed; DFD runs at 1.9 videos per second, about 31 minutes; Celeb-DF++ and LAV-DF test_in follow) and has not been checked for completeness.
- BASE (the baseline weights did not exist yet) and the buffers path of `load_finetuned`.
- The LAV-DF test_in path (audio features of the test videos through the encoders, `ft_scores` on test_in) and the I-6 fusion on real GPU scores (the fusion code ran on primed scores of E-010 and E-011).
- `run_eval.sh` (written, not run: it would read sealed rows).

## Estimated GPU time at the freeze (from the sidecar throughputs; add model loading of a minute or two per encoder)

| Stage | Items | Speed used | Time |
|---|---|---|---|
| I-0 T-V IMG-07 frozen features | 11,781 test clips x 16 frames = 188k frames | 680 to 740 frames/s | about 5 min |
| I-0 T-I IMG-13 | 66,805 test images | 65 to 110 images/s | 10 to 17 min |
| I-0 T-A AU-8, first window | 71,237 + 24,788 test clips | 240 to 360 clips/s plus first-time decoding into the cache | 5 to 7 min GPU, plus 5 to 10 min decoding |
| I-1 T-V IMG-17 fine-tuned, 3 seeds | about 227k frames per seed (val, dev_x, test) | about 420 frames/s (E-005 evaluation pass) | about 9 min per seed, 27 min |
| I-2 T-A AU-8 trim_norm | 96k clips | as above | 5 to 7 min plus decoding |
| I-4 T-A AU-4 and AU-6 | 96k clips each | 210 to 280 and 64 to 70 clips/s | 6 to 8 min and 23 to 25 min |
| BASE T-V Xception, 3 seeds | as I-1 | not measured, faster than IMG-17 | about 20 to 25 min |
| I-5, I-6, I-6b (LAV-DF test_in 2,000 clips) | 32k frames per model, 9 models, 3 audio encoders | | about 15 min |
| Total | | | about 2 to 2.5 hours, plus about 40 minutes of CPU crops (DFD is the slow one) |

## Open items (not decided by this fork)

1. Which iteration tag is "the final system" for the two confirmatory comparisons, and whether the fused audio-visual score of I-6 may be part of it: the E-011 control (D-059) shows the in-domain visual head is at chance inside the produced videos, so the I-6 fused AUC on the fused label is dominated by a pipeline cue. This fork scores it as specified and flags it; the claim is for the freeze decision.
2. For I-5 and I-6 the temperature and threshold of the evaluation are fitted on LAV-DF val (that is what `dfd.eval.run` does with the val rows of the run). E-010 reported both the source-domain and the LAV-DF temperature; only the LAV-DF-fitted one appears in the final evaluation of these tags.
3. The fusion rule written for I-6 is the logistic regression only (the E-011 choice on val); the sum rule is not scored. If the sum rule is wanted it needs its own tag (two rules in one tag would be read as six seeds of one candidate).
4. I-6 fusion is fitted per seed on the fusion-fit subset (temperatures and weights per seed, then the fused probabilities are averaged by the evaluation), whereas E-011 averaged the log-odds over seeds first; the two differ slightly (0.992 against 0.9935 on val). Stated here so that the final number is not compared with E-011's without this remark.
5. Dataset key naming: the sweep sidecars call Celeb-DF dev_x "celebdf"; the score files use the manifest key `celebdf_v2`.
6. Celeb-DF++ clips without a finished crop or a face frame are dropped from the score file and listed as MISSING by the checker; nothing in the harness imputes a score.
7. Ledger: each tag and each dataset in it gets its own ledger entry (so LAV-DF has I-5, I-5o, I-5p, I-6, ...); the derived datasets `celebdf_family` and `celebdf_pp_with_v2_real` are ledger entries too, and they contain rows that also appear in `celebdf_v2` and `celebdf_pp` (a pooled view, not extra data).
8. If the E-013 baseline learning rate is not the one of the first 1e-5 and the baseline directory is renamed, `score_final.py` reads data/_results/baseline/selection.json at run time and will follow it.

## Incident (report, rule 9)

During the first CPU tests I set `CUDA_VISIBLE_DEVICES=` (empty) to keep the run off the GPU. On this Windows shell an empty value does not hide the GPU (`CUDA_VISIBLE_DEVICES=-1` does). The dry-run of I-1 to I-4 therefore scored 300 FF++ val and 300 Celeb-DF dev_x clips (16 frames each) per seed with the three IMG-17 models on the GPU for about two minutes while the parent's E-012 job was running (peak VRAM seen afterwards 5.6 GiB in total, no error in either job). No result of E-012 is affected apart from its wall time. All later CPU tests used `CUDA_VISIBLE_DEVICES=-1`, and the instructions in this file use that value.
