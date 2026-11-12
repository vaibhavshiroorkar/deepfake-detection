# VIVA BENCHMARK BRIEFING (one page)

Written 2026-10-31. These are the points to say aloud. Each number is traced in docs/POST_FREEZE_BENCHMARK_COMPARISON.md and docs/EXPERIMENTS.md.

## What trained what

- **Face video (T-V):** trained on FF++ c23 only. Model selection used Celeb-DF v2 dev_x.
- **Speech (T-A):** trained on ASVspoof 2019 LA train. Model selection used In-the-Wild dev_x.
- **Generated image (T-I):** trained on ForenSynths ProGAN only.
- **Everything else was test data:** Celeb-DF v2 and ++, DFD, the ASVspoof eval set, In-the-Wild test, the CNN and diffusion image sets, FakeAVCeleb and LAV-DF. Each was read once and logged in the ledger.
- **Downloaded but not training:** a downloaded dataset is not training data. DF26, DeepAction and Community Forensics touched only experimental heads, which are not product heads.

## What is new (31 October)

- **Two post-freeze evaluations on data no script had read before.** Both were preregistered and hashed before scoring. The frozen checkpoint was not changed.
- **E-033 (LAV-DF):** 18,383 test clips. These exclude the 2,000 sealed clips and every clip that shares a source video with them.
- **E-034 (Community Forensics-Eval):** 58 unread shards; 7,209 images after removing exact duplicates.
- **Five predeclared groups each:** four nested sizes plus one diagnostic. They are offered in the redesigned Evaluation Lab as dataset, then group, then run.

## Headline numbers (AUROC, 95% group-bootstrap interval)

| What | Value | Say |
|---|---|---|
| Face video, Celeb-DF family (confirmatory T1, E-014) | 0.8218 [0.8020, 0.8393] | Our main cross-dataset result |
| Speech, In-the-Wild (T2) | 0.9148 [0.8896, 0.9383]; EER 15.78 percent (post hoc, E-030) | Lower error rate than published from-scratch models (RawNet2 33.9 percent) on the same corpus, higher than a frozen 2B XLS-R (7.2); Tier B, our test excludes 12 dev speakers |
| Face video, LAV-DF unread clips (E-033, G4) | 0.3803 [0.3748, 0.3860] | Below chance. Within produced clips only it is at chance (0.486), which points to the dataset pipeline rather than the edit. The 733 clips with several faces (largest face used) do not drive it: without them 0.3806 (post hoc) |
| Speech, LAV-DF unread clips (E-033, G4) | 0.6447 [0.6372, 0.6526] | Moderate; 0.733 when the edit falls inside the roughly 4 s the head hears |
| Generated image, CF-Eval unread shards (E-034, G4) | 0.7909 [0.7803, 0.8014]; misses 78 percent at the threshold | Same as on other shards (E-028). PNG only: 0.569. Say: "performance drops when format cannot separate the classes", an association, not proof that the head uses format |

## The three caveats to volunteer

1. **No comparison is Tier A (direct).** No paper tests on our exact items. Tier B means the same dataset with a different subset, metric or label. Tier C is context only. Never say "better than X". Say "on the same dataset, under a different protocol, our number is ...".
2. **The new results are replications on unread samples of distributions we had already seen.** They are not independent dataset-level tests and not confirmatory tests. The only confirmatory claims remain T1 to T3.
3. **Metrics are not interchangeable.**
   - AP depends on the class ratio. On the same E-034 scores, mean AP is 0.34 or 0.77 depending on how the cells are paired, so it cannot be set against the paper's 0.592.
   - Frame-level is not video-level.
   - Accuracy on a skewed set (FakeAVCeleb 90 percent fake) hides the missed class. That is why specificity and balanced accuracy are shown.

## Likely questions, short answers

- **"Why only these two new datasets?"** Every other set was already allocated to training, selection or a sealed read. Reusing any of them would mean re-reading a sealed split.
- **"Why five groups?"** To show how much a small sample can mislead. The speech G1 estimate (190 clips) is 0.69, while G4 (10,599 clips) is 0.645. The groups are nested views of one population, not five datasets.
- **"How do you know they are unseen?"**
  - A search of every split file, manifest, results folder and script.
  - An exact SHA-256 duplicate check: 0 duplicates for LAV-DF; 73 removed for CF-Eval, including 55 copies of E-028 images inside the dataset itself.
  - Near duplicates were not checked; we say so.
- **"Did you tune anything on these results?"** No. The thresholds are the checkpoint's, fixed on validation data in October. The protected files hash identically before and after.
- **"Is the face-video head bad, then?"** It is a face-swap detector trained on FF++. It generalises to Celeb-DF and DFD at about 0.82 to 0.90. It does not detect LAV-DF's word-level lip edits, and the product labels its output "Possible manipulation" with high uncertainty.
- **"Why no audio-visual score?"** Fusion was tested (E-010 to E-012). On LAV-DF it rested on a production cue, so it stays a planned milestone (D-059). The product shows separate audio and visual results.

## Unresolved, and what could not be run

- **Not run:**
  - The RAISE-paired Community Forensics generators (download cost).
  - The 4 undownloaded ForenSynths generators.
  - WildDeepfake, ASVspoof 2021 and AV-Deepfake1M (not on disk).
- **Blocked:**
  - Celeb-DF++ videos outside the test list (dev or sealed identities).
  - FakeAVCeleb's 11,490 unassigned fakes (identities straddle the halves).
  - Re-scoring the sealed LAV-DF subset with the product weights (a second sealed read).
- **Not yet cited:** the new papers are registered in docs/RESEARCH.md but not yet cited in the submitted paper or report. Adding them is a separate decision for the team.
