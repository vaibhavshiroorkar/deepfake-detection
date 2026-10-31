# POST-FREEZE EVALUATION PROTOCOL: how the new samples were chosen and judged

Written 2026-10-31, after the preregistrations of E-033 and E-034 were hashed (01:20:02 +0530) and while scoring was running, before any metric of either experiment was computed or seen. Results are not in this file; they are appended to docs/EXPERIMENTS.md and summarised in docs/POST_FREEZE_BENCHMARK_COMPARISON.md. Decision: D-094. Evidence for every dataset role: docs/RESEARCH_EVALUATION_AUDIT.md.

This file answers the twelve questions a viva examiner is likely to ask about the new evaluations. Each answer describes what the scripts in scripts/posteval/ actually do.

## 1. Why these datasets?

Because they were the only data the frozen product checkpoint had never seen and that no earlier experiment had read, once every candidate was checked against the split files, the ledger and the experiment entries (audit section 3).

- **LAV-DF official test split (E-033).** 26,100 clips. 2,000 of them are the sealed subset of the freeze (ledger I-5 and I-6). 5,717 more share a source video with a sealed clip. That leaves 18,383 clips that no script has read (searched on 2026-10-31). LAV-DF has both faces and speech, so it measures two product heads, T-V (face video) and T-A (speech). Neither head was trained, tuned, calibrated or selected on LAV-DF.
- **Community Forensics-Eval, unread shards (E-034).** E-028 used 92 of the 413 shards. The other 321 had never been downloaded. T-I (the generated-image head) was trained only on ProGAN images. Of the unread shards, 202 contain no RAISE photographs (each about 20 MB). A seeded selection of 58 of those (5.19 GB) fits the download rate we measured, about 3.5 MB/s.
- **Every other dataset (FF++, Celeb-DF v2 and ++, DFD, ASVspoof 2019 LA, In-the-Wild, the ForenSynths sets, the diffusion sets, FakeAVCeleb) is fully allocated.** Each sample is either in training, validation, a development slice, or a sealed split that has already been read. Re-measuring them would mean either re-reading a sealed split or reusing development data, so they were not used.

## 2. Why these sample sizes?

The sizes were fixed in the preregistration, before any score existed. Three considerations set them:

- **Precision.** The uncertainty of an AUROC shrinks roughly with the square root of the number of independent units. Each step up in size multiplies the independent units by 2.5 to 5. The steps are 60, 300, 1,200 and 3,000 source groups for LAV-DF, and 100, 400 and 1,200 images per class for Community Forensics. This makes the narrowing of the intervals visible.
- **Floor.** The smallest group still has at least 80 clips (LAV-DF) or 100 images (CF) in each class, so precision, recall and specificity are defined in every bootstrap resample.
- **Cost.** Scoring is the expensive step: about 160 to 190 ms per clip, and CLIP ViT-L per image. So the whole eligible pool is scored once, and every group is a re-analysis of those stored scores. A larger group costs only bootstrap time.

## 3. How were the five groups constructed?

- **G1 to G4 (general groups) are nested.** Each contains every smaller one. This shows how estimates change with sample size on the same population.
  - LAV-DF: the order is a shuffle of the 5,170 source groups with `random.Random(20261031)`, after sorting by id. G1 to G4 take the first 60, 300, 1,200 and 3,000 source groups.
  - Community Forensics: one shuffled order per class, with the same seed. G1 to G3 take the first 100, 400 and 1,200 real and generated images. G4 is every downloaded image.
- **G5 (diagnostic) answers one predeclared question per head.**
  - T-V: the Large group without the real originals. Every remaining clip went through the dataset's production pipeline. Does the head react to the visual edit, or to the pipeline?
  - T-A: the Large group, keeping an audio fake only if its first edited segment starts inside the roughly 4 s window that the speech head scores. How much error comes from edits the head never hears?
  - T-I: every real PNG image plus an equal number of generated PNG images. E-028 showed that file format alone separates this evaluation set with AUC 0.93. Can the head separate them when format cannot?
- **FULL, the whole eligible pool (LAV-DF only),** is offered only as an explicit action. It is never the default.
- **The groups are not independent of each other,** and they are not five datasets. They are nested views of one population, so their intervals must not be combined or treated as replications.

## 4. How were real and fake samples selected?

- **No sample was chosen by its score or its content.**
- LAV-DF groups take whole source groups: a real original together with every fake made from it (on average 3.6 clips per group). Each group therefore contains both classes in the dataset's own proportions. The counts are recorded, not forced to balance: T-V G4 has 5,078 fake and 5,521 real.
- Community Forensics groups G1 to G3 are balanced by construction (an equal count per class). G4 keeps the natural composition of the downloaded shards (about 63 percent generated), and the class share is shown with every result.

## 5. How was identity and speaker leakage prevented?

- **Against the model:** no product head was trained or selected on LAV-DF or Community Forensics, so no identity in the new data can have been learned from these sets.
- **Against the sealed data:** the LAV-DF pool excludes every clip that shares a source video with the sealed subset. LAV-DF's official test identities are disjoint from its train and dev identities by the authors' rule.
- **Inside the evaluation:** the bootstrap resamples whole source groups, so a real clip and its fakes are always drawn together.
- **Limitation:** LAV-DF does not publish identity ids, so clips of the same person from different source videos may land in different groups. This makes the LAV-DF intervals somewhat too narrow, and it is stated with every result.
- **Community Forensics** has no person metadata. Images are the units, as in E-028.

## 6. How were duplicates checked?

- **Exact duplicates:** the SHA-256 of every file or image byte string, checked before any metric.
  - LAV-DF: against all 16,000 LAV-DF clips used before (the sealed subset in all three of its splits, and the I-6 training set).
  - Community Forensics: against the 11,550 E-028 images and the ForenSynths ProGAN validation and test images.
  - Any duplicate is excluded and listed in data/_results/posteval/<E>/duplicates.json.
- **Near duplicates** (re-encodes, crops) were not checked, because no perceptual hashing was run. This is stated as a limitation. In LAV-DF, near duplicates exist by construction (a fake is an edited copy of its original), which is why fakes and originals share a group.

## 7. How was unseen data told apart from previously exposed data?

A sample counts as "unread" only if it is absent from every split file, manifest, results folder and log that any script wrote. The search covered data/_results, data/_splits, data/_manifests, data/_logs, scripts and src on 2026-10-31. Absence from the training manifest alone was not enough.

The two pools are unread at the sample level, but not at the distribution level:

- The product heads' behaviour on LAV-DF was already seen on LAV-DF dev and the sealed subset, through the research weights.
- T-I was already measured on other shards of Community Forensics-Eval (E-028).

The new results are therefore labelled **replications on unread samples of a seen distribution**. They are not independent dataset-level generalisation tests, and they are not confirmatory tests. The only confirmatory tests of the project remain T1 to T3 (D-041, D-062 (b)).

## 8. Why these metrics?

- **AUROC (headline).** It measures how well a head ranks fakes above reals without choosing a threshold. That is the property that can transfer across datasets, and it is the metric most comparable to published work.
- **Threshold metrics: precision, recall (sensitivity), specificity, FPR, FNR, F1, balanced accuracy, accuracy.** These show what the product would actually do at its fixed threshold. The threshold was chosen on each head's own validation data (D-024) and is not tuned on these samples.
- **Balanced accuracy and specificity** are reported because the class shares differ between groups. Accuracy is always shown with the class balance, because a skewed set can make it look good while one class is missed.
- **EER** is given for comparison with speech papers.
- **Community Forensics:** the mean per-generator AP and accuracy are added as a secondary analysis, because that is what the Community Forensics paper reports.
- **What the scores are:** temperature-scaled scores, calibrated only on each head's own validation data. On these datasets they are ranking scores, not probabilities.

## 9. What do the confidence intervals mean?

- **Method:** each interval is a 95 percent percentile bootstrap. 2,000 resamples of whole groups (source videos for LAV-DF, images for Community Forensics), with seed 20261031, using the project's own cluster bootstrap through the Evaluation Lab.
- **What it covers:** how much the metric would move if we had drawn other groups from the same population. It does not cover the difference between this dataset and real-world content. It does not cover identity-level correlation that the grouping cannot see (LAV-DF). It does not cover the choice of threshold.
- **Small groups:** G1 intervals are wide on purpose. That is the point of showing the sizes side by side.
- **Nested groups:** because G1 to G4 are nested, their intervals are not independent evidence.

## 10. What are the limitations?

- **Seen distributions:** both pools come from distributions already seen at the distribution level (section 7).
- **LAV-DF specifics:**
  - The edits are word-level, inside talking-head clips made by one pipeline, so a clip labelled fake is mostly unedited.
  - The speech head hears only about 4 s.
  - T-V was trained on FF++ face swaps and reenactment, not on LAV-DF's lip and expression edits.
  - Identities are unknown.
- **Community Forensics specifics:**
  - Only 12 of 21 generators are covered (the RAISE-paired generators were left out because of download cost).
  - Format and size differ between real and generated images. G5 controls only for format.
  - Rows within a shard are probably correlated, which may make the image-level intervals optimistic.
- **Selection noise:** the T-V diagnostic and the T-A window rule are approximations. The T-A rule uses the first fake segment on the untrimmed track.
- **No perceptual near-duplicate check.**
- **One run each:** each experiment was run once with one frozen checkpoint, so there is no seed variation to report. The rule of at least 3 seeds applies to comparing candidates, and no candidate is compared here.

## 11. Why might our result differ from a published paper's result?

- **Training data and task.** The published LAV-DF and FakeAVCeleb detectors were trained on those datasets. Ours never saw them.
- **Split and test items.** Published LAV-DF numbers use the full test split; ours excludes 7,717 clips. Community Forensics pairs each generator with its own reals; our groups pool the reals.
- **Metric.** AP is not AUROC. Mean per-generator values are not pooled values. Frame-level is not video-level.
- **Label.** The LAV-DF paper labels a clip fake if any modality is edited. Our heads are judged against their own modality, and against the any-modality label only in a secondary analysis, with no fusion.
- **Preprocessing.** Face detector, number of frames, audio window, resizing.
- **Thresholds.** Published accuracies use thresholds that are often unstated. Ours is fixed from the checkpoint.

## 12. Why is the comparison direct in some cases but not others?

A comparison is direct (Tier A) only when the dataset version, the test items, the labels, the preprocessing and the metric all match. No published paper evaluates on our exact test items, so no comparison in this project is Tier A.

- **Tier B:** same dataset, but a different subset, split, metric or label. Examples: our In-the-Wild EER against Müller et al. and Pascu et al.; our mean per-generator AP on Community Forensics-Eval against the Community Forensics paper's Ojha row.
- **Tier C:** context only. Examples: LAV-DF detectors trained on LAV-DF; FakeAVCeleb results on a 70/30 split that is not stated to be identity-disjoint; frame-level results.

A Tier B or C row places our number next to a published number. It does not rank the two models.

## Addendum after the results (2026-10-31, continuation session)

Sections 1 to 12 above were written before any result and are not changed. Three facts that only became known from the runs:

- **Duplicates inside Community Forensics-Eval.** The set holds byte-identical images across shards. 55 E-034 images equal E-028 images, and 43 repeat other E-034 images. The 73 duplicates were excluded by the rule of section 6, leaving 7,209 images. E-028's own 11,550 rows hold 11,472 distinct images; E-028 is frozen and not recomputed.
- **Multiple faces in LAV-DF.** 733 clips carried the MULTIPLE_FACES warning, and the product scores the largest face. A post hoc, descriptive check found the T-V result does not depend on them: FULL AUROC 0.3800 with them, 0.3806 without.
- **AP and class ratio.** The preregistered per-generator AP (each generator against all reals) has cell prevalences of 2 to 22 percent, so it is not on the scale of the published mean AP. A post hoc paired version gives 0.77 for the same scores. AUROC is used for the comparison; both AP values are reported as computed.

