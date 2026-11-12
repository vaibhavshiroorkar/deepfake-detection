# Demo selection: how we chose the demo inputs, the sample groups and the decision lines

This note explains three choices behind our demonstration. It covers which files we upload on the Analyse page, how the evaluation groups in the Evaluation Lab were built, and where each detector's decision line comes from. It is written as the questions we expect to be asked, with our answers. Every number below is read from a file in this repository; the file is named next to it.

Rebuild commands:

```
PYTHONPATH=src .venv/Scripts/python.exe scripts/platform/make_demo_set.py          # demo inputs -> data/_demo_inputs/demo_set/
PYTHONPATH=src .venv/Scripts/python.exe scripts/platform/make_demo_inputs.py       # the original 8 demo files (docs/DEMO.md)
PYTHONPATH=src .venv/Scripts/python.exe scripts/posteval/build_sealed_groups.py    # E-014 re-analysis groups (D-096)
python scripts/posteval/build_manifests.py e033                                    # LAV-DF groups (E-033, D-094)
```

---

## Part A. Demo inputs (Analyse page)

### Q1. Which files do you upload during the demo?

We use the 23 files in `data/_demo_inputs/demo_set/`. They cover every kind of manipulation our detectors were built for, plus real files for each detector. We also keep the original eight files in `data/_demo_inputs/` (docs/DEMO.md). Those include a synthetic no-face clip and two error cases: an unsupported file type and a damaged video.

| Folder | File | Label | What it is | Source split |
|---|---|---|---|---|
| face_video | 01_real_original_ffpp.mp4 | real | unmanipulated YouTube video | FF++ val |
| face_video | 02_fake_deepfakes_faceswap_ffpp.mp4 | fake | face swap, autoencoder (Deepfakes) | FF++ val |
| face_video | 03_fake_faceswap_graphics_ffpp.mp4 | fake | face swap, graphics-based (FaceSwap) | FF++ val |
| face_video | 04_fake_face2face_reenactment_ffpp.mp4 | fake | expression reenactment (Face2Face) | FF++ val |
| face_video | 05_fake_neuraltextures_reenactment_ffpp.mp4 | fake | mouth reenactment, neural rendering (NeuralTextures) | FF++ val |
| face_video | 06_real_celebdf.mp4 | real | celebrity interview | Celeb-DF v2 dev slice |
| face_video | 07_fake_faceswap_celebdf.mp4 | fake | higher-quality face swap | Celeb-DF v2 dev slice |
| speech | 08_real_bonafide_asvspoof.flac | real | bona fide speech | ASVspoof 2019 LA dev |
| speech | 09_fake_tts_A01_asvspoof.flac | fake | text to speech (A01) | ASVspoof 2019 LA dev |
| speech | 10_fake_tts_A04_asvspoof.flac | fake | text to speech, waveform concatenation (A04) | ASVspoof 2019 LA dev |
| speech | 11_fake_vc_A05_asvspoof.flac | fake | voice conversion (A05) | ASVspoof 2019 LA dev |
| speech | 12_fake_vc_A06_asvspoof.flac | fake | voice conversion, spectral filtering (A06) | ASVspoof 2019 LA dev |
| speech | 13_real_inthewild.wav | real | public speech recording | In-the-Wild dev slice |
| speech | 14_fake_inthewild.wav | fake | speech deepfake found online | In-the-Wild dev slice |
| image | 15_real_progan_val.png | real | photograph | ForenSynths ProGAN val |
| image | 16_fake_progan_val.png | fake | ProGAN image (the training generator) | ForenSynths ProGAN val |
| image | 17_real_face_photo_cnn.jpeg | real | face photograph | ForenSynths CNN dev slice |
| image | 18_fake_stargan_cnn.png | fake | face attribute editing GAN (StarGAN), unseen generator | ForenSynths CNN dev slice |
| image | 19_fake_whichfaceisreal_cnn.jpeg | fake | generated face (whichfaceisreal), unseen generator | ForenSynths CNN dev slice |
| audio_visual | 20_real_lavdf_train.mp4 | real | talking head, nothing edited | LAV-DF official train, not used by any head |
| audio_visual | 21_fake_video_only_lavdf_train.mp4 | fake (video) | lips re-rendered, audio untouched | same |
| audio_visual | 22_fake_audio_only_lavdf_train.mp4 | fake (audio) | words replaced in the audio, video untouched | same |
| audio_visual | 23_fake_audio_and_video_lavdf_train.mp4 | fake (both) | audio and lips both edited | same |

`data/_demo_inputs/demo_set/manifest.json` records, for every file, its sample id, split, label, the size of the pool it was drawn from, the source path and its SHA-256.

### Q2. How did you pick these particular files? Did you choose ones the model gets right?

No. For each category we fixed the pool first: one non-sealed split, filtered to one manipulation type. We sorted the pool by sample id and drew one file with `random.Random(20261031 + category index)`. The script never loads a model and never reads a score, so it cannot prefer files the detector gets right. If the detector misses one of these files in the demo, that miss stays in the demo. It is part of what the demo shows.

### Q3. Why only validation and development files, and never test files?

Our test splits are sealed. They were read once, when the iteration was frozen (E-014). After that, any look at an individual test file risks informal tuning. The script resolves every pool through `loaders.select_records(final=False)`, which refuses the test splits (`test_in`, `test_x`). The face-video, speech and image pools are FF++ val, ASVspoof 2019 LA dev, ForenSynths ProGAN val, and the development slices of Celeb-DF v2, In-the-Wild and the CNN subset. For LAV-DF we take clips from the official train directory and exclude every clip used to train the I-6 heads (`lavdf_headtrain`). The product detectors were trained on FF++, ASVspoof 2019 LA and ProGAN only, so these LAV-DF clips are unseen by every detector that gives a verdict.

### Q4. The Lab shows BigGAN, CycleGAN, GauGAN, StyleGAN, StyleGAN2 and Deepfake. Why are none of those in the demo set?

All six exist only in the sealed CNN test split, so we cannot use them as demo files without opening sealed data. The development slice of the same collection holds two other generators the image detector never saw, StarGAN and whichfaceisreal. We use those instead. The six sealed generators appear only as stored scores in the Lab (Part B).

### Q5. Why does every detector get real files too?

A detector that flags everything would look perfect on fake files alone. A real file shows whether the detector stays quiet when nothing was manipulated. That's why each section has at least one real file: FF++ original and Celeb-real for face video, bona fide ASVspoof and In-the-Wild for speech, ProGAN and face photographs for images, and an unedited LAV-DF clip for the audio-visual case.

### Q6. Can these files be shown in a public video or put in the slides?

Not all of them. Celeb-DF v2 allows research use only and forbids publishing any part of its videos or derived data, so files 06 and 07 are for the live demo on our own machine only. LAV-DF is for non-commercial research and education, and so are the other sources (docs/DATA_LOG.md). The folder is git-ignored, and it is rebuilt by the script, not copied around.

---

## Part B. Sample groups (Evaluation Lab)

### Q7. What is an evaluation group?

An evaluation group is a fixed list of sample ids from a dataset whose scores are already stored. When someone picks a group and presses Run, the Lab recomputes the metrics from those stored scores: AUROC, precision, recall, specificity, F1, accuracy and EER, each with a group-bootstrap 95% interval. No model runs, nothing is re-scored, and no threshold changes.

### Q8. Which datasets have groups?

| Dataset in the Lab | Detector | Where the scores come from | Decision |
|---|---|---|---|
| LAV-DF test, unread part (T-A and T-V) | speech, face video | E-033: post-freeze scoring of previously unread clips | D-094 |
| Community Forensics-Eval, unread shards (T-I) | image | E-034: post-freeze scoring | D-094 |
| In-the-Wild, sealed test (T-A) | speech | E-014: the sealed evaluation at the freeze | D-096 |
| Celeb-DF v2, sealed test (T-V) | face video | E-014 | D-096 |
| ForenSynths CNN, 6 unseen generators, sealed test (T-I) | image | E-014 | D-096 |

### Q9. How were the five groups built?

The same way for every dataset:

- **G1, G2, G3 (Small, Medium, Large):** samples are shuffled once with seed 20261031, separately within each class, and each group takes the first k real and the first k fake. For In-the-Wild and CNN k is 100, 500 and 2,000; for Celeb-DF v2 it is 50, 200 and 600, because the sealed part has only 744 real videos. The groups are nested: each one contains every smaller one. LAV-DF is shuffled by source video instead (60, 300 and 1,200 source groups, with all fakes made from a source kept together), because its fakes share their source with a real original.
- **G4:** the whole set. For the sealed datasets G4 is exactly the block E-014 reported.
- **G5 (diagnostic):** a group built to test one specific worry about the detector (Q10).

| Dataset | G1 | G2 | G3 | G4 | G5 |
|---|---|---|---|---|---|
| In-the-Wild | 100 + 100 | 500 + 500 | 2,000 + 2,000 | 16,159 real, 8,629 fake, 42 speakers | 7,166 + 7,166 |
| Celeb-DF v2 | 50 + 50 | 200 + 200 | 600 + 600 | 744 real, 4,057 fake, 344 groups | 444 real, 4,057 fake |
| CNN generators | 100 + 100 | 500 + 500 | 2,000 + 2,000 | 23,807 real, 24,998 fake, 16 groups | 9,028 real, 9,019 fake |

(Counts are real + fake, from `data/_manifests/sealed_reanalysis/e014_sealed.groups.json`.)

### Q10. What does each diagnostic group (G5) test?

- **In-the-Wild, balanced within each speaker:** we keep every speaker who has both real and fake recordings, with equal numbers of each. Inside this group, knowing the speaker tells you nothing about the label. So it checks whether the speech detector separates real from fake speech of the same person, rather than one voice from another.
- **Celeb-DF v2, same-source reals only:** the YouTube-real videos come from a different source than the fakes. Leaving them out means reals and fakes both come from the same celebrity interviews, so a difference in source cannot do the work.
- **CNN, image-to-image generators only:** CycleGAN, GauGAN and Deepfake, each with its own real images. These generators start from a real input, so their fakes keep the layout of real photographs. This checks whether detection holds when the fake is built from a real image rather than generated from noise.
- **LAV-DF (T-V):** produced videos only, to see whether the face head reacts to the visual edit or to the dataset's production pipeline. **LAV-DF (T-A):** audio edits inside the 4.04 s the speech head actually scores.
- **Community Forensics:** PNG files only, matched by format, because file format alone separates that set (E-028).

### Q11. How do you know the groups were not chosen to make the results look good?

There are three safeguards:

1. **The score column is never read.** The builder reads only `sample_id`, `group_id` and `label` from the stored score file. It never reads the score column (`scripts/posteval/build_sealed_groups.py`).
2. **The rules are fixed in code.** The group sizes, the seed and the G5 rules are written in the code, and every rule depends only on labels and metadata.
3. **The groups are frozen.** The groups file is hashed, each Lab descriptor records that SHA-256, and the Lab refuses to run if the file has changed.

We kept the poor results too. The LAV-DF face-video groups, where the head scores below chance, are shown under limitations (D-095).

### Q12. Isn't re-running a sealed test set against your own rule?

We do not re-run anything on it. The sealed split was scored once, at the freeze, and those scores are logged in `data/_final/scores/I-4.csv`, whose SHA-256 is in the final ledger. The Lab only re-analyses those logged scores. Before every run it checks that the ledger has an entry for that iteration, dataset and split with the same file hash, and it refuses otherwise. Each result says so on screen: "Nothing was scored again now; ... This is not a new held-out test." These re-analyses are presentation views. They must not feed back into any model or threshold choice (D-096).

### Q13. How did you check the Lab gives the same answer as the paper?

We ran G4 of each sealed dataset and compared it with the E-014 report (`data/_final/eval/I-4.json`). All four logged metrics agree to four decimals:

| Dataset | AUROC | Balanced accuracy | Recall | Precision |
|---|---|---|---|---|
| In-the-Wild | 0.9148 | 0.8422 | 0.8560 | 0.7272 |
| Celeb-DF v2 | 0.8999 | 0.8037 | 0.7172 | 0.9727 |
| CNN generators | 0.9316 | 0.8102 | 0.6562 | 0.9514 |

The intervals differ slightly from E-014 because the bootstrap draws differ.

### Q14. Why do small groups give such wide intervals?

The bootstrap resamples whole independent groups (speakers, identities, generators), not single files. Celeb-DF G1 has 100 videos from 53 identity groups, so its AUROC interval is wide ([0.8329, 0.9474] in our browser check). CNN has only 16 generator groups, so even the whole set has a wide interval. We show the interval next to every number for this reason.

---

## Part C. Decision lines

### Q15. What is a decision line?

Each detector turns a file into a score between 0 and 1. The decision line is the fixed score above which the check reports "merits attention". Below it, the check reports that it found nothing within its tested scope. The line is never moved for a file, a dataset or a demo.

### Q16. How did you set the line for each detector?

All three model-based detectors use the same rule: **maximum balanced accuracy on the validation split of the training dataset (D-024)**. We pick the score that best separates real from fake on validation data, with each class counting equally, and freeze it. Before that, the scores are calibrated with one temperature, also fitted on validation. The values are read from `checkpoints/I-5-freeze-candidate/metadata.json` (product checkpoint, seed 0):

| Detector | Decision line | Fitted on |
|---|---|---|
| Face manipulation in video (T-V) | 0.6139 | FF++ validation, 700 clips |
| Synthetic speech (T-A) | 0.6954 | ASVspoof 2019 LA development, 24,844 clips |
| AI-generated image (T-I) | 0.2073 | ForenSynths ProGAN validation |

The Analyse page shows these rounded as "line 0.61" and "line 0.70".

### Q17. Why balanced accuracy and not plain accuracy?

Validation sets are not balanced, and nor are real-world inputs. With 15% real videos, a line that calls everything fake gets 85% accuracy and is useless. Balanced accuracy is the mean of the true positive rate and the true negative rate, so the line has to work for both classes.

### Q18. Why not choose the line that works best on the test data, or on each demo dataset?

That would be tuning on test data: the line would fit the data it is then judged on, and every number after that would be optimistic. Our rule is fixed on validation only (CLAUDE.md rule 3). On a new dataset the same line may be too high or too low. We accept that, and report it as a measured result. One example is the low recall on Celeb-DF G1: 0.66 at the validation line. AUROC does not depend on the line, so we show it alongside.

### Q19. Why do the sealed Lab groups show a different threshold from the product line?

For example, Celeb-DF v2 shows 0.5649, while the face-video product line is 0.6139. The E-014 evaluation used the three seed runs of the frozen iteration, and each seed has its own line fitted on validation by the same rule (D-024). The Lab re-analysis uses those same three per-seed lines, so it reproduces E-014 exactly. The value shown is their mean. The product uses seed 0 only, on temperature-calibrated scores, which gives 0.6139. The LAV-DF and Community Forensics groups use the product line directly.

### Q20. What about the audio-visual sync check and the AI-generated video check?

- **Lip sync** has no score line. It measures the audio-to-lip offset in frames. Offsets between -6 and +2 frames count as consistent. That band was fitted in E-025 on 230 real clips and checked on 230 different real clips, where 1.3% [0.4, 3.8] of real clips were flagged. Below a confidence of 3.0 the result is inconclusive. The band measures timing only: Wav2Lip fakes at -3 frames fall inside it, so it is not a fake detector.
- **AI-generated video** is experimental and gives no finding at all. On the DF26 benchmark it ranked generated and real videos no better than chance (E-020), and real clips also score high (E-026).

### Q21. Is there one overall fake or real verdict?

No. Each check answers for itself, inside its own tested scope. We do not fuse them into one score. In-domain fusion on LAV-DF learned a production cue (E-011, E-012), so fusion is not a product feature (D-059).
