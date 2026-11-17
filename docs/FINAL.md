# Shortcut-neutral audio-visual deepfake detection

The final project, in brief. Only what the project uses is here. The full record is in the [handbook](HANDBOOK.md). Why each choice was made is in the [project log](PROJECT-LOG.md).

Vaibhav Shiroorkar, Shivansh Gupta and Adithya Jithesh, with equal credit ([AUTHORS.md](../AUTHORS.md)). Supervised by Dr. Tejashree Phatak.

Each claim below carries a label:

- **Verified:** checked against a committed result file, named next to the claim.
- **Assumed:** believed but not tested; the reason, and what would test it, are given.

## The problem

Audio-visual deepfake detectors report 95 to 99 AUC on their benchmarks and fail elsewhere. We show that a large part of that benchmark accuracy comes from how the files were made, not from what they show. We build a protocol that removes those cues, then measure detectors on what is left.

## Contributions

1. **A shortcut census.** Verified: `results/shortcut-audit/`.
   - On FakeAVCeleb, file metadata and audio edges alone predict the label at 0.994 AUC. Fake audio is MP3 where real audio is AAC. Most face swaps are MPEG-4 where real clips are H.264. Generated speech starts and ends with digital silence.
   - LAV-DF reaches 0.936 from weak cues combined, and DFDC is close to clean at 0.588.
   - HiFi-AVDF, with six commercial generators, reaches 0.9985. The cues are frame size, fixed clip lengths (Sora 2 at exactly 10.0 s, Veo 3.1 at 8.0 s), bitrate, edge silence and sample rate.
2. **A shortcut-neutral protocol.** Verified: same folder.
   - One codec, a fixed 3.00 s span from the first sound, a fixed loudness, and dither.
   - Naive cleaning fails (0.933), because re-muxing exposes hidden AAC padding.
   - Our version lowers the ceiling to 0.632 on FakeAVCeleb, against a preregistered bar of 0.60, so H1 is not met. On the locked corpora it lowers HiFi-AVDF from 0.9985 to 0.629, LAV-DF from 0.936 to 0.557, and DFDC from 0.588 to 0.526.
3. **A lip-sync detector trained on real clips only.** It never sees a fake, so it cannot learn a generator's habits.
   - In-domain, its scores follow the mechanism: voice cloning is caught best, and face swaps that keep the original lips and voice least. Verified: `results/p001-sync/`.
   - Assumed: the ranking by method reflects lip-voice agreement rather than the leftover audio cue. The audio-label ceiling on the neutral copies is still 0.793, and it overlaps the voice-cloning row; an ablation that removes the audio residual would test this.
4. **False-alarm control from real clips only.** On new corpora, the threshold carried over from training flags 31 to 48% of real clips. A threshold set from 200 real target clips holds the rate near 5%. H4 is met on HiFi-AVDF and on LAV-DF under the file label, in all three seeds. Verified: `results/p001-h4/`.
5. **A preregistered fusion test (H2).** Visual, audio and sync paths, fused by a model fixed in advance (amendment A8), against the visual path alone. Holds on its rule; see the results below.

## Method

```text
clip -> neutralize (v4) -> 25 fps video, 16 kHz audio, 3.00 s
     -> mouth crops (MediaPipe, YuNet when the face is small) + audio
     -> frozen AV-HuBERT Base: lips-only and audio-only features, 2 windows of 2 s
     -> alignment head trained on real clips only (lips vs shifted or foreign audio)
     -> clip score = mean misfit over windows
     -> threshold: validation (source) or real-only conformal (target)
```

The visual path (EfficientNet-B0 with a GRU) and the audio path (Wav2Vec2 Base) train on neutral FakeAVCeleb, and a logistic fusion of all three paths is tested in H2.

Code:

- `src/deepfake_detection/preprocessing/neutralize.py`
- `src/deepfake_detection/avsync/`
- `src/deepfake_detection/evaluation/conformal.py`
- `src/deepfake_detection/fusion/h2.py`
- `scripts/train_sync_path.py`

## Data

| Corpus | Role | Clips used |
|---|---|---|
| FakeAVCeleb v1.2 | Training, validation, in-domain test | 20,284 after neutralization |
| LAV-DF | Locked cross-corpus test | 3,823 (1,000 original recordings with all their edits) |
| DFDC (subset) | Locked cross-corpus test | 3,032 |
| HiFi-AVDF | Locked test: Kling 2.5, PixVerse V5, Seedance 1.0, Sora 2, Veo 3.1, Wan 2.5 | 3,808 after neutralization (1,905 real, each paired with a generated copy) |

The FakeAVCeleb split is 70/15/15 by person. Results are also reported on an identity-strict test subset. Locked corpora never train, select or calibrate anything.

## Protocol

- Hypotheses are written and committed before the runs ([P-001](../results/preregistration/P-001-shortcut-neutral-av-study.md)). Changes are dated amendments, A1 to A10, and failed tests are reported.
- Three training seeds (17, 29, 43). 95% intervals come from 1,000 bootstrap resamples of people, not clips.
- Thresholds are chosen on validation, with a record of the clips that chose them.
- Every evaluation leaves a receipt: the command, and hashes of its inputs, code and outputs.

## Results

**Shortcut ceiling** (file and audio-edge cues only; higher means a weaker benchmark). Verified: `results/shortcut-audit/`.

| Corpus | Raw | Neutralized |
|---|---|---|
| FakeAVCeleb | 0.994 | 0.632 |
| LAV-DF | 0.936 | 0.557 (file label), 0.575 (window label) |
| DFDC | 0.588 | 0.526 |
| HiFi-AVDF | 0.9985 | 0.629 |

**Sync path, in-domain** (neutral FakeAVCeleb test, ROC-AUC). Verified: `results/p001-sync/`.

| Seed 17 | Seed 29 | Seed 43 |
|---|---|---|
| 0.775 [0.713, 0.832] | 0.809 [0.751, 0.861] | 0.806 [0.748, 0.859] |

By method (seed 29): voice cloning 0.942, Wav2Lip 0.84 to 0.86, face swap alone 0.63 to 0.68.

**Sync path, cross-corpus** (ROC-AUC, seeds 17, 29 and 43). Verified: `results/p001-sync-targets/`.

| Target | AUC by seed | Neutral ceiling |
|---|---|---|
| LAV-DF, file label | 0.597, 0.622, 0.622 | 0.557 |
| LAV-DF, window label (A9) | 0.634, 0.657, 0.655 | 0.575 |
| DFDC | 0.553, 0.575, 0.565 | 0.526 |
| HiFi-AVDF | 0.559, 0.570, 0.572 | 0.629 |

On HiFi-AVDF the sync path falls below the file-cue ceiling. By generator, PixVerse V5 is caught best, and Sora 2 falls below chance. Generators that make sound and picture together leave little lip mismatch.

**False alarms (H4)**, conformal threshold from 200 real target clips, bar FPR at most 0.07. Verified: `results/p001-h4/`.

| Target | FPR by seed | Met |
|---|---|---|
| HiFi-AVDF | 0.057, 0.055, 0.047 | 3 of 3 |
| LAV-DF, file label | 0.056, 0.069, 0.063 | 3 of 3 |
| LAV-DF, window label | 0.086, 0.056, 0.075 | 1 of 3 |

At these thresholds the sync path catches only 5 to 11% of fakes. DFDC is not run: it has 194 real clips, fewer than the 200 calibration needs.

**Fusion (H2)**, fusion minus visual-only AUC on the clip label, mean of three seeds, Holm over the first three rows. Verified: `results/p001-h2/`.

| Target | Difference [95%] | Visual alone | Fusion | Holm |
|---|---|---|---|---|
| FakeAVCeleb test, strict | +0.088 [+0.078, +0.100] | 0.90 | 0.98 to 0.99 | yes |
| DFDC | +0.003 [-0.008, +0.013] | 0.68 to 0.73 | 0.69 to 0.72 | no |
| LAV-DF, window label | +0.170 [+0.156, +0.183] | 0.37 to 0.41 | 0.50 to 0.62 | yes |
| HiFi-AVDF (reported) | +0.056 [+0.049, +0.063] | 0.54 to 0.55 | 0.59 to 0.61 | n/a |

H2 holds by its rule (two of three). The gain on FakeAVCeleb is the cloned-voice clips the visual path never trained on. On LAV-DF the visual path is below chance, and fusion trails the audio path (0.63 to 0.85) and the sync path (0.64 to 0.66) alone in every seed (verified). That its FakeAVCeleb-fit weights cause this is assumed; fitting the fusion on a held-out corpus would test it. HiFi-AVDF, reported outside the rule, repeats the LAV-DF pattern: fusion trails the audio path alone (0.68 to 0.71) in every seed (verified).

## Limitations

- **One training corpus.** FakeAVCeleb is from 2021, with about 500 real videos, and the sync path learned from about 350 speakers. The results show how a detector trained this way transfers, not that every detector fails.
- **H1 missed its preregistered bar.** The neutral ceiling is 0.632 against 0.60. Video bitrate remains, and it follows image content.
- **Cleaning refuses short clips:** 1,260 of them, all generated speech and none real.
- **Three protocol flaws were found after results were read, and fixed in the open:**
  - LAV-DF windows that missed the edit (A9);
  - small faces the landmarker missed (A10);
  - clips with less video than audio (handbook mistake 67).

  The results from before each fix are kept.
- **No baselines yet.** AVH-Align and HAVIC have not been run on our protocol, and H3 (which stream was faked) has not run.
- **Possible speaker overlap.** FakeAVCeleb, LAV-DF and AV-HuBERT's pretraining share speaker sources (VoxCeleb2, LRS3). Assumed to inflate cross-corpus sync scores slightly; a speaker-overlap check would test it.

## Reproduce

```bash
.venv/Scripts/python.exe scripts/shortcut_audit.py --manifest data/manifests/fakeavceleb.csv --dataset FakeAVCeleb --output results/shortcut-audit/fakeavceleb
.venv/Scripts/python.exe scripts/neutralize_dataset.py --manifest data/manifests/fakeavceleb.csv --dataset FakeAVCeleb --output-root data/neutral --manifest-out data/manifests/fakeavceleb-neutral.csv
.venv/Scripts/ddf.exe split build --manifest data/manifests/fakeavceleb-neutral.csv --output-dir runs/p001/split --dataset FakeAVCeleb --seed 17
.venv/Scripts/python.exe scripts/extract_avhubert_features.py --manifest data/manifests/fakeavceleb-neutral.csv --cuts data/manifests/fakeavceleb-neutral.cuts.csv --windows 2 --step 1 --output runs/avhubert-fakeavceleb-neutral
.venv/Scripts/python.exe scripts/train_sync_path.py --features runs/avhubert-fakeavceleb-neutral --split-dir runs/p001/split --output runs/p001/sync
```

The targets follow the same steps:

1. neutralize, with `--size` for HiFi-AVDF;
2. extract features with `--locator` (A10);
3. score with `scripts/score_sync_targets.py`;
4. run H4 with `scripts/run_h4.py`;
5. relabel LAV-DF with `scripts/lavdf_window_labels.py` (A9);
6. run H2 with `scripts/run_h2.py`.

Every number above has its files in `results/`.

## AI assistance

Claude (Anthropic) assisted with implementation, analysis and documentation. All design decisions, results and claims were reviewed by the authors.
