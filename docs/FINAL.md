# Shortcut-neutral audio-visual deepfake detection

The final project, in brief. Only what the project uses is here. The full record is in the [handbook](HANDBOOK.md). Why each choice was made is in the [project log](PROJECT-LOG.md).

Vaibhav Shiroorkar, Shivansh Gupta and Adithya Jithesh, with equal credit ([AUTHORS.md](../AUTHORS.md)). Supervised by Dr. Tejashree Phatak.

## The problem

Audio-visual deepfake detectors report 95 to 99 AUC on their benchmarks and fail elsewhere. We show that a large part of that benchmark accuracy comes from how the files were made, not from what they show. We build a protocol that removes those cues, then measure detectors on what is left.

## Contributions

1. **A shortcut census.** On FakeAVCeleb, file metadata and audio edges alone predict the label at 0.994 AUC:
   - fake audio is MP3 where real audio is AAC
   - most face swaps are MPEG-4 where real clips are H.264
   - generated speech starts and ends with digital silence

   LAV-DF reaches 0.936 from weak cues combined. DFDC is close to clean at 0.588. HiFi-AVDF, with six commercial generators, reaches 0.9985 from frame size, fixed clip lengths (Sora 2 at exactly 10.0 s, Veo 3.1 at 8.0 s), bitrate, edge silence and sample rate.
2. **A shortcut-neutral protocol.** One codec, a fixed 3.00 s span from the first sound, a fixed loudness, and dither. Naive cleaning fails (0.933), because re-muxing exposes hidden AAC padding. Our version lowers the ceiling to 0.632. We report that residual and the clips cleaning removes.
3. **A lip-sync detector trained on real clips only.** It never sees a fake, so it cannot learn a generator's habits. Its scores follow the mechanism: voice cloning is caught best, and face swaps that keep the original lips and voice least.
4. **False-alarm control from real clips only.** A threshold calibrated on real clips from the new domain, against the threshold carried over from training. *Pending.*

## Method

```text
clip -> neutralize (v4) -> 25 fps video, 16 kHz audio, 3.00 s
     -> mouth crops (MediaPipe) + audio
     -> frozen AV-HuBERT Base: lips-only and audio-only features, 2 windows of 2 s
     -> alignment head trained on real clips only (lips vs shifted or foreign audio)
     -> clip score = mean misfit over windows
     -> threshold: validation (source) or real-only conformal (target)
```

Code: `src/deepfake_detection/preprocessing/neutralize.py`, `src/deepfake_detection/avsync/`, `scripts/train_sync_path.py`, `src/deepfake_detection/evaluation/conformal.py`.

## Data

| Corpus | Role | Clips used |
|---|---|---|
| FakeAVCeleb v1.2 | Training (real clips only, for the sync path), validation, in-domain test | 20,284 after neutralization |
| LAV-DF | Locked cross-corpus test | 3,823 (1,000 original recordings with all their edits) |
| DFDC (subset) | Locked cross-corpus test | 3,032 |
| HiFi-AVDF | Locked test: Kling 2.5, PixVerse V5, Seedance 1.0, Sora 2, Veo 3.1, Wan 2.5 | 3,810 (1,905 real, each paired with a generated copy) |

The FakeAVCeleb split is 70/15/15 by person. Results are also reported on an identity-strict test subset. Locked test sets never train, select or calibrate anything.

## Protocol

- Hypotheses written and committed before the runs ([P-001](../results/preregistration/P-001-shortcut-neutral-av-study.md)). Failed tests are reported.
- Three training seeds (17, 29, 43). 95% intervals from 1,000 bootstrap resamples of people, not clips.
- Thresholds chosen on validation, with a record of the clips that chose them.
- Every evaluation leaves a receipt: the command, and hashes of inputs, code and outputs.

## Results

**Shortcut ceiling** (file and audio-edge cues only; higher means a worse benchmark)

| Corpus | Raw | Neutralized |
|---|---|---|
| FakeAVCeleb | 0.994 | 0.632 |
| LAV-DF | 0.936 | *pending* |
| DFDC | 0.588 | *pending* |
| HiFi-AVDF | 0.9985 | *pending* |

**Sync path, in-domain** (neutral FakeAVCeleb test, ROC-AUC)

| Seed 17 | Seed 29 | Seed 43 |
|---|---|---|
| 0.775 [0.713, 0.832] | 0.809 [0.751, 0.861] | 0.806 [0.748, 0.859] |

By method (seed 29): voice cloning 0.942, Wav2Lip 0.84 to 0.86, face swap alone 0.63 to 0.68.

**Sync path, cross-corpus:** *pending (LAV-DF, DFDC, HiFi-AVDF per generator).*

**False alarms (H4):** *pending.*

## Limitations

- **H1 missed its preregistered bar.** The neutral ceiling is 0.632 against 0.60. Video bitrate remains, and it follows image content.
- **Cleaning refuses short clips:** 1,260 of them, all generated speech and none real.
- **The multimodal fusion of visual, audio and sync paths** and the modality-prediction test are not finished in this version.
- **No baselines yet.** AVH-Align and HAVIC have not been run on our protocol.
- **Possible speaker overlap.** FakeAVCeleb, LAV-DF and AV-HuBERT's pretraining share speaker sources (VoxCeleb2, LRS3).
- **DFDC has 194 real clips,** fewer than the 200 the false-alarm test needs for calibration.

## Reproduce

```bash
.venv/Scripts/python.exe scripts/shortcut_audit.py --manifest data/manifests/fakeavceleb.csv --dataset FakeAVCeleb --output results/shortcut-audit/fakeavceleb
.venv/Scripts/python.exe scripts/neutralize_dataset.py --manifest data/manifests/fakeavceleb.csv --dataset FakeAVCeleb --output-root data/neutral --manifest-out data/manifests/fakeavceleb-neutral.csv
.venv/Scripts/ddf.exe split build --manifest data/manifests/fakeavceleb-neutral.csv --output-dir runs/p001/split --dataset FakeAVCeleb --seed 17
.venv/Scripts/python.exe scripts/extract_avhubert_features.py --manifest data/manifests/fakeavceleb-neutral.csv --cuts data/manifests/fakeavceleb-neutral.cuts.csv --windows 2 --step 1 --output runs/avhubert-fakeavceleb-neutral
.venv/Scripts/python.exe scripts/train_sync_path.py --features runs/avhubert-fakeavceleb-neutral --split-dir runs/p001/split --output runs/p001/sync
```

The cross-corpus and false-alarm steps are in `runs/p001-2day.sh`. Every number above has its files in `results/`.

## AI assistance

Claude (Anthropic) assisted with implementation, analysis and documentation. All design decisions, results and claims were reviewed by the authors.
