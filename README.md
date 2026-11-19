# Multimodal Audio-Visual Deepfake Detection

A final-year project that checks a talking-head video for signs of a deepfake. Three trained models look at it: one at the face, one at the voice, and one at whether the lips move with the words. The website gives a plain answer and shows each model's score along the video.

**Website:** https://vaibhavshiroorkar.github.io/deepfake-detection/ (the detector runs on our own PC, so it answers only while that PC is on).

By Vaibhav Shiroorkar, Adithya Jithesh and Shivansh Gupta, guided by Dr. Tejashree Phatak, K. J. Somaiya School of Engineering. Equal credit; see [AUTHORS.md](AUTHORS.md).

## What it does

1. **You give it a video**, a voice recording or a picture: upload a file, or record with your camera or microphone.
2. **It reads the file for AI labels.** Some generators write a tag saying the file is AI-made. If one is there, the answer is decided.
3. **It checks the whole video, 3 seconds at a time.** Each part is cleaned to one format first, so the models judge the content and not how the file was saved.
4. **Three models score each part:**
   - Face: EfficientNet-B0 with a GRU over 16 face crops.
   - Voice: AntiDeepfake XLS-R 1B, a published model trained on many speech datasets (the study's own voice model was Wav2Vec2 Base; see below).
   - Lip sync: AV-HuBERT features with a head trained on real clips only.
5. **You get an answer:** signs of manipulation, can't tell, or no signs found. The page also shows each model's score against the flag line, a timeline coloured part by part, and the three face frames the face model leaned on most, as heatmaps.

Scores run from 0 (looks real) through 0.5 (neutral) to 1 (looks fake). They are not probabilities: a score is placed by comparing the clip with thousands of real clips. The flag line (0.60) sits where about 5 in 100 real clips cross it.

## How well it works

On clips the models never trained on, with the website's own thresholds:

| Test set | Fakes caught | Real clips wrongly flagged |
|---|---|---|
| FakeAVCeleb (new speakers) | 75% | 0% |
| HiFi-AVDF (2025 generators) | 66% | 7% |
| LAV-DF (2022 word edits) | 8% | 7% |
| DFDC | 7% | 7% |
| All four together | 28% of 6,677 | 6% of 4,481 |

Since 19 November the website's voice model is AntiDeepfake XLS-R 1B instead of the study's Wav2Vec2 Base, which had heard only FakeAVCeleb. On fixed held-out sets it caught 96% of Veo 3 voices against 12% (`results/v2-eval/report.json`). It made HiFi-AVDF much better and LAV-DF worse, where the old model caught 37%. The face and lip-sync models are still the study's: no face model we tried works on 2026 video generators, so a fully generated video is mostly caught by its voice. `DDF_WEB_MODELS=p001` switches the website back to the study's models. Full numbers are in [docs/FINAL.md](docs/FINAL.md), and every number in the paper and report comes from [results/numbers.json](results/numbers.json).

## The research behind it

The study ([P-001](results/preregistration/P-001-shortcut-neutral-av-study.md)) found that public deepfake datasets leak their labels through file details. Codec, length and silence alone nearly separate real from fake on FakeAVCeleb. So every clip is cleaned to one format before any model sees it. The results:

- [File-detail audit](results/shortcut-audit/): what file details alone can predict, before and after cleaning.
- [Lip-sync model](results/p001-sync/README.md) and [how it transfers](results/p001-sync-targets/README.md).
- [False alarms](results/p001-h4/README.md): a threshold set from real clips of a new source keeps false alarms near 5%.

The [paper, report, slides and poster](deliverables/README.md) are built from these results.

## Run it

Install Python 3.11 to 3.13, FFmpeg and [uv](https://docs.astral.sh/uv/). Use `cu130` instead of `cpu` on an NVIDIA GPU.

```powershell
uv sync --extra cpu --extra ml --extra media --extra web --group dev
uv run ddf-web --preload
```

Open http://127.0.0.1:8000. The trained checkpoints and datasets are not in Git; [docs/DEMO.md](docs/DEMO.md) lists what the website needs and how to put it online through Tailscale Funnel. [docs/SECURITY.md](docs/SECURITY.md) covers what running it publicly exposes.

## Test it

```powershell
uv run pytest
```

FFmpeg-based tests skip when FFmpeg is missing. Before a push, `bash scripts/ci_local.sh` runs every CI step (lint, format, lock, docs, tests, browser test, smoke run). The `pre-push` hook in `.githooks/` runs it on every push once the clone uses it: `git config core.hooksPath .githooks`.

## More

- [Handbook](docs/HANDBOOK.md): the entry point for the whole project, including every mistake and how it was caught.
- [Research workflow](docs/workflow.md): commands for data, training, fusion and evaluation, plus the research dashboard.
- [Pending work](docs/PENDING.md) and [changelog](docs/CHANGELOG.md).
- [Decision log](docs/PROJECT-LOG.md), [data card](docs/data-card.md) and [model card](docs/model-card.md).
- [Licence and dataset terms](LICENSE.md). The datasets may not be redistributed.
- The October team project's history stays reachable at the tag `team/submission-final`, and its results are in [results/legacy-team/](results/legacy-team/README.md).
