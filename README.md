# Multimodal deepfake detection (final-year project)

Research prototype: image, video and audio-in-video analysis, with a modular platform that reports per-module findings, never one fused verdict. It is not a validated deepfake detector. Read the limitations of each module in its report entry before relying on any output.

- Setup, optional C2PA, required local artifacts and hashes, how to run the tests: [docs/SETUP.md](docs/SETUP.md)
- Running the demonstration (live dashboard, demo inputs, expected outputs): [docs/DEMO.md](docs/DEMO.md); engineering verification record with saved logs: [docs/TESTING_RECORD.md](docs/TESTING_RECORD.md)
- Project memory and rules: [CLAUDE.md](CLAUDE.md); current state and next step: [docs/STATUS.md](docs/STATUS.md)
- Experiments, decisions, data and licences: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md), [docs/DECISIONS.md](docs/DECISIONS.md), [docs/DATA_LOG.md](docs/DATA_LOG.md)
- What git history can and cannot show about experiment ordering: [docs/RESEARCH_INTEGRITY.md](docs/RESEARCH_INTEGRITY.md)

Status of the platform modules (2026-10-26): the generated-video module is experimental and always inconclusive (its installed head did not generalise to DF26, AUC about 0.5); the lip-sync offset band is a provisional operating rule, not a detector; C2PA verification needs the optional `c2pa-python` package and an operator-configured trust anchor to ever report valid credentials.

Install: `uv sync` (Python 3.13); `uv sync --extra c2pa` adds C2PA verification. Large data and checkpoints are git-ignored; see docs/SETUP.md section 5. Tools that `uv sync` does not install (ffmpeg, pdftotext, the Playwright Chromium browser) are listed in docs/SETUP.md section 1a. To hide the GPU on Windows use `CUDA_VISIBLE_DEVICES=-1` (an empty value does not).
