# Multimodal deepfake detection (final-year project)

Research prototype: image, video and audio-in-video analysis, with a modular platform that reports per-module findings, never one fused verdict. It is not a validated deepfake detector. Read the limitations of each module in its report entry before relying on any output.

- Setup, optional C2PA, required local artifacts and hashes, how to run the tests: [docs/SETUP.md](docs/SETUP.md)
- Running the demonstration (live dashboard, demo inputs, expected outputs): [docs/DEMO.md](docs/DEMO.md); engineering verification record with saved logs: [docs/TESTING_RECORD.md](docs/TESTING_RECORD.md)
- Project memory and rules: [CLAUDE.md](CLAUDE.md); current state and next step: [docs/STATUS.md](docs/STATUS.md)
- Experiments, decisions, data and licences: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md), [docs/DECISIONS.md](docs/DECISIONS.md), [docs/DATA_LOG.md](docs/DATA_LOG.md)
- What git history can and cannot show about experiment ordering: [docs/RESEARCH_INTEGRITY.md](docs/RESEARCH_INTEGRITY.md)

Final state (2026-10-29, tag `submission-final`): the submitted paper, report and deck are in [submission/](submission/README.md) and are built from the sources in `deliverables_v2/` (D-090). The folders `paper/`, `report/` and `slides/` hold the first versions, superseded and kept unchanged as an archive. The platform reports per-module findings only, with the labels of D-087. Experimental modules never give a definitive finding: the generated-image check is experimental (D-083) and the generated-video module is always inconclusive (its installed head did not generalise to DF26, AUC about 0.5). Post hoc false-alarm rates of the face-video detector on real videos are 0.1098 on the Celeb-DF family and 0.1405 on DFD (E-032). Scores are ranking scores, not probabilities that a file is fake. The lip-sync offset band is a provisional operating rule, not a detector; C2PA verification needs the optional `c2pa-python` package and an operator-configured trust anchor to ever report valid credentials.

Install: `uv sync` (Python 3.13); `uv sync --extra c2pa` adds C2PA verification. Large data and checkpoints are git-ignored; see docs/SETUP.md section 5. Tools that `uv sync` does not install (ffmpeg, pdftotext, the Playwright Chromium browser) are listed in docs/SETUP.md section 1a. To hide the GPU on Windows use `CUDA_VISIBLE_DEVICES=-1` (an empty value does not).
