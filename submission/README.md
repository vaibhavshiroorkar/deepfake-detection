# Submission package: Multimodal Deepfake Detection for Images, Video and Audio

Authors (in this order): Shivansh Gupta (16014023052), Vaibhav Shiroorkar (16014024805), Adithya Jithesh (16014024803). Guide: Dr. Tejashree Phatak. Department of Electronics and Computer Engineering, K. J. Somaiya School of Engineering, Somaiya Vidyavihar University, 2026-27.

## What is in this folder

| File | What it is |
|---|---|
| Multimodal_Deepfake_Detection_Research_Paper.pdf | The research paper, IEEE conference format with the college front matter (16 pages: 8 front matter, 8 from the title page to the end of the references) |
| Multimodal_Deepfake_Detection_Project_Report.pdf | The technical report (37 pages: 8 front matter and 29 body, references and Appendix A) |
| Multimodal_Deepfake_Detection_Presentation.pdf | The presentation: 14 main slides, a backup divider and 10 backup slides (25 in all); speaker notes are in the .pptx |
| sources/paper/ | LaTeX source of the paper with figures, bibliography, the college style and the class files (build: `latexmk -g -pdf -interaction=nonstopmode main.tex`, run twice) |
| sources/report/ | The report as a Word document |
| sources/slides/ | The presentation as a PowerPoint file, with speaker notes |
| records/ | Copies of DECISIONS.md, EXPERIMENTS.md, STATUS.md and PROTOCOL.md (the project record behind every number). In EXPERIMENTS.md the rows that name single files of Celeb-DF, Celeb-DF++ or FakeAVCeleb are replaced by a marker, because the terms of those data sets forbid distributing derived data |
| REHEARSAL.md | One sentence, one number and the likely question for each of the 14 main slides, and a three-minute demo order |
| MANIFEST.txt | SHA-256 of every file in this folder |

Not included, on purpose: any data set, frame, crop or per-video score of Celeb-DF, Celeb-DF++ or FakeAVCeleb, and any model weight (some weights are research-only). The software itself (the detectors, the platform and the website) is in the project repository, on the branch `main` after the merge (tag `submission-v2`) or on `trim/docs`.

## Start the website with the real checkpoint (from the repository root)

Needs the project environment (`uv sync`, see docs/SETUP.md) and the frozen checkpoint in `checkpoints/I-5-freeze-candidate` (not in this folder).

    PYTHONPATH=src .venv/Scripts/python.exe deliverables_v2/website/serve_v2.py 8012 --live

Open http://127.0.0.1:8012/ in a browser, choose an image, a video or an audio file and press Analyse. The Evaluation Lab is at http://127.0.0.1:8012/#lab. Without `--live` the page runs on invented demonstration data and says so in a banner. The original dashboard (`python -m dfd.platform.dashboard.serve 8002 --live`) is kept unchanged.

## Run the tests

    CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe -m pytest tests -p no:cacheprovider -rs -q

with the 13 data-dependent tests deselected (their ids are listed in docs/TESTING_RECORD.md section 2). Result recorded on 2026-10-29: 713 passed, 2 skipped (opt-in GPU tests), 13 deselected (data-dependent tests). The tests show software behaviour, not detection accuracy.

## Reproduce one table from the logged scores

The post hoc false-alarm rates (0.1098 on the 744 real videos of the Celeb-DF family, 0.1405 on the 363 real DFD videos) are recomputed from the stored sealed scores, with no model run:

    .venv/Scripts/python.exe scripts/eval/benchmark_audit_2026-10-27/frozen_rates.py I-4 ffpp:test_in celebdf_v2:test_x celebdf_family:test_x celebdf_pp_with_v2_real:test_x dfd:test_x asvspoof2019_la:test_in in_the_wild:test_x forensynths_progan_test:test_in forensynths_cnn_subset:test_x diffusion_test:test_x

It reads `data/_final/scores/I-4.csv` and `data/_final/eval/I-4.json` (local files; not distributed) and prints the rates. The replay of 2026-10-28 is saved in docs/test_evidence/2026-10-28_phase_b/. The Evaluation Lab page re-analyses the same score files with the project's metrics and group bootstrap and reproduces the logged AUC of 0.8218 for the Celeb-DF family; a final split is accepted only as a re-analysis of an evaluation already in the ledger.

## How to read the numbers

Sealed results were read once per frozen iteration; only three tests are confirmatory (T1, T2, T3). The final video system is non-inferior to the re-run Xception baseline at both levels (margin 0.0100); the declared 95 percent superiority rule was met but does not survive the Bonferroni correction (98.33 percent interval [-0.0011, +0.0355]); the baseline ran at 224 px. Part of the speech gain removes a silence and level shortcut. Everything else is exploratory, post hoc or context, and is labelled so in the documents.
