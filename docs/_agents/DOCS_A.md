# DOCS_A: figures and paper (fork A, 2026-10-25)

## What changed

Figures (scripts/make_figures.py, assets/figures/):
- fig08_ladder_results (new, PNG and PDF): development-slice AUC with 95 percent intervals, video (I-0 probe, Xception baseline, I-1 fine-tuned SigLIP base) and speech (I-0, I-2, I-3, I-4). Generated through the key `fig10` of the script (the key `fig08` is the existing fig08_shortlist, whose name is unchanged).
- fig09_lavdf_cells (new, key `fig11`): LAV-DF validation, per kind of fake against the real originals and within the produced videos, visual and audio heads trained on LAV-DF (E-011 and its post-hoc control).
- fig03_ladder (same file name) is now a table of the measured ladder (about 7.0 x 3.9 in, text 9 pt) with verdict cells; it replaces the schematic of the planned ladder. fig04_staircase is unchanged (it is schematic and says nothing pending).
- The marker assets/figures/READY_fig08_fig09.txt exists; fig08_ladder_results was relabelled once (SigLIP base instead of an internal ID) after the first marker, the marker has a second line.
- The docstring of make_figures.py says which figures plot measurements.

Paper (paper/main.tex, paper/college-style.tex, paper/figures/*.pdf; 23 PDF pages, was 20):
- Abstract, introduction, method (audio trim and normalise), protocol (Xception baseline, delta rule 0.0143, encoder choice on validation only), section "Iterations" (was "Iteration Plan"): measured ladder figure, two tables (I-0 to I-3, I-4 to I-7) with hypotheses, thresholds and outcomes, mathematical statements of the levers, why aggregation and loss were not run, low-rank adapters named as not needed.
- Software: version-2 checkpoint paragraph and the parity check result (PASSED: 12 validation items, max difference 0.0019, tolerance 0.02), web page still on the working-version checkpoint.
- Results: scratch-reference sentence (T-A 0.22 to 0.29 per seed means a sign-flipped shortcut, with the trivial-feature AUCs), T-I wording (both CLIP models 1.0000 on validation, leader separated on dev_x on two generator units, no "saturated"), new subsections and paragraphs (iterations I-1 to I-4 with Table VII and Fig. 6; re-run baseline; LAV-DF with the D-059 wording, Table VIII and Fig. 7; pulse check), staircase text (the freeze is Thu 29 Oct 2026, final system definition, comparisons, delta), calibration after fine-tuning, limitations (12 speakers, baseline lr at the grid edge, one retried run, S-1 limit), conclusion.
- Only the sealed staircase table (4 rows x 2 cells) and one sentence each in the abstract and conclusion stay `[pending]` (9 placeholders, each with its E-000 comment).
- Contents page fits on one page again (college-style.tex: smaller toc font, section spacing 0.45 em; two subsections became run-in paragraphs).

Checks: latexmk clean (0 undefined references, 0 overfull), check_docs paper rows all PASS except number provenance (stale NUMBERS.json, rebuilt by the parent). Against a temporary rebuild of the numbers (not saved in docs/): 509 of 512 matched, the only number not found is 0.0143.

## For the parent: provenance

1. `0.0143` (delta) is not in any logged result file: it exists only in D-060, D-061 and DOC_FACTS 15. Either add it to a source that build_numbers.py reads (for example a line in E-013 or the baseline result block) or extend the builder; otherwise the paper, and probably the report and slides, fail the number rule.
2. Numbers added, with the experiment they come from (all copied from docs/EXPERIMENTS.md blocks or DOC_FACTS 15):
   - E-004 / I-0: 0.693 [0.664, 0.731], 0.610, 0.8367, 0.7985 (existing numbers reused in the new text).
   - E-005: validation 0.9869, 0.9893, 0.9918 (stage 1 at 1e-5); 0.5186 (SigLIP at 1e-4); 0.5583 (repeated run); learning-rate sensitivity 0.40 to 0.47; seeds 0.9893, 0.9896, 0.9908 and 0.9210, 0.9166, 0.8912; 0.9899 [0.9809, 0.9967]; 0.9096 [0.8776, 0.9388]; probe 0.8767 [0.8582, 0.8966] and 0.7396 [0.7031, 0.7832]; gains +0.1132 [+0.0948, +0.1299], +0.1700 [+0.1228, +0.2114]; training subset 0.9245 and 0.9994; gap 0.105 and 0.080; per method 1.000, 0.991, 0.999, 0.969; 0.852 and 0.921 (frame and clip); 0.0522, 0.0331, 0.0261, 0.0174; 1,750 steps.
   - E-007: 0.6723 [0.6126, 0.7584], +0.0623 [+0.0114, +0.1048], 0.9889 [0.9831, 0.9933], -0.0054, 0.135 [0.054, 0.382].
   - E-008: 0.7455 [0.7052, 0.8121], +0.0732 [+0.0122, +0.1225], 0.9869 [0.9818, 0.9911], -0.0020, 0.6569, 0.9436, 0.332, 0.043.
   - E-009: 0.8130 [0.7523, 0.8808], +0.0675 [+0.0300, +0.0929], 0.9964 [0.9943, 0.9982], members 0.7455, 0.7823, 0.7526, correlation 0.534, speeds 64 and 237 clips per second, ECE 0.0692, 0.1670, 0.0417, 0.0016.
   - E-003 / M-004 (scratch reference sentence): 0.375 [0.299, 0.453], 0.719 [0.425, 0.933], 0.594 [0.537, 0.649], 0.248 [0.073, 0.546], 0.22 to 0.29.
   - E-010: 0.3838 [0.3599, 0.4079], 0.6476 [0.6244, 0.6701].
   - E-011 and control: 0.7491 [0.7280, 0.7697], 0.8604 [0.8435, 0.8769], 0.9935 [0.9905, 0.9962], +0.0023 [+0.0009, +0.0040], 0.9912 [0.9864, 0.9951], 0.5070 [0.4771, 0.5361], 0.8574 [0.8394, 0.8752]; figure values 0.9905, 0.9921, 0.5175, 0.8711, 0.8556 with their intervals.
   - E-012: 0.5169 [0.4945, 0.5392].
   - E-013: 0.9872 [0.9784, 0.9942], 0.0010, 0.8750 [0.8435, 0.9089], 0.0072, +0.0027 [-0.0009, +0.0060], +0.0346 [+0.0136, +0.0550], +0.1817 [+0.1396, +0.2166]; delta 0.0143 (D-060, D-061).
   - E-006: 432 clips, 130, 0.301 [0.260, 0.346], 0.6, 1.118 dB, 0.280, 0.288, 0.333, 150 clips per dataset.
   - Parity of the version-2 checkpoint: 12 items, 0.0019, 0.02 (data/_results/freeze/parity.json, reported by the parent); 304 unit tests (INTEGRATION.md).

## Not filled / open

- The sealed-test staircase table and the two confirmatory comparisons: not run (freeze Thu 29 Oct 2026), `[pending]`.
- fig03_ladder is now a table-like figure; the report and the slides that embed it should expect the new aspect ratio (7.0 x about 3.9 in).
- The image-task T-I is identical in I-0 and I-4 (no comparison); the paper says so.
- The earlier figures (fig01, fig02, fig04) still use text below 9 pt at the printed size of the paper (they were made for the full page width and are shown at column width in places); the new figures follow the 9 pt rule (fig08_ladder_results and fig09_lavdf_cells are placed full width).
- A gap in the evidence worth the parent's decision: the sentence that most of the gain of the fine-tuning iteration over the working version comes from fine-tuning rests on one comparison (Xception minus the DINOv2 small probe); it is worded as a suggestion.
