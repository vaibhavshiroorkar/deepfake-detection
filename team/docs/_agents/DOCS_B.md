# DOCS_B: report and slides updated with the results of I-1 to I-7 and the baseline (2026-10-25)

Written by fork B. Scope: report/** and slides/**. No git command was run. No GPU was used.

## What changed

Report (report/ch4_iter.py is new; ch4.py, ch4_results.py, ch5.py, front.py edited; built with `uv run --no-project --with python-docx --with lxml python report/build_report.py`; 80 PDF pages, 12 figures, 30 tables, 5 `[pending]` cells, layout test 26 of 26 rows within 2 mm):
- New section 4.8 "Results of the Iterations" with: the measured ladder table and figure (fig08_ladder_results), iteration 1 (grid table, probe against fine-tuned table, battery after), iterations 2 to 4 (speech table, members), iterations 5 and 6 on LAV-DF (pre-check, in-domain heads table, fig09_lavdf_cells, post-hoc control, produced-only head), iteration 7 (S-1 numbers), the Xception baseline and the margin, the final-system definition and the two confirmatory comparisons (in the staircase section), the software status paragraph, and a "Limits of these results" list.
- Replaced "planned" wording: chapter 4 box and summary, the pulse subsection (result added), the I-8 paragraph (not run), the staircase paragraph (baseline trained, delta stated), the abstract, the conclusions, the important points and the further work of chapter 5.
- Status table (table 28) now has 5 sealed-test rows, all `[pending]`; the freeze is stated as Thu 2026-10-29.
- Version-2 checkpoint, 304 tests and the passed GPU parity check (11 items, largest difference 0.0019) are stated, and that the page still runs the I-0 checkpoint.

Slides (slides/build_slides.py; 37 slides, 21 main slides plus references and 15 backup including the divider; notes on every slide; 7 `[pending]`; slide checker 614 of 614):
- Slide 11 Iteration ladder now shows the measured ladder (boxes coloured by verdict, fig08_ladder_results). The slide is no longer a mirror of fig03_ladder, so it was removed from the MIRROR note list.
- New slides: 13 "I-1: fine-tuning the face encoder" (with the Xception baseline), 14 "I-2 to I-4: three levers for speech", 15 "I-5 and I-6: LAV-DF" (fig09_lavdf_cells). New backups: 33 "Fine-tuning grid and checkpoint v2", 34 "LAV-DF control and subset".
- Updated: slide 1 notes, rPPG (outcome cell now the S-1 result), timeline note, limitations, conclusion (exploratory results, sealed cells pending), backup contents, staircase (margin, final system).

## Numbers added (all copied from the logs; experiment IDs are in comments and speaker notes only)

E-005 (grid, winner, probe comparison, battery), E-007 (speech trim and normalise), E-008 (augmentation), E-009 (three-encoder average, members, correlation, speeds), E-010 (frozen heads on LAV-DF), E-011 (in-domain heads, fusion, weights, ECE not shown) and its post-hoc control, E-012 (produced-only head), E-013 (Xception baseline, seed sd 0.0010 and 0.0072), E-006 (S-1: 130 of 432, 0.301 [0.260, 0.346], per dataset, 1.118 dB, median SNR -0.222 dB). The margin is 0.0143 (corrected by the parent from 0.0144; text says it is computed from the unrounded standard deviations).

## Checks

- `python scripts/check_docs.py`: 71 pass, 0 fail (after the parent's NUMBERS.json rebuild). Before that, with a scratch rebuild of the numbers, the only unmatched number was the margin.
- report/layout_check.py 26 of 26; slides/check_slides.py 614 of 614, build with no overflow warnings.
- Rendered and looked at: report pages 56 to 70 and 1 to 8 (tables, figures, captions), slides 11, 13, 14, 15, 16, 19, 20, 24, 33, 34. A duplicate "(exploratory)" in two figure captions was found and removed.

## Not filled or open

- Sealed-test cells (staircase, two confirmatory comparisons, LAV-DF sealed test) stay `[pending]` by instruction. In the report they cite experiment label "E-014 (freeze)" in their Word comments; no such entry exists yet, so change the label if the freeze uses another ID.
- The figure fig08_ladder_results (owned by fork A) has small text at slide size (about 9 to 10 pt on slide 11, with the figure at 8 of 12 columns); it is readable but is the smallest text of the deck. A larger-font version of the figure would help.
- The timeline figure fig06_timeline (report chapter 1, slide 18) is the original plan; slide 18 carries a status line, the report figure does not.
- Parity item count: the parent message said 12 items; data/_results/freeze/parity.json has 11 (6 FF++ videos, 3 ASVspoof files, 2 ProGAN images). The documents say 11.
- The slides preview PDF (slides/DeepFake_Detection_Presentation_preview.pdf) was refreshed through PowerPoint; the report preview PDF is refreshed by the build.
- The independent verification by a fresh agent has not been run by this fork.
