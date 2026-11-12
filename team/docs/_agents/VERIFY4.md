# VERIFY4: independent check of paper, report and slides

Verifier: independent agent, 2026-10-25. Nothing in the repository was edited except this file. Copies and renders are under the scratchpad directory `verify4` (`C:\Users\admin\AppData\Local\Temp\claude\d--Final-Year-Project\e19a6579-de4f-4bd2-ac56-c467c128c7bc\scratchpad\verify4\`).

## What was checked

- paper/main.pdf: 26 pages, every page rendered (70 dpi contact sheets, 100 dpi zoom on pages with figures, tables and dense text).
- report: the preview PDF (89 pages, same timestamp as the docx) was re-exported from a COPY of the docx with Word (`word_tools.ps1 -Mode pdf`): 89 pages, byte size within 9 bytes of the preview, so the preview is current. Every page rendered.
- slides: a COPY of the pptx was exported with PowerPoint COM: 39 slides, identical size to the committed preview PDF (current). Every slide rendered. Font sizes read with python-pptx: every text run, table cell and bullet is 12 pt or larger; the charts are embedded pictures.
- Extracted text (pdftotext, UTF-8) of all three, plus the raw XML of the docx and pptx (comments, speaker notes, metadata, hidden text, tracked changes: none hidden, no tracked changes, no hidden slides).

## Result in one line

The rendered pages are clean (no clipped text, no figure cut off, no stray markup, no E-/D-/M-/R- style IDs, no reviewer or agent words, identity fields match docs/IDENTITY.md field by field, all required numbers match docs/EXPERIMENTS.md). The blocking problems are hidden text inside the Office files: 76 Word comments and 39 sets of speaker notes full of internal IDs.

## Defect table

| # | document | page or slide | defect | severity |
|---|---|---|---|---|
| 1 | report .docx | whole file (76 comments anchored through the text; none appear in the PDF export) | The docx carries 76 Word comments by author "Project team", for example "DECISIONS D-001, D-002, D-011, D-018, D-019; RESEARCH R-001 to R-003, R-017, R-033", "CLAUDE.md rules 1 and 8; DOC_FACTS; STATUS", "CLAUDE.md section 2". They show in Word's All Markup view and are internal IDs and repository file names. Strip all comments (or deliver the PDF only) before submission. | blocking |
| 2 | slides .pptx | notes of all 39 slides (not visible on rendered slides) | Speaker notes contain about 226 internal IDs (D-032, D-062 c, E-004, E-001, M-004, M-005, R-/O-010) and file paths or file names (`data/_results/I-0/battery.md`, `PLAN.md`, `ITERATIONS.md`, `PROTOCOL.md section 1`, `data/_results/T-V/model_comparison.md`) on 38 of 39 notes. They print with notes pages and show in presenter view. Strip or rewrite before submission. | blocking |
| 3 | paper vs report | paper p16 (Limitations, PDF 24) vs report p65 (PDF 78, 4.10.1) | Contradiction on the baseline: paper says "two runs were repeated after data-loader errors" for the Xception baseline; the report says "one run was repeated after a data-loader error" for the same baseline. EXPERIMENTS.md E-013b logs one baseline run repeated (seed 2); the other repeated run (DINOv2 base at 1e-4) belongs to the I-1 grid, not the baseline. Make the paper say one, or say which two. | medium |
| 4 | report | p19 (PDF 32), after Eq. (26) | Broken inline fraction: "balanced prior pi = 12" is rendered as a small "12" over a bar with no denominator; it must read pi = 1/2. Confirmed at 220 dpi. | medium |
| 5 | slides | slide 11 (iteration ladder chart) | Chart text (axis ticks, category labels, value labels) measured at about 10 pt, below the 12 pt limit for chart text. Slide 16 chart text is about 12 pt (borderline, passes). | medium |
| 6 | paper | p1 (Abstract and Index Terms), p2 (PDF 10) "[31]-[33]", p17-18 reference [13] "------," | Dash characters remain in the PDF text: em dash in "Abstract---" and "Index Terms---" (IEEEtran class), en dash in the citation range "[31]-[33]", double em dash as the repeated-author mark in reference [13]. CLAUDE.md rule 11 forbids em-dashes in any document. Change the class setting or the bibliography style, or accept as IEEE style and record it. The report and the slides have no em or en dash (only the mathematical minus sign U+2212, which is not a dash). | medium |
| 7 | report | p63 (PDF 76, 4.10) | Stale wording: "The confirmatory claims are two comparisons, declared before any final run" and then "three tests in all". The text is internally consistent (two comparisons, one of them on two sealed sets) but the paper, the slides and the amended E-014 say "three confirmatory tests". Use the same words in all three documents. No place says two tests, delta 0.0143 as the current margin, or a baseline rate of 1e-4 as the current baseline. | low |
| 8 | report | p5 (PDF 6, page v) | Page holds only the "Keywords" line and is otherwise empty (orphan from the Abstract page). | low |
| 9 | report | p32 (PDF 45), p49 (PDF 62), p55 (PDF 68), p42 (PDF 55) | Large white gaps before tables or figures that moved to the next page: about half a page on p32 before Table 10, about 40 percent on p49 after Table 21, about half on p55 after Table 26, about a quarter on p42. Table 18 (p42) is slightly wider than the text block on the left. Not wrong, but visible. | low |
| 10 | report | p57 (PDF 70-71, Table 27) | Text says the control was run at eight amplitudes from 0.0005 to 0.032; Table 27 shows seven rows (the 0.0005 row is omitted). Slide 18 omits both 0.0005 and 0.001. The omitted rows are in EXPERIMENTS.md (M-005), the headline numbers are right. | low |
| 11 | paper, report, slides | paper Fig. 6 (p14); report Fig. 7 (p50) and Fig. 5 (p45); slide 11 | Rounding inconsistency: the I-1b development AUC (0.9315, raw 0.931514) is labelled 0.931 in paper Fig. 6, report Fig. 7 and slide 11, and 0.932 in paper Fig. 4 and report Fig. 5. Correct three-decimal rounding of 0.931514 is 0.932. | low |
| 12 | paper | p4 (PDF 12) | Orphan heading: "IV. EXPERIMENTAL PROTOCOL" sits at the bottom of the left column and its first subsection "A. Setup" starts the right column. | low |
| 13 | paper | p1 (PDF 9) | Abstract is very long (about 350 words); IEEE guidance is about 150 to 250. | low |
| 14 | paper | p18 (PDF 26) | Last page holds five reference lines and is otherwise empty. Also p6 and p14 have visible vertical gaps. | low |
| 15 | slides | slides 31, 33, 37, 39 | Lower 40 to 50 percent of the slide is empty. | low |
| 16 | slides | slides 15 and 32 | Numbers on the development slice (slide 15 speech table, slide 32 battery) carry no "exploratory" label on the slide itself, unlike slides 9 to 14 and 31. Slide 15 has "Sealed tests: [pending]" but no exploratory word. | low |
| 17 | slides | slide 19 | The three screenshots are cropped through a line of text at the top and their text is unreadable at slide size (the paper says the same for its Fig. 5 and labels it). | low |
| 18 | slides vs paper | slide 22 vs paper abstract and report abstract | Slide 22 quotes "face development AUC 0.7396 to 0.9096 by fine-tuning" (same encoder, probe vs fine-tuned); the paper and report abstracts quote 0.693 to 0.910 (working version vs fine-tuned SigLIP). Both are correct and exploratory; the different baselines are not stated on the slide. | low |
| 19 | paper, report | S-1, I-7 labels | Internal-looking labels "S-1" and "I-7" appear in the text (paper p10, p12; report 4.7.2, 4.8.5). They are defined in the text and are iteration or check names, not repository IDs; no E-, D-, M-, R- ID appears. | low |
| 20 | paper source | paper/main.tex | LaTeX comments carry E-/R- IDs and CLAUDE.md (for example lines 8, 133, 593, 659). Invisible in the PDF; visible to anyone who gets the source (Overleaf). | low |
| 21 | slides | slide 32 | Unspaced dataset identifier "whichfaceisreal" (twice); the paper and report write "Which-Face-Is-Real". | low |

Counts: blocking 2, medium 4, low 15.

## Item 1: render checks (no defect found unless listed above)

- Text clipped, overlapping, touching the footer rule or the page border: none found in the paper (26 pages), the report (89 pages) or the slides (39).
- Figures cut off: none. Figures 1 to 3 of the paper carry a small grey in-figure caption line above the figure body; intended.
- Tables off the page: none (paper Table VI and report Table 18 are the widest, both inside the border).
- Empty pages: none fully empty. Nearly empty: report p5 (page v, Keywords only), report p66 (end of chapter 4, 8 lines), paper p18.
- Equations: all rendered correctly in the report (checked Eq. 8 to 12, 24 to 26, 39 to 42 at 90 dpi) except the inline fraction in item 4. Paper equations (1) to (5) correct.
- Captions: every figure and table in the three documents has a caption; list of figures and list of tables in the paper (7 figures, 9 tables) and the report (12 figures, 32 tables) match the pages (spot checked 10 entries each).
- Stray markup (backslashes, braces, "??"): none.
- [pending] cells, counted from the extracted text: paper 14 (13 cells in Table IX on p16 plus one in the Conclusion sentence on p17); report 7 (all in Table 30, p65); slides 8 (slide 11, 13, 14, 15, 19, 22 twice, 31). They are placeholders for sealed results, none looks like an error.

## Item 2: identity fields (docs/IDENTITY.md), all match

Paper cover block and certificate/declaration pages (PDF 1 to 3, 9), report cover, certificate, declaration and acknowledgement (PDF 1 to 4), slide 1 and the docx/pptx core properties were compared field by field:

- University "Somaiya Vidyavihar University", school "K. J. Somaiya School of Engineering", department "Electronics and Computer Engineering", degree wording "Bachelor of Technology (Electronics and Computer Engineering)", cover wording ("Submitted In Partial Fulfillment of Requirements / For the Degree Of / ... / By"), address "Vidyavihar, Mumbai - 400 077", year "2026-2027": match.
- Students in the order Vaibhav Shiroorkar 16014024805, Shivansh Gupta 16014023052, Adithya Jithesh 16014024803: match in the paper (3 occurrences each), the report (3 each) and slide 1 (1 each, report order).
- Guide Dr. Tejashree Phatak, Assistant Professor; Head Dr. Jagannath Nirmal; Examiner and Director as labels only; "Mumbai-77", Date blank; "LY B. Tech, Semester VII"; footer "Department of Electronics and Computer Engineering Semester VII 2026-27 Batch Page n": match.
- Slide footer reads "Electronics and Computer Engineering" (the shortened department name allowed by IDENTITY.md); the old "LY Project" line is absent. No Electronics and Telecommunication anywhere.

## Item 3: internal IDs and phrases

- Extracted text of the three PDFs: no E-xxx, D-xxx, M-xxx, R-xxx IDs (including E-013b forms), no "reviewer", "verifier", "fork", "agent", "autonomously", "needs user review". Only "S-1" and "I-7" labels (item 19). "project team" appears in the report and slides as a neutral phrase.
- Hidden text: see items 1, 2 and 20.
- Dashes: see item 6.

## Item 4: numbers (all confirmed against docs/EXPERIMENTS.md E-013b, E-016, E-017, M-005 result blocks and NUMBERS.json)

| fact | value in EXPERIMENTS.md | paper | report | slides |
|---|---|---|---|---|
| baseline at 3e-4: validation | 0.9912 [0.9849, 0.9963] | p13 text and Table VII | Table 28 (p59) | slide 13 |
| baseline at 3e-4: dev_x | 0.8716 [0.8364, 0.9076] | p13, Table VII | Table 28, p67 conclusion | slide 13 |
| delta | 0.0100 (floor of the D-060 rule) | p6, p13, p16 | p59, p64 | slides 13, 21, 22, 26 |
| first pass at 1e-4 kept as history | 0.9872 and 0.8750, delta 0.0143 | p6 and p13 ("first pass ... delta = 0.0143", extended once to 3e-4) | Table 28, p59, p64 | not stated on slides (notes only); no slide treats it as current |
| I-1b | validation 0.9705, dev_x 0.9315, differences -0.0194 and +0.0219, does not enter | Table III/VII, p13, p17 | Table 24 (p52), Table 21 | slide 14 |
| I-1c | validation 0.9903, dev_x 0.9192, differences +0.0004 and +0.0096, does not enter | same places | Table 24, p53 | slide 14 |
| M-005 | 0.008 primary (377/432 = 0.873), 0.004 green only (341/432 = 0.789), at 0.004 primary 0.546, control 0.301 | p15 (VII-D) | Table 27, p57-58 | slide 18 |
| S-1 | 130/432 = 0.301 against 0.6 | p15, abstract | p47 to p57 | slides 17, 18 |
| I-1 | validation 0.9899, dev_x 0.9096 | all | all | slides 13, 22 |
| three confirmatory tests, 98.33 percent robustness line | E-014 amendment | Fig. 3, p5, p15, Table IX, p16 | p63 to p64, Table 30 | slide 26 |

Contradictions searched and not found: a freeze date (only the planned bars "Integration and freeze" on Thu 29 in report Fig. 1 and slide 20, plus "Saturday 24 to Friday 30 October" in the notes), a baseline learning rate of 1e-4 or delta 0.0143 presented as current, I-1b or I-1c entering the final system (all three say "does not enter" and the final system is the D-061 definition), results claimed from sealed data, "state of the art", "real-time" (only inside a quoted study of DeepFaceLive fakes, report p10), "better than" or "outperforms" about this system. The one contradiction found is item 3 (one versus two repeated runs). Wording drift only: item 7.

## Item 5: claims discipline

- Nothing marked [pending] is also given as a result. The conclusions of the paper (p17), the report (p67, p68) and slide 22 state that the sealed-test results and the three confirmatory tests are not yet run.
- Exploratory labels: the paper abstract, Table V to VII captions, the report abstract and tables, and slides 9 to 14, 16, 18, 22, 31, 34 to 36 carry "exploratory"; gaps on slides 15 and 32 are in item 16.
- Limits of M-005 are present in all three: paper p15 ("does not show that a real pulse exists ..., what its amplitude or form is, that it survives H.264 compression (the sinusoid was added after decoding), or anything about forgeries; a constant frequency and amplitude is also easier"), the Limitations of paper p16, report p58 and 4.10.1 (channel weights chosen by the implementer, no source on typical amplitudes read), and slide 18 ("It does not show": real pulse, H.264, forged videos).
- The only comparisons with a published detector are labelled exploratory differences to the re-run Xception baseline, with no sealed evidence; no "comparable" claim is made.
