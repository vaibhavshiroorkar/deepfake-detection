# VERIFY5: independent check of paper, report and slides after the sealed freeze

Verifier: independent agent, 2026-10-25. Nothing in the repository was edited except this file. Copies, exports and renders are under the scratchpad `C:\Users\admin\AppData\Local\Temp\claude\d--Final-Year-Project\e19a6579-de4f-4bd2-ac56-c467c128c7bc\scratchpad\verify5\`.

## What was checked

- paper/main.pdf: 27 pages, every page rendered (70 dpi contact sheets of 6, 130 dpi zoom on the abstract page, the sealed-results pages and the tables).
- report: a COPY of the docx was exported with Word (`word_tools.ps1 -Mode pdf`): 93 pages, and the extracted text is byte-identical to report/DeepFake_Detection_Technical_Report_preview.pdf (93 pages). The preview is current. Every page rendered (contact sheets of 6) and the sealed-results pages (65 to 69, 72) at 100 to 110 dpi.
- slides: a COPY of the pptx was exported with PowerPoint COM: 42 slides; the extracted text is identical to slides/DeepFake_Detection_Presentation_preview.pdf (current). Every slide rendered (contact sheets of 6, zoom on 11, 19, 20, 21, 24). Font sizes read with python-pptx: every text run and table cell is 12 pt or larger (no run below 12 pt); the only charts are pictures: slide 11 chart text is about 12.4 pt on the slide (FS 13 in the figure, scale 0.957), slide 16 about 13 pt. Both pass, slide 11 narrowly.
- Extracted text (pdftotext, UTF-8) of all three, plus raw XML of the docx (comments, headers, footers, footnotes, tracked changes: none) and the pptx (speaker notes, hidden slides: none).
- Numbers compared with data/_results/E-014/sealed_block.md and the E-014 block of docs/EXPERIMENTS.md.

## Result in one line

The three documents agree with each other and with the E-014 block on every sealed number, the T3 wording and the two failed speech predictions, show no pending cell, no em or en dash, no clipped or overlapping text, and carry the right identity fields. The two blocking items are hidden internal IDs in the Office files (Word comments, speaker notes), unchanged since VERIFY4.

## Defect table

| # | document | page or slide | defect | severity |
|---|---|---|---|---|
| 1 | report .docx | whole file (77 Word comments by author "Project team", anchored through the text; none appear in the PDF export) | The docx still carries 77 Word comments with internal IDs and repository names, for example "DECISIONS D-001, D-002, D-011, D-018, D-019; RESEARCH R-001 to R-003, R-017, R-033", "DECISIONS D-003 to D-006, D-016, D-025, D-041", "DOC_FACTS section 2". They show in Word's All Markup view and in the review pane. Same defect as VERIFY4 item 1, not fixed (76 then, 77 now). Strip the comments (or deliver the PDF only). | blocking |
| 2 | slides .pptx | speaker notes of 37 of the 42 slides (not visible on the rendered slides; they print with notes pages and show in presenter view) | The notes carry 246 internal IDs (D-032, D-062, D-067, D-068, E-001, E-004, E-014, M-004, R-xxx style) and instruction-like text, for example slide 19 ends "Use exactly this wording and claim nothing beyond it." Same defect as VERIFY4 item 2, not fixed. Strip or rewrite the notes before submission. | blocking |
| 3 | docs/EXPERIMENTS.md (not a deliverable) | E-014 entry, lines 1065 and 1066 | Heading still reads "the two confirmatory comparisons" and the status line still reads "PLANNED ... run at the freeze (planned Thu 2026-10-29)", although the same entry holds the sealed result run on 2026-10-25 and the amendment to three tests. The amendment explains the change, but the heading and status contradict the result block directly below. The three deliverables are consistent (three tests, 25 October). | medium |
| 4 | paper | pdf 16 (page 8, VI Software, "has not been switched ... still runs the working-version checkpoint") and pdf 26 (page 18, Conclusion, "the web page has not been switched to the frozen system") | Wording says the web page was not switched. docs/STATUS.md records that the web page check on the freeze checkpoint (real adapter, image, video, audio) passed on 2026-10-25. The paper nowhere mentions that check. The wording is true of the default page, but it can be read as "not checked". | low |
| 5 | report | pdf 61 (page 48, end of 4.7.3), pdf 76 (page 63, end of 4.9), pdf 82 (page 69, 4.11.4 last bullet), pdf 86 (page 73, Conclusions bullet "has not been switched to it") and pdf 86 (page 73, 5.2 Further Work first bullet "Switch the local web page ... and run its page check on it") | Same statement as item 4, five places, and Further Work lists as open a page check that passed on the freeze checkpoint on 2026-10-25. Page 62 (pdf 75) reports the parity check on the candidate checkpoint but not the page check. | low |
| 6 | slides | slide 21 (bullet 2), slide 23 (bullet 6), slide 37 (last bullet) | "The page still runs the working version; it has not been switched to the frozen system." Same as item 4. | low |
| 7 | paper | pdf 9 (page 1, Abstract and Index Terms) | No space after the colon: "Abstract:Face swaps ..." and "Index Terms:deepfake ..." (the nodash class replaced the em dash by a bare colon). The abstract is about 560 words, more than twice the usual 150 to 250 words for IEEE. | low |
| 8 | report | pdf 17 (page 4), Figure 1 and its list entry (pdf 10) | Figure 1 still shows the original plan with "Integration and freeze" on Thu 29 and gives no note that the freeze and sealed run were made on 25 October (slide 22 does say "The bars show the original plan"). The body text of 4.11 gives the right date, so this is a missing qualifier, not a contradiction. | low |
| 9 | slides | slide 24 vs paper pdf 9 and report abstract (pdf 5) | Slide 24 quotes "Face development AUC 0.7396 to 0.9096 by fine-tuning" (same encoder, probe against fine-tuned); the paper and report abstracts quote 0.693 to 0.910 (working version against fine-tuned SigLIP). Both exploratory and correct; the slide does not say the baselines differ. Same as VERIFY4 item 18. | low |
| 10 | slides | slide 21 | The three screenshots are cropped through a line of text at the top and their text is unreadable at slide size (the page proper is shown only as a thumbnail). Same as VERIFY4 item 17. | low |
| 11 | slides | slides 33, 35, 36, 39 | The lower 35 to 45 percent of the slide is empty. | low |
| 12 | report | pdf 18 (page 5, three lines), pdf 25 (page 12, only 2.9), pdf 31 (page 18, half empty before Table 3), pdf 55 (page 42, a quarter filled after Table 18), pdf 68 (page 55, about 60 percent empty after Table 26) | Large white gaps where a table or chapter break moved to the next page. Not wrong, but visible. | low |
| 13 | report | pdf 55 (page 42, Table 18) | The table is about 2 mm wider than the text block on each side (still inside the page border). | low |
| 14 | paper, report, slides | paper pdf 24 (Fig. 7), report pdf 69 (Figure 8), slide 16 | The AUC axis of the LAV-DF bar chart runs to 1.1 (an AUC cannot exceed 1). Cosmetic. | low |
| 15 | paper source | paper/main.tex | 33 lines of LaTeX comments carry E-, R- and CLAUDE.md references (for example line 8). Invisible in the PDF, visible to anyone with the source (Overleaf). | low |

Counts: blocking 2, medium 1, low 12 (15 rows).

## Item 1: render checks

- Clipped or overlapping text, text on the footer rule or border, cut-off figures, tables off the page: none found in the paper (27 pages), the report (93 pages) or the slides (42), apart from items 10 and 13. Paper figures 1 to 3 carry a small grey in-figure caption line above the figure body (intended). Paper Figures 5 and the slide 21 screenshots are small by design and labelled so.
- Empty pages: none fully empty. Nearly empty: report pdf 18, pdf 25 (item 12); paper pdf 27 is the reference tail (half page).
- Orphan headings: none found. Paper "IV. Experimental Protocol" now has its first text in the same column on pdf 12 (VERIFY4 item 12 is gone).
- Stray markup: none. No `[pending]` or other placeholder in any of the three documents (grep on the extracted text: none). The report page v orphan Keywords line of VERIFY4 is gone (the abstract now fills pages iv and v).
- Equations: report equations 1 to 42 render (checked pages 26 to 36 at contact-sheet size, 12 and 13 in the Word export); the inline fraction of VERIFY4 item 4 (pi = 1/2) now renders correctly on pdf 32 (page 19). Paper equations 1 to 5 correct.
- Tables of contents, lists of figures and tables: page numbers checked against the rendered footers for the report (chapters, 4.11 to 5.2, References, Appendices, Tables 30 to 33, Figure 12) and the paper (sections VII-E, VII-F, VIII, IX, Figures 3, 4, 6, 7, Tables IX to XI): all agree.

## Item 2: identity fields against docs/IDENTITY.md

University, school, department (Electronics and Computer Engineering), degree wording, the three students and roll numbers in the report order (Vaibhav Shiroorkar 16014024805, Shivansh Gupta 16014023052, Adithya Jithesh 16014024803), guide and designation, Head of the Department (Dr. Jagannath Nirmal), Examiner and Director as labels only, address line, "Mumbai-77", blank date, year 2026-2027, certificate class wording (LY B. Tech, Semester VII) and the footer "Department of Electronics and Computer Engineering Semester VII 2026-27 Batch Page n": all match on the paper cover (pdf 1), certificate (pdf 2), declaration (pdf 3) and title block (pdf 9), the report cover (pdf 1), certificate (pdf 2) and declaration (pdf 3), and the title slide (slide 1, report order; the deck footer carries the department name without "Department of", which IDENTITY allows). No defect.

## Item 3: internal IDs, forbidden words, dashes

- Rendered text of all three: no E-, D-, M-, R- ID, no E-013b form, no reviewer, verifier, fork, agent, autonomously or needs user review. The only ID-like tokens are S-1 (the pulse signal check, defined in the text) and the iteration names I-0 to I-8 with I-1b, I-1c, I-5o, I-5p, I-6o and so on; these are project vocabulary, not repository IDs.
- Hidden text: items 1, 2 and 15. The notes and comments contain no reviewer, verifier, fork, agent, autonomously or needs user review text and no em or en dash.
- Dashes: no em dash or en dash (U+2012 to U+2015) in the extracted text of the paper, the report or the slides, nor in the docx and pptx XML text. The only dash-like characters are U+2212 (mathematical minus, 42 in the paper, 155 in the report, 4 in the slides), which are not dashes.

## Item 4: sealed numbers against the E-014 block

All of the following were found in all three documents, in the right cell:

- T1: +0.1980 [+0.1688, +0.2242], 98.33 percent [+0.1627, +0.2303], AUC 0.8218 [0.8020, 0.8393] against 0.6238 [0.6015, 0.6486].
- T2: +0.1633 [+0.1177, +0.2266], 98.33 percent [+0.1085, +0.2481], AUC 0.9148 [0.8896, 0.9383] against 0.7516 [0.6802, 0.8035].
- T3: +0.0176 [+0.0025, +0.0327], 98.33 percent [-0.0011, +0.0355], baseline AUC 0.8042 [0.7841, 0.8223], delta 0.0100.
- Staircase: paper Table X (point estimates), report Tables 31 and 32 and slide 20 (with intervals), every row and column compared with the block (Celeb-DF family, Celeb-DF v2 alone, Celeb-DF++ with v2 reals in the report, DFD, FF++, ProGAN, CNN subset, diffusion, In-the-Wild, ASVspoof, Xception column): all match. The seventy-one block rows were searched by script; the only misses are rows deliberately not printed (the paper prints no intervals for the staircase, the deck omits the Celeb-DF++ row and the own-label LAV-DF rows).
- Sealed LAV-DF: paper Table XI, report Table 33, slide 39: 0.3226, 0.6462, 0.9921, 0.7344, 0.9945, 0.5216, 0.5032, 0.8480, 0.5203 (and own-label 0.4044, 0.6646, 0.7477, 0.8491, 0.5243 in the paper) match I-5, I-5o, I-5p, I-6, I-6o, I-6p, I-6b, I-6bo, I-6bp.
- Earlier facts: baseline at 3e-4 validation 0.9912, dev_x 0.8716 (report Table 28 pdf 72, slide 13, paper VI Iterations); first pass delta 0.0143 stated only as the first pass, current margin 0.0100 everywhere; I-1b and I-1c do not enter (all three); M-005 recovery at 0.008 (primary, 377 of 432, 0.873) and 0.004 (green only, 0.789) (paper pdf 23, report Table 27, slide 18); S-1 0.301 [0.260, 0.346] against 0.6 (all three); ledger 74 entries (paper pdf 23, report pdf 77, slide 28, EXPERIMENTS.md); predictions: three of five inside their ranges, speech predictions 0.60 to 0.85 and 0.50 to 0.70 failed upward (EXPERIMENTS.md E-014 predictions agree).

## Item 5: consistency and claims

- No document says the sealed tests are not run, "two confirmatory tests" as the current plan (the paper and the report say two of the three compare with the working version, one with the baseline), 0.0143 as the current margin, or a freeze date other than 25 October 2026 in running text (report Figure 1 and slide 22 show the original plan; item 8).
- T3 is worded as "superior at the 95 percent level, not robust at the 98.33 percent (Bonferroni) level, non-inferior at both" in the paper (abstract, pdf 23, pdf 24 Table IX, conclusion), the report (abstract, Table 30, 4.11.1, Chapter 5), and the slides (slides 19, 24 and notes). No place calls T3 robust or "better than" the baseline without the qualification.
- Both failed speech predictions are stated in all three (paper abstract, pdf 23 and 25, limitations and conclusion; report abstract, 4.11, 4.11.4, Chapter 5; slides 19, 23).
- Claims: no "state of the art", "real-time" in a claim about the system (the report quotes "real-time DeepFaceLive fakes" from a cited paper), no "outperforms". "Better than" appears only in cited literature summaries (paper pdf 18, "better than trained baselines in the authors' own tests"). "Superior" appears only for T3 at the 95 percent level.
- Limits stated in all three: 344 groups (44 identity groups plus 300 real YouTube videos), 42 speakers, one baseline family at 224 pixels, delta from three seeds and a floor, learning rate at the edge of the grid, I-1b and I-1c not tested sealed, LAV-DF cue, no pulse ground truth, research-only weights.
- The web page statement: items 4 to 6.

## Item 6: document currency

report/DeepFake_Detection_Technical_Report_preview.pdf (93 pages) and slides/DeepFake_Detection_Presentation_preview.pdf (42 slides) are text-identical to fresh exports of the committed docx and pptx. paper/main.pdf (27 pages, 22:27) is newer than paper/main.tex (22:21).
