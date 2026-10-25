# VERIFY3: independent verification of the paper, the report and the slides

Verifier: fresh agent, not a builder of the documents. Date of the check: 2026-10-25 (after the builds of 17:23 to 17:32 IST). Nothing was edited except this file. No git, no GPU.

Files checked: paper/main.tex and paper/main.pdf (23 PDF pages: 8 front matter, 15 numbered), report/DeepFake_Detection_Technical_Report.docx and its preview PDF (80 PDF pages: 12 roman, 68 numbered), slides/DeepFake_Detection_Presentation.pptx and its preview PDF (37 slides). Scratch renders (outside the repo): C:\Users\admin\AppData\Local\Temp\claude\d--Final-Year-Project\1c0278b8-f806-412d-a119-0de0ebf72754\scratchpad (p, r, s folders and the sheet*.png contact sheets).

Stale previews: none. The report preview PDF was written 6 s after the .docx (17:23:59 vs 17:23:53), the slide preview 24 s after the .pptx, the paper PDF 1 s after main.tex. The docx text was also read directly from the XML and agrees with the preview.

## 1. Render review (all 23 + 80 + 37 pages viewed, in contact sheets; suspect pages re-rendered at 100 to 130 dpi)

### Paper
- Layout: no clipped body text, no overlapping text, tables aligned, page numbers present on every page (roman i to vii on the front matter, "Page 1" to "Page 15" in the body), contents, list of figures and list of tables page numbers match the real pages (checked all 7 figures, 9 tables, all section starts).
- Stale content: none. "planned", "being checked", "to be run" do not occur next to iterations; "not yet run / pending" occurs only for the sealed staircase (Table IX, 4 cells plus the conclusion sentence) and the two confirmatory comparisons. 5 `[pending]` strings in total.
- DEFECT, figure text far below 9 pt at printed size (measured from the PDF text spans, page = main.pdf body page):
  - Fig. 2 (body p.5, group-disjoint splits): 3.3 to 5.0 pt (figure drawn at 6.97 in with 8 pt text, then included at `\columnwidth` = 3.5 in, scale 0.5).
  - Fig. 3 (body p.6, staircase): 3.4 to 5.2 pt, same cause (`fig04_staircase.pdf` at `\columnwidth`).
  - Fig. 1 (body p.4, scoring paths): 5.8 to 7.4 pt (0.9 text width).
  - Fig. 4 ladder: 8.3 pt table text; Fig. 6 (p.12): 7.1 pt tick labels; Fig. 7 (p.13): about 8.4 pt category labels; Table VII (p.11) 7.0 pt; Table III and IV 9 pt (fine).
  - Fig. 5 (p.8) screenshots: 6 to 7 pt labels, the page text inside the screenshots is unreadable (inherent to a screenshot at 0.86 text width).
- DEFECT, Fig. 5 (p.8, also List of Figures p.v "Local web page against the development stand-in", and text on p.7 "The screenshots ... were taken against that stand-in") still shows the MOCK-banner stand-in, not the real checkpoint (see section 2, item 4).
- Minor: large vertical gaps between paragraphs on body p.3 left column; body p.15 is half empty (references only). Acceptable.

### Report
- Layout: tables aligned; page numbers present on all 80 pages; contents, list of figures and list of tables match the real pages (Figure 2 p.26, Fig. 3 p.31, Fig. 4 p.38, Fig. 5 p.42, Fig. 6 p.44, Fig. 7 p.46, Fig. 8 p.50, Fig. 9 and 10 p.54, Fig. 11 p.55, Fig. 12 p.56, Table 28 p.57 all correct).
- DEFECT, clipped chapter box: report p.23 (Chapter 4 opening box). The last line "sealed-test results, which do not exist yet, are marked as pending." is cut in half by the bottom of the box (visible in the 110 dpi render). Chapter 1, 2, 3 and 5 boxes are intact.
- DEFECT, figure text under 9 pt: figures are raster images placed at 6.1 in from sources drawn at 6.97 in with 6.3 to 9 pt text, so figure text is about 5.5 to 8 pt (the schematic figures 1 to 6 and 12: about 5.5 to 7 pt; Fig. 2 on p.26 measured by eye at about 6 pt against 12 pt body text; charts Fig. 7 and 8 about 8 pt). Screenshots in Fig. 9 to 11 (p.54, 55) are not legible at page size (inherent).
- DEFECT, Figures 9 and 10 (p.54): captions say "(a) result for an image, (b) result for a video" but the images carry no (a)/(b) labels (see item 5).
- Minor: on full pages the last body line sits directly on the footer rule (p.4 "The references and two appendices follow.", p.23 "...differed from a"); likely template behaviour (layout_check compares to the template and passes), but descenders touch the rule. p.53, p.49 and p.62 have large blank areas (float placement). The report now has 68 numbered pages (D-053 item 7 says 59; the body to the conclusion ends on p.62).
- Stale content: none beyond item 1 below. `[pending]` appears only in Table 28 (5 cells, sealed staircase and confirmatory comparisons). Table 20 caption says "as planned before the runs": legitimate, it is the pre-run plan table, with a pointer to the measured ladder.

### Slides
- All 37 slides viewed. No clipped or overlapping text, no empty placeholder boxes, footer department line and slide numbers on every slide (1 to 37). Minimum text run is 12 pt on a 13.33 in slide (verified in the PDF); embedded charts: slide 11 chart is placed at 7.76 in from a 7.04 in source with 9 pt, so about 9.9 pt; slide 15 chart about 11.7 pt. Slide 17 screenshots (2.3 to 2.7 in wide) are illegible but are labelled (a) image, (b) video with audio, (c) audio file and show the real checkpoint (no MOCK banner).
- Stale content: none. 7 `[pending]` strings: slide 11, 13, 14, 17, 20 (2 rows) and 29, all sealed tests or the two confirmatory comparisons.
- Minor: backup slides 29, 30, 31, 37 leave the lower third empty. Slide 8 (13 pt body) and slides 3, 5, 9, 10 use 12 to 15 pt in tables although the checker line says "body text at least 16 pt": the checker evidently exempts tables.
- Minor: the same encoder shows different "s per 1,000" on slide 8 (CLIP ViT-L/14 OpenAI 4.45, median of 3 from the cost record), slide 26 (4.5, face frames, one extraction run) and slide 27 (4.9, images); all three match their sources (cost_record.json, model_comparison tables) but the labels do not say they are different runs or tasks.

## 2. D-053 defect list and the mistakes table of STATUS.md (2026-10-25)

| Item | Paper | Report | Slides | Verdict |
|---|---|---|---|---|
| 1. 9.7 percent real / 90.3 percent fake, not 10.8 and not 89 | prose only ("strongly imbalanced toward fakes"), no percent: OK | 4.2.2 p.26/27 "9.7 percent": OK. **Section 3.2, p.11 still says "the pooled Celeb-DF family test set is 89 percent fake" (docx XML and preview): NOT FIXED** | slide 36 "Real is 9.7 percent of the pool": OK | **FAIL (report)** |
| 1b. Slide minutes recomputed from step_time_alone.json | n/a | 4.7.1 "203 against 238 seconds": OK | slide 28 Min column 1.4, 3.4, 4.0 equals 47.7, 112.5, 132.2 ms x 1,800 steps / 60: OK | OK |
| 2. T-I wording (no "saturated, does not separate") | p.8: "Both CLIP models reach 1.0000 on the validation split ... cannot separate them; on the development slice the leader (0.9935) is separated from every other candidate by the paired bootstrap, on two generator units only": OK | 4.5.2 p.34 says only "tie set contains only itself", interval caveat for two groups; does not say that validation is 1.0000 for IMG-13 and IMG-15 or that validation cannot separate them (only Table 14 shows 1.0000): partial | slide 9 says only that two generator units make the interval non-population; slide 27 has the numbers; no explicit sentence: partial | OK for paper, partial report and slides (low). No document says "saturated" or "does not separate candidates" (grep clean). |
| 3. T-A INCONCLUSIVE (best lower bound 0.5338 vs 0.55), from-scratch references at or below chance, 0.22 to 0.29 sign-flipped shortcut, all dev_x verdicts exploratory, sealed tests untouched | p.8: all stated, incl. "Values below 0.5 mean a sign-flipped shortcut and not a worse model" | 4.5.3 inconclusive with 0.5338; 4.5.4 "0.219 to 0.286 ... no label inversion ... property of the shift between the corpora" plus the sign reversal of leading silence and energy (Table 17); sealed tests "none was read": OK, the words "sign-flipped shortcut" are not used | slide 9 "inconclusive ... 0.246, below chance"; slide 32 "reverse sign between the corpora ... a data shift, not a label bug"; slide 19 "no sealed test set has been read": OK | OK (low wording gap in report and slides) |
| 4. Paper uses real-checkpoint screenshots; stale text "is being checked" replaced | **Fig. 5 is still the stand-in with the MOCK banner; text p.7 and p.8 says so. NOT FIXED.** "is being checked" is gone: OK | Fig. 9 stand-in (labelled), Fig. 10 and 11 real checkpoint | slide 17 real checkpoint | **FAIL (paper)** |
| 5. Figure text at least 9 pt at printed size; screenshots get (a)(b)(c) labels inside the image, not cropped | **FAIL**: Fig. 2 and 3 at 3.3 to 5.2 pt, Fig. 1 6.3 pt, Fig. 6 7.1 pt; (a)(b)(c) present in Fig. 5 | **FAIL**: figures about 5.5 to 8 pt; Fig. 9 and 10 have no (a)(b) labels in the image | slides OK (charts about 10 to 12 pt, screenshot labels present) | **FAIL (paper, report)** |
| 6a. Paper contents on one page | p.iv one page: OK | n/a | n/a | OK |
| 6b. Report chapter box and list pages fixed | n/a | **Chapter 4 box clipped on p.23**; list pages (figures p.viii, tables p.ix to x): OK | n/a | **FAIL (one box)** |
| 6c. File properties set to the real build date; reference deck metadata removed | PDF: created 2026-10-25 17:32, author = the three students: OK | docx core: created and modified 2026-10-25T11:53Z, author the three students: OK | **pptx docProps/core.xml: created 2022-02-21, modified 2026-07-31, revision 9; docProps/app.xml still says Slides 9, TotalTime 510 and lists the reference deck title "Multimodal Audio-Video DeepFake Detection" in TitlesOfParts. NOT FIXED.** A modified date of 2026-07-31 predates the project (rule 5, dates are real). | **FAIL (pptx)** |
| 7. Not changed, reported to the user | black border report, literature chapter 6 pages, 48 references, report 59 pages, paper 11 body pages | now 68 numbered report pages (to p.62 conclusion, then references and appendices) and 15 paper body pages (incl. 1.5 of references): the numbers in D-053 item 7 are stale | n/a | informational |
| STATUS row: DATA_LOG 10.8 and 89 percent | DATA_LOG.md line 334 says 9.7 / 90.3: OK | report 3.2 still 89 (see item 1) | OK | FAIL (report) |
| STATUS row: T-I "saturated" not written | OK | OK | OK | OK |
| STATUS row: figures with repeated title, tiny text, "previous" in ladder figure | figure titles absent, word "previous" absent from figure text (checked fig03_ladder.pdf, fig08_ladder_results.pdf); tiny text remains (item 5) | same | OK | partial |
| STATUS row: paper reference venues, DeepfakeBench NeurIPS 2023 | [2] "NeurIPS Datasets and Benchmarks Track, 2023": OK; no "venue not verified" text | [5]: OK | slide 21 "NeurIPS Datasets and Benchmarks": OK | OK |

## 3. Identity fields against docs/IDENTITY.md

All present and identical in the three documents:
- Paper (header p.1, cover and certificate pages i to iii): authors in report order Vaibhav Shiroorkar, Shivansh Gupta, Adithya Jithesh; roll numbers 16014024805, 16014023052, 16014024803 on cover, certificate and declaration; "Department of Electronics and Computer Engineering, K. J. Somaiya School of Engineering, Somaiya Vidyavihar University, Vidyavihar, Mumbai - 400 077", "Dr. Tejashree Phatak (Guide), Assistant Professor"; HoD Dr. Jagannath Nirmal; Examiner and Director label only; Date blank, Place Mumbai-77; cover year 2026-2027; certificate wording "Semester VII of LY B. Tech"; footer "Department of Electronics and Computer Engineering Semester VII 2026-27 Batch Page n".
- Report (p.i to iii): same names, order, roll numbers, guide, designation, HoD, degree "Bachelor of Technology (Electronics and Computer Engineering)", cover wording, address line, 2026-2027, footer on every page.
- Slide 1: title, "Department of Electronics and Computer Engineering", group members in report order with roll numbers, "Guide: Dr. Tejashree Phatak, Assistant Professor". Slide footers carry the department name only.
- No deviations found. (Only the pptx metadata issue above.)

## 4. Internal ids and reviewer phrases in visible text

- Patterns D-0xx, E-0xx, O-0xx, M-0xx, V-0xx, R-0xx: none in the PDF text of the paper, the docx XML text, the pptx slide text or the speaker notes.
- "the user" occurs twice in the report (abstract: "so that the user can see which modality carries the evidence"; 4.9: "the user uploads an image"), both mean the end user of the software: acceptable. "orchestrator", "subagent", "fork", "as instructed", task-"brief", TODO, XXX, lorem, "claude": none.
- Em-dash (U+2014) and en-dash (U+2013): 0 in all five text dumps (paper PDF, report PDF, slide PDF, docx XML, pptx XML including notes). check_slides.py also reports 0.
- Project-level labels that remain in visible text and are defined in the text: I-0 to I-8, T-V, T-I, T-A (paper), S-1 (paper, report). They are document terms, not internal ids.

## 5. Claims discipline (CLAUDE.md rule 8)

Grep over all three documents (and the slide notes) for: state of the art, state-of-the-art, real-time, saturated, outperform, better than, best, superior, beats, surpass, "does not separate".
- "state of the art", "real-time", "outperform", "better than", "surpass": 0 occurrences.
- "saturated": only slide 12 notes and slide 30 ("scores saturated at 0 or 1", a description of score values, not a claim about the task); the T-I validation split is described correctly in the paper, never as "saturated and does not separate".
- "best": only in "best lower bound / best candidate / best seed / not the best model in the world" (descriptive); "superior": defined as "only when the whole interval is above zero" (paper p.6, slide 23, report 3.9) and used for E-013 only as "the hypothesis that it is higher is supported as an exploratory result" (report 4.8.5), "exploratory" in paper p.11 and slide 13/20.
- "beats": only in the pre-run hypothesis texts (Table III, report 4.5): hypotheses, not results.
- Required statements, each checked:
  - IMG-13 and IMG-15 reach 1.0000 on validation; on dev_x the leader IMG-13 (0.9935) is separated from the others by the paired bootstrap on only two generator units: paper p.8 yes; report Table 14 and 4.5.2 partial (see item 2); slides partial (slide 9, 27).
  - T-A sweep verdict INCONCLUSIVE: paper p.8, report Table 15 and 4.5.3, slide 9: yes.
  - From-scratch references at or below chance on dev_x (T-V 0.5166; T-A 0.22 to 0.29 sign-flipped shortcut): paper yes in these words; report and slides give the numbers and "property of the shift" / "reverse sign", not the words "sign-flipped shortcut" (low).
  - All dev_x verdicts exploratory: yes in all three (paper p.7, p.9; report 4.5, 4.8; slides 9, 11, 13, 14).
  - Sealed test sets untouched: paper ("not yet run", Table IX pending), report ("None of the sealed test sets was read", 4.8), slides 19 and 24: yes.
  - LAV-DF fused result with the production-pipeline-cue caveat and no audio-visual detection claim: paper abstract, p.12, limitations and conclusion; report abstract, 4.8.3, 5.1; slides 15, 19, 20, 34: yes ("No audio-visual claim", "driven by a production-pipeline cue of the visual head"). Checked that the visual head's within-produced AUC 0.5070 and the E-012 0.5169 are given.
  - "Comparable with a published detector" is defined only as non-inferiority with a margin; the Xception comparison is labelled exploratory on the development slice and the sealed comparison pending: yes in all three.
- No claim defect found in this section.

## 6. Hand check of numbers (51 values; source in brackets; all documents agree where the same quantity occurs)

OK unless marked WRONG. "EXP" is docs/EXPERIMENTS.md; "res" is data/_results.

1. Paper abstract, Table VII, report Table 21/23, slide 13: SigLIP base fine-tuned dev 0.9096 [0.8776, 0.9388] (EXP E-005 result table line "dev_x 0.9096 [0.8776, 0.9388]"). OK, same in all three.
2. Validation 0.9899 [0.9809, 0.9967] (E-005). OK, three documents.
3. Linear probe of the same encoder 0.7396 [0.7031, 0.7832] dev, 0.8767 [0.8582, 0.8966] validation (E-005, E-001). OK.
4. Fine-tuned minus probe +0.1700 [+0.1228, +0.2114] dev, +0.1132 [+0.0948, +0.1299] validation (E-005). OK.
5. Training subset 0.9994 [0.9986, 1.0000] vs probe 0.9245 (E-005). OK.
6. Xception baseline dev 0.8750 [0.8435, 0.9089], validation 0.9872 [0.9784, 0.9942] (E-013, res/baseline/e013.json). OK, three documents.
7. IMG-17 minus Xception +0.0346 [+0.0136, +0.0550] dev, +0.0027 [-0.0009, +0.0060] validation (E-013). OK.
8. Xception minus I-0 probe +0.1817 [+0.1396, +0.2166] (E-013). OK.
9. delta 0.0143 (paper p.6, p.13; report 4.8.5, 4.10; slides 18, 24, 19): seed sd 0.000973 and 0.007175 in e013.json; 2 x 0.007175 = 0.014349, floor 0.01. OK.
10. Baseline learning-rate grid 0.9646, 0.9809, 0.9883 (res/baseline/selection.json). OK (report 4.8.5).
11. Paper Table V T-V: OpenAI CLIP 0.7464 [0.6721, 0.8306] val 0.9046 (E-001). OK.
12. Same table: DINOv2 small 0.6933 [0.6639, 0.7314] val 0.7985; EfficientNet-B4 0.6471 val 0.7478; Xception 0.6162 val 0.7184 (E-001). OK.
13. From-scratch T-V 0.5166 [0.5033, 0.5324] (E-001). OK (paper, report Table 13, slide 26).
14. T-I OpenAI CLIP 0.9935 [0.9803, 0.9982], val 1.0000 (E-002). OK.
15. T-I LAION CLIP 0.9563 [0.7760, 0.9975], diff -0.037 [-0.204, -0.001] (E-002). OK.
16. T-I scratch 0.5500 [0.5080, 0.5735] (E-002). OK.
17. T-A Whisper 0.6380 [0.5338, 0.7700] val 0.9995; WavLM 0.6334; HuBERT 0.6100 [0.5434, 0.7256] val 0.9942 (E-003). OK.
18. T-A scratch 0.2458 [0.0706, 0.4404], per seed 0.2319, 0.2193, 0.2862 (STATUS log and E-003), documents say 0.22 to 0.29 / 0.219 to 0.286. OK.
19. I-0 table (paper Table VI, report Table 18): T-V train 0.8367 [0.8267, 0.8475]; T-A EER 0.4183; T-I ECE 0.0462 / 0.0050; T-A ECE 0.3293 / 0.4052 (E-004). OK.
20. Seed sd 0.0107, 0.0006, 0.0026 and half-widths 0.0338, 0.0089, 0.0911 (E-004; slide 29, report 4.6.1). OK.
21. I-2 dev 0.6723 [0.6126, 0.7584], +0.0623 [+0.0114, +0.1048], validation change -0.0054 [-0.0115, -0.0006] (E-007). OK.
22. I-2 RMS energy on In-the-Wild 0.135 [0.054, 0.382] (E-007 manipulation check). OK.
23. I-3 dev 0.7455 [0.7052, 0.8121], +0.0732 [+0.0122, +0.1225], validation 0.9869 (E-008). OK.
24. I-3 precondition 0.6569 (I-2 head on augmented copy) and 0.9436 (new head) (E-008). OK.
25. I-4 dev 0.8130 [0.7523, 0.8808], +0.0675 [+0.0300, +0.0929], validation 0.9964 [0.9943, 0.9982] (E-009). OK.
26. I-4 mean pairwise correlation 0.534 dev, 0.624 validation; 237 and 64 clips per second (E-009). OK.
27. I-4 ECE ensemble validation 0.0417 / 0.0016, dev 0.0692 / 0.1670 (E-009, res/I-4/ha3.md). OK (paper p.13).
28. I-1 grid DINOv2 small 3e-5 0.7056 / 0.6412, 1e-4 0.5871 / 0.5679; SigLIP 0.9893 / 0.9210; DINOv2 base 0.9918 / 0.9036 (E-005 table). OK (report Table 22).
29. I-1 seeds 0.9893, 0.9896, 0.9908 validation; 0.9210, 0.9166, 0.8912 dev (E-005). OK.
30. I-1 selection: leader DINOv2 base 0.9918 [0.9847, 0.9974], SigLIP minus leader -0.0025 [-0.0068, +0.0009] (res/I-1/selection.json). OK.
31. I-1 aggregation: frame 0.852, clip mean 0.921, top-3 0.926; balanced accuracy 0.974 / 0.824; ECE 0.0261 / 0.0174 and 0.0522 / 0.0331 (E-005, D-054). OK.
32. I-1 per manipulation: 1.000, 0.991, 0.999, 0.969 (E-005). OK.
33. I-5 frozen heads on LAV-DF: visual 0.3838 [0.3599, 0.4079], audio 0.6476 [0.6244, 0.6701], visual vs fused label 0.2786, vs audio-only fakes 0.2876 (E-010). OK (paper Table VIII, report 4.8.3, slide 34).
34. I-6 heads on LAV-DF: visual 0.7491 [0.7280, 0.7697], audio 0.8604 [0.8435, 0.8769], fused-label logistic regression 0.9935 [0.9905, 0.9962], sum of log-odds 0.9795 [0.9726, 0.9857] (E-011). OK, three documents.
35. Fusion minus better head +0.0023 [+0.0009, +0.0040]; weights visual 4.930, audio 0.813 (E-011). OK.
36. Post-hoc control: visual within produced videos 0.5070 [0.4771, 0.5361], audio 0.8574 [0.8394, 0.8752] (E-011). OK.
37. E-012: 0.5169 [0.4945, 0.5392], +0.0100 [-0.0159, +0.0372], real vs audio-only fakes 0.5347, all val 0.5342 (E-012). OK (report 4.8.3, slide 34).
38. S-1: 130 of 432 = 0.301 [0.260, 0.346]; FF++ 0.280, Celeb-DF 0.288, LAV-DF 0.333; threshold 1.118 dB; median -0.222 dB (E-006, res/S-1). OK.
39. Parity max difference 0.0019 (T-V 0.0002, T-I 0.0019, T-A 0.0005), tolerance 0.02 (E-004). OK (report, slide 31, paper p.7).
40. Celeb-DF family test: 744 real, 6,906 fake, 7,650 total; real share 9.7 percent (744/7,650 = 0.0973); fake share 90.3 percent (DATA_LOG.md line 334). OK in report 4.2.2 (p.26/27) and slide 36. **WRONG in report 3.2 p.11: "89 percent fake" (docx and preview); correct value 90.3 percent.**
41. Celeb-DF++: 5,078 extracted, 2,079 excluded (1,477 target + 602 source-only), 150 duplicate, 2,849 fake and 0 real (DATA_LOG.md line 333, D-045). OK (5,078 - 2,079 - 150 = 2,849).
42. Celeb-DF v2 sealed test 4,801 videos (744 real, 4,057 fake) and dev slice 1,728 (146 real, 1,582 fake) (EXP line 127; 590 + 300 - 146 = 744; 5,639 - 1,582 = 4,057). OK.
43. CNN synthesis sealed set 48,805 images after removing 1,200 duplicate reals; dev slice 5,998 (2,999 + 2,999); leakage check on 80,803 and 41,739 files (EXP lines 125 to 127). OK.
44. Sizing probe: float32 134.5 ms, 1,903 img/s, 5,771 MiB; bf16 77.6 ms, 3,300, 3,787 MiB; fp16 83.4 ms, 3,068, 3,787 MiB (EXP lines 67 to 69). OK (report Table 5).
45. Face detector table: 12.1 / 12.4, 4.8 / 4.8, 29.7 / 30.9 ms per frame; 6.0, 20.2, 4.6 percent hits on face-free images; IoU 0.887 to 0.923 (M-002, S2C.md). OK (report Table 9, 4.3).
46. Fine-tuning budget minutes: 47.7, 112.5, 132.2 ms per step x 1,800 = 86, 203, 238 s = 1.4, 3.4, 4.0 min (data/_logs/step_time_alone.json, D-054 text). OK (slide 28, report 4.7.1).
47. Slide 8 cost: EfficientNet-B4 17.5 M parameters, 6.1 GFLOPs, 1.73 s per 1,000; Xception 16.7 GFLOPs, 3.32 s (res/cost_record.json). OK.
48. Audio diagnosis (report Table 17, slide 32): leading silence 0.416 [0.350, 0.488], 0.375 [0.299, 0.453], 0.719 [0.425, 0.933]; energy 0.572, 0.594, 0.248 [0.073, 0.546] (EXP line 174, M-004). OK.
49. I-0 validation balanced accuracy 0.750 / 0.617, 0.998 / 0.955, 0.971 / 0.529 (E-004). OK (slide 29, report Table 19).
50. Worst errors: 7 of 20 on one identity group (id1: 2 false alarms + 5 misses in the E-005 worst list); 6 of 10 misses on id1 at I-0 (E-004 item 7). OK.
51. Table 6/7 split sizes in the report: FF++ train 3,600 clips (720 real), val 700 (140 real), dev slice 15 of 59 groups; ASVspoof 25,380 / 24,844 / 71,237; In-the-Wild dev slice 6,991 (3,804 real, 3,187 fake), 12 of 54 speakers (EXP, STATUS, E-003 header). OK.

Cross-document agreement on one row each: baseline row (paper Table VII, report Table 26, slide 13): identical. I-4 row (paper Table IV, report Table 24, slide 14): identical. Fused-label LAV-DF row (paper p.12, report Table 25, slide 34): identical. Test counts (report Table 7, slide 36): identical. delta in all three: 0.0143. Ladder summary (paper Fig. 4 and Table VII, report Table 21 and Fig. 5, slides 11 and 20): identical.

Result of the hand check: 50 OK, 1 WRONG (report 3.2, 89 percent). The checker limits are those stated in STATUS: a wrong quantity with a number that exists elsewhere passes.

## 7. Checkers (system python, repo root, PYTHONIOENCODING=utf-8)

- `python scripts/check_docs.py --numbers`: final line "71 pass, 0 fail". It did not catch the "89 percent fake" string in report 3.2 (the string "89" rounds from a logged value in its provenance rule), as predicted by STATUS.
- `python report/layout_check.py`: final line "26 of 26 rows within 2 mm". It does not look at chapter boxes or figure text sizes (the clipped Chapter 4 box on p.23 passes).
- `python slides/check_slides.py`: final line "614 pass, 0 fail" (552 of 552 result-like numbers matched; identity, notes, dashes, ids OK).

## 8. Prioritised defect list

| # | Severity | Location | Evidence | Suggested fix |
|---|---|---|---|---|
| 1 | High | Report 3.2 (p.11), docx and preview | "(the pooled Celeb-DF family test set is 89 percent fake)"; D-045 correction, DATA_LOG line 334 and D-053 item 1 give 90.3 percent fake (9.7 real). STATUS "Mistakes" row says the three documents are corrected: false for this sentence. | Replace by "90.3 percent fake, 9.7 percent real"; add a check_docs rule that rejects "89 percent" in the three documents. |
| 2 | High | Paper Fig. 2 (body p.5), Fig. 3 (p.6), Fig. 1 (p.4), Fig. 6 (p.12), Table VII (p.11); report Figures 1 to 6 and 12 | Text spans in main.pdf: Fig. 2 3.3 to 5.0 pt, Fig. 3 3.4 to 5.2 pt, Fig. 1 5.8 to 7.4 pt, Fig. 6 7.1 pt; the report images are about 5.5 to 7 pt. D-053 item 5 says "every figure text is at least 9 pt at the printed size". | Place the two column figures as `figure*` (full width) or redraw them at 9 pt for 3.5 in; enlarge the fonts in make_figures.py for the report width (6.1 in); re-measure with a span listing. |
| 3 | High | Paper Fig. 5 (p.8), List of Figures, text p.7 and p.8 | Screenshots show the MOCK banner and "Likely fake 89.5 percent" (stand-in) and the caption says "development stand-in". D-053 item 4 says the paper uses the real-checkpoint screenshots. | Re-embed assets/screens/real_*.png (image, video with audio, audio file) as the report Fig. 10 and the slides do, with (a)(b)(c) labels, and reword the text on p.7. |
| 4 | Medium | pptx docProps (core.xml, app.xml) | created 2022-02-21, modified 2026-07-31, revision 9; app.xml "Slides 9", "TotalTime 510", TitlesOfParts "Multimodal Audio-Video DeepFake Detection" (reference deck). D-053 item 6 says removed. 2026-07-31 predates the project (rule 5). | Reset core.xml dates to the build date, revision 1, drop the reference TitlesOfParts and slide count (python-pptx `core_properties` plus rewrite app.xml). |
| 5 | Medium | Report p.23, Chapter 4 opening box | Last line "sealed-test results, which do not exist yet, are marked as pending." is cut in half by the box bottom (110 dpi render). D-053 item 6 says chapter boxes fixed. | Increase the box height or shorten the text to four lines; add a chapter-box row to layout_check. |
| 6 | Medium | Report Figures 9 and 10 (p.54) | Captions use "(a) ... (b)" but the images carry no (a)/(b) label; D-053 item 5 requires labels inside the image. | Add the labels to the images as in paper Fig. 5 and slide 17. |
| 7 | Low | Report 4.5.2 (p.34) and slides 9, 27 | The required sentence (IMG-13 and IMG-15 reach 1.0000 on validation; on dev_x the leader 0.9935 is separated by the paired bootstrap, two generator units only) appears in full only in the paper p.8. | Copy the paper sentence to report 4.5.2 and add one bullet to slide 9. |
| 8 | Low | Report 4.5.4 and slides 9, 32 | The T-A references (0.22 to 0.29) are described as "property of the shift" / "below chance" but not as "a sign-flipped recording-condition shortcut". | Add that phrase (paper p.8 has it). |
| 9 | Low | Report body pages (for example p.4, p.23) | Last body line touches the footer rule; layout_check passes because it measures against the template. | Check against the template in Word; add 2 to 3 pt space before the footer if the template has it. |
| 10 | Low | Slides 8, 26, 27 | CLIP OpenAI "s per 1,000": 4.45, 4.5, 4.9 (cost-record median, one extraction run face frames, one run images). All match sources but are not labelled as different runs. | Add "(cost record, median of 3)" or "(extraction run)" to the column or footnote. |
| 11 | Low | check_slides.py | Reports "body text at least 16 pt: PASS" although tables and body on slides 3, 5, 8, 9, 10 are 12 to 15 pt (PDF spans). | State in the checker docstring that tables are exempt, or raise the sizes. |
| 12 | Low (informational) | D-053 item 7 and STATUS | Page counts are stale: report now 68 numbered pages (D-053 says 59), paper 15 body pages (11), deck 37 slides (32). | Update the counts in STATUS and D-053 consequences. |

## 9. Verdict

The content is in good shape: the iterations I-1 to I-7 and the Xception baseline are present with results in all three documents, only the sealed staircase and the two confirmatory comparisons are `[pending]`, no document says the image task is saturated, no unsupported claim ("state of the art", "real-time", "outperform") exists, the LAV-DF fused result carries the production-pipeline-cue caveat and no audio-visual detection claim, the T-A sweep is INCONCLUSIVE everywhere, identity fields are exact, there are no internal ids, reviewer phrases or dashes, and 50 of 51 hand-checked numbers match their logs, with the three documents agreeing on every shared row. The defect fixes of D-053 are not all done, however: one stale wrong number survives in the report (89 percent fake, p.11), the paper still shows the stand-in screenshots, figure text in the paper (4 to 7 pt) and the report (about 5.5 to 8 pt) is well below the 9 pt the decision log claims, one report chapter box is clipped, the report screenshots lack (a)/(b) labels, and the pptx still carries the reference deck's creation date, a 2026-07-31 modification date and its slide-count metadata. All three checkers pass (71 pass, 26 of 26 rows, 614 pass) but none of them can see these seven problems, so a clean checker run must not be read as a clean verification. Not ready to hand in until items 1 to 6 of the list are fixed; items 7 to 12 are polish.
