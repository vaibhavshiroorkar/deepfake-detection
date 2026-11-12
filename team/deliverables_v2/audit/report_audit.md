# Report audit (analysis only, 2026-10-28)

Audit of the ORIGINAL technical report (report/, 119 pages). The audit subagent was refused permission to write this file and returned its findings by message; the main agent wrote this file from that message (brief of 2026-10-28, rule "a subagent that cannot write its file reports by message, and you write the file"). Page numbers are PDF pages of the original; the printed page number is the PDF page minus 16. Nothing in the repository was changed by the audit.

## STOP flags

No headline number is wrong. Every decimal with 3 or more places in the abstract and Chapters 1 to 5 was searched verbatim in docs/EXPERIMENTS.md (this checks that the token exists, not that it sits next to the same quantity). Percent and 2-decimal figures were spot-checked. Items not found in EXPERIMENTS.md:

1. p.53 (4.5.1): "1.732 s per 1,000 inputs" is in DECISIONS.md and NUMBERS.json only.
2. p.82 (4.10.3): "0.999983 on the face crop" is in DECISIONS, STATUS, TESTING_RECORD and NUMBERS.json only; it is a demo reading, not a measurement.
3. p.86 (4.10.9): "flagged about 0.11 of the real videos of the Celeb-DF family and 0.14 of DFD" is only in DECISIONS.md; post hoc reading of sealed scores, no E-entry (now E-032, logged 2026-10-28).
4. Contradiction: Chapter 5 p.111 says "497 passed, 2 skipped"; Section 4.11 and Table 34 (p.89 to 90) say 710 passed; 497 is the 26 October record.
5. Chapter 5 p.111 says "304 unit tests", an earlier count (4.9, p.78); the suite is now 713.
6. Abstract p.5, sentence 1, says the score is "calibrated on validation data only"; Sections 4.9 and 4.10.3 say the audio score is a ranking score and 4.14.1 says thresholds do not transfer (rule 8 exposure).

## 1. Current state

| Item | Value |
|---|---|
| Pages | 119 (A4) |
| Front matter | 16 pages: cover, certificate, declaration, acknowledgement, abstract (2), contents (3), list of figures (2), list of tables (3), nomenclature (2) |
| Words (pdftotext) | 45,486 default mode (about 1,400 are repeated footers); 44,779 by page split |
| Abstract | 665 words over 2 pages |
| Numbered figures / tables / equations | 14 / 45 / 42 (only equations 22, 23 and 42 are cited later) |
| References | 53 (template says 10 to 15) |
| En or em dashes | 0 in the PDF text |

Words by part: front 3,767; Ch1 1,713; Ch2 2,580; Ch3 4,289; Ch4 27,915; Ch5 2,492; references 1,663; appendices 360.

| Part | PDF pages | Count |
|---|---|---|
| Ch1 Introduction | 17 to 21 | 5 |
| Ch2 Literature Survey | 22 to 28 | 7 |
| Ch3 Mathematical Background | 29 to 40 | 12 |
| Ch4 Implementation and Experimentation | 41 to 107 | 67 (56 percent of the PDF, 69 percent of the body) |
| Ch5 Conclusions and Further Work | 108 to 113 | 6 |
| References | 114 to 117 | 4 |
| Appendix A and B | 118 and 119 | 2 |

Chapter 4 subsections (start PDF page, length in pages): 4.1 Environment 41, 1.5; 4.2 Datasets 42, 3.6; 4.3 Preprocessing 46, 1.45; 4.4 Harness 47, 2.0; 4.5 Encoder comparison 49, 6.8; 4.6 Working version 56, 4.1; 4.7 Ladder 60, 4.0; 4.8 Results 64, 10.8; 4.9 Web application 75, 3.6; 4.10 Platform 79, 9.9; 4.11 Testing 89, 1.8; 4.12 Protocol 90, 1.7; 4.13 Sealed results 92, 5.0; 4.14 After the freeze 97, 9.8; 4.15 Summary 107, 0.6. Software description (4.9 to 4.12, printed pages 59 to 76) takes 17 pages before the first sealed result at 4.13. Set-up 7.8 pages; pre-freeze experiments 25.7; software 15.3; protocol and sealed 6.7; post-freeze 9.8.

## 2. Claim audit (54 rows checked; 47 numbers match exactly; 3 not in EXPERIMENTS.md; 1 stale test count; 3 literature numbers traced to RESEARCH or NUMBERS_LIT)

Key: Pre = declared before the run; PH = post hoc; Expl = exploratory (validation or development slice); Seal-C = confirmatory sealed (T1 to T3); Seal-E = sealed but exploratory.

| # | Location | Claim | E-ID | Match | Pre/PH, level | Verdict |
|---|---|---|---|---|---|---|
| A1 | Abstract p5 | Fine-tuning raises dev AUC 0.6933 to 0.9096 | E-004, E-005 | yes | Pre, Expl (15 identities) | WEAKEN: the two numbers differ in encoder (DINOv2 small vs SigLIP base); write "SigLIP probe 0.7396, fine-tuned 0.9096, paired +0.1700 [+0.1228,+0.2114]" |
| A2 | p5 | 1b 0.9315 and 1c 0.9192 do not enter | E-016, E-017 | yes | Pre, Expl | KEEP |
| A3 | p5 | Speech 0.6100 to 0.8130 | E-007 to E-009 | yes | Pre, Expl (12 speakers) | KEEP, add "12 speakers" |
| A4 | p5 | Pulse fails; injected 0.008 sinusoid recovered in 87% | E-006, M-005 | yes | Pre / PH (control) | KEEP, mark control as post hoc |
| A5 | p5 | LAV-DF fused result driven by pipeline cue | E-011 | yes | PH, Expl | KEEP, label post hoc |
| A6 | p6 | T1 0.8218 vs 0.6238, +0.1980 [+0.1688,+0.2242] | E-014 | yes | Pre, Seal-C | KEEP |
| A7 | p6 | T2 0.9148 vs 0.7516, +0.1633 [+0.1177,+0.2266] | E-014 | yes | Pre, Seal-C | KEEP |
| A8 | p6 | T3 +0.0176 [+0.0025,+0.0327]; 98.33 [-0.0011,+0.0355]; non-inferior at both | E-014 | yes | Pre, Seal-C | KEEP (wording now D-091) |
| A9 | p6 | Two speech predictions failed upward | E-014 | yes | Pre | KEEP |
| A10 | p6 | "344 groups and 42 speakers" | E-014 | yes, misleading | | WEAKEN: "344 groups (about 44 identities)" (300 of the 344 are single YouTube videos) |
| A11 | p6 | Generated-video head at chance, 0.4957 | E-020 | yes | Pre, held-out | KEEP |
| A12 | p6 | Frozen system "ranks well" on FakeAVCeleb | E-015 | yes | Pre, Seal-E | WEAKEN: delete "well"; note codec alone gives 0.9985 |
| A13 | p5 | "calibrated on validation data only" | E-004 | audio ECE 0.3293 to 0.4052 after scaling | | WEAKEN (STOP 6) |
| A14 | p5 to 6 | 665-word abstract | | | | MOVE platform and post-freeze text to 3 sentences, target 250 words |
| R5 | p55 to 56 | "no label inversion: values below 0.5 indicate a sign-flipped shortcut" | M-004 | yes | PH, Expl | WEAKEN: "consistent with"; 12 speakers do not establish it |
| R7 | p60 | "face size and number show no effect" (0.762 vs 0.692) | E-004 | yes | PH | WEAKEN: "no effect detected on this slice" |
| R11 | p71 to 72 | "The part of the system that detects a manipulation on LAV-DF is the audio head" | E-010 to E-012 | yes | PH, Expl | WEAKEN: "only head with evidence above chance inside produced videos (0.8574)" |
| R13 | p74 to 75 | Xception 0.8716; +0.0380 [+0.0172,+0.0592]; "most of the gain comes from fine-tuning" | E-013b | yes | Pre, Expl | KEEP, add "on this slice" |
| R14 | p86 to 87 | Specificity 0.996, sensitivity 0.480; 0.11 and 0.14 flagged; image model missed 0.7395, macro 0.7039 | E-015, E-028 | partly | | KEEP; 0.11 and 0.14 now E-032 |
| R17 | p95 | "AUC rises in each step 0.7516, 0.8125, 0.8672, 0.9148" | E-014 | yes | Seal-E | WEAKEN: "point estimates rise; adjacent intervals overlap" |
| R20 | p98 to 99 | Codec model 0.9985; pixels 0.9585; re-encode 0.9732 to 0.8772; matched 0.8856 vs 0.8763 | E-021, E-023 | yes | Pre | KEEP, shorten |
| R21 | p99 | DeepAction audit; pixels 0.773 and 0.787 | E-018, E-019 | yes | Pre | MOVE to appendix |
| R22 to R26 | p100 to 104 | DF26 0.4957; 338/440 flagged; 0.7548; 0.8023; image model 0.7039; retrained 0.8839 | E-020, E-026 to E-029 | yes | Pre (second look disclosed) | KEEP, merge Tables 40 and 41, and 42 and 43 |
| R27 | p104 | EER 6.94 [5.95,7.82] vs AASIST 1.13, "5.81 points worse: ... pays for that in-corpus" | E-030 | numbers yes | PH | WEAKEN: delete the causal clause; context (AASIST not re-run, rule 8) |
| R28 | p104 to 105 | "Class A" rows including AASIST and RawNet2 get a difference | | | | WEAKEN: class A needs the same training protocol; AASIST was not re-run |
| R29 | p105 to 106 | Lip-sync: 97.8%; 15.0% and 13.3%; band holds 98.2%; 1.3% [0.4,3.8]; 0 of 160 | E-022, E-024, E-025 | yes | Pre / Expl | KEEP last three; MOVE steps 1 and 2 |
| K2 | Ch5 p108 to 109 | Dev numbers repeated in full | | yes | Expl | DELETE; point to Table 21 |
| K5 | p111 | "The speech task is the weakest" | E-014 | | | WEAKEN or DELETE: only the 12-speaker slice is low |
| K6 | p111 | "497 passed" | TESTING_RECORD | NO | | FIX (STOP 4) |

Wording changes recommended on 13 rows: A1, A10, A12, A13, A14, R5, R7, R11, R17, R27, R28, K2, K5. No confirmatory claim (T1 to T3) needs weakening beyond the D-091 wording. "Superior" appears only for T3. No "real-time", "state of the art" or "better than X" in the sections read (manual read, not a full-text search of the PDF).

## 3. Section audit (verdict: CORE, SUPPORTING (short), APPENDIX, DELETE; target = main-text pages)

| Section | Now | Verdict | Target | Reason |
|---|---|---|---|---|
| Abstract | 2.0 | CORE, rewrite | 1.0 | 665 words; carries platform and FakeAVCeleb stories; claim A13 |
| Ch1 | 5.0 | CORE, shrink | 3.5 | Figure 1 is a schedule; objective 7 and 1.3 describe the later platform; 1.6 repeats the contents |
| Ch2 | 7.0 | CORE, shrink | 5.0 | Template says "around 2 to 5 pages"; Table 2 never referenced |
| 3.2 AUC | 1.0 | CORE | 0.5 | Textbook proof |
| 3.3, 3.4 EER, precision, balanced accuracy | 0.85 | SUPPORTING | 0.45 | precision derivation unused |
| 3.5, 3.6 Brier, ECE, Platt, temperature | 1.75 | SUPPORTING (cite R-016) | 0.8 | Murphy decomposition and Platt never used |
| 3.7 to 3.9 bootstrap, paired, non-inferiority | 1.75 | CORE | 1.6 | Eq 22 and 23 cited |
| 3.10 Table 3 | 0.5 | DELETE | 0 | restates 3.2 to 3.6 |
| 3.11 Linear probe | 0.7 | SUPPORTING | 0.4 | Hessian derivation |
| 3.12 LoRA | 0.4 | DELETE (cite) | 0 | never run |
| 3.13 Aggregation | 0.6 | SUPPORTING | 0.3 | top-k bias formula unused |
| 3.14 Loss functions | 0.6 | DELETE | 0 | the loss lever was never run |
| 3.15 Regularisation, augmentation | 0.7 | SUPPORTING | 0.3 | ridge and SVD unused |
| 3.16 to 3.18 combining, fusion, SNR | 2.0 | CORE or SUPPORTING | 1.3 | 7.6 percent bound quoted; Eq 42 cited |
| 3.1, 3.19 | 1.05 | CORE | 0.6 | notation; template summary |
| 4.1 Environment | 1.5 | SUPPORTING (Table 4 only) | 0.5 | CUDA wheel, matmul check, toy-network Table 5, download speed support no claim |
| 4.2 Datasets and splits | 3.6 | CORE (4.2.1 to 4.2.3), SUPPORTING (Table 6) | 3.1 | group-disjoint design answers leakage questions; Table 6 licence text is 1.2 pages |
| 4.3 Preprocessing | 1.45 | SUPPORTING | 0.6 | three-detector comparison and worker-count throughput feed no result |
| 4.4 Harness | 2.0 | SUPPORTING | 0.9 | rule 3 evidence stays; Table 10 and timings are test detail |
| 4.5 Encoder comparison | 6.8 | CORE, shrink | 3.5 | Fig 3 duplicates Table 12, which duplicates Table 45; Tables 14 to 17 near-identical |
| 4.6 Working version | 4.1 | CORE / SUPPORTING | 2.3 | Table 19 repeats Table 16 |
| 4.7 Ladder | 4.0 | SUPPORTING | 1.0 | Table 20 and Fig 5 repeat Table 21 |
| 4.8 Results | 10.8 | CORE | 6.7 | LR grid Table 22, member rows, Fig 8 repeats Table 26 |
| 4.9 Web app | 3.6 | SUPPORTING | 1.0 | framework selection; field table; three screenshots, one a stand-in |
| 4.10 Platform | 9.9 | SUPPORTING (3 pages) | 3.0 | built after the freeze; no detection result; software documentation |
| 4.11 Testing | 1.8 | SUPPORTING | 0.7 | earlier-record rows |
| 4.12 Protocol | 1.7 | CORE, dedupe | 1.0 | three tests described twice here and again in 4.13.1 |
| 4.13 Sealed | 5.0 | CORE | 3.3 | limits repeated in 4.14.10, 5.1, 5.2 |
| 4.14 Post-freeze | 9.8 | CORE (4.14.1, 4.14.8), SUPPORTING (rest) | 5.0 | four generated-video subsections for a module that is always inconclusive |
| 4.15 Summary | 0.6 | CORE | 0.3 | repeats Ch5 |
| 5.1 Conclusions | 4.2 | CORE, rewrite | 1.8 | about 30 numbers repeated; K5, K6 |
| 5.2 Further work | 1.6 | CORE | 0.7 | 17 items |
| References | 4.0 | CORE | 3 to 4 | 53 vs template 10 to 15 |
| Appendices | 2.0 | SUPPORTING | 1.5 | hashes duplicate Table 12 |

Judgements: Chapter 3 keep only what results use (AUC rank form, EER, balanced accuracy, ECE, Brier, temperature scaling, group bootstrap, paired and non-inferiority Eq 22 and 23, the 3.16 correlation formula, SNR); delete LoRA, loss functions, Murphy, Platt, ridge, Table 3. Chapter 4: CUDA wheel, toy-network probe, download speed, Table 9 face-detector comparison and worker-count throughput support no claim; Table 10 is rule evidence, appendix. The suspected repetition of sweep tables in the appendix is not present (appendices hold only the settings table and the weights table); the real repetition is within Chapter 4 (Table 16 vs 18 and 19; Table 12 vs 45; Table 44 vs the 4.5 head description). Platform sections SUPPORTING, 3 pages. Post-freeze 5 pages suffice.

## 4. Table and figure audit

All 14 figures and 42 of 45 tables are cited; Tables 2, 8 and 23 are never cited.

Figures: 1 seven-day plan (drop); 2 splits (keep); 3 encoder shortlist, duplicates Table 12 (drop); 4 file to score (keep); 5 ladder, duplicates Tables 20 and 21 (drop); 6 pulse chain (move); 7 measured ladder (keep); 8 LAV-DF cells, duplicates Table 26 (move); 9 web page vs stand-in (drop); 10 web page real checkpoint (keep one); 11 web audio warning (move); 12 platform (keep); 13 dashboard (move); 14 staircase protocol (keep).

Tables: keep (in text) 1, 4, 6, 7, 11 (compress), 13, 18, 21, 23 (uncited, add a reference), 24, 26, 28, 33, 34 (trim), 35, 36, 37, 38 (shorten), 39, 44, 45. Move to appendix: 8 (uncited), 9, 10, 12 (merge with 45), 14 and 15 (summary rows stay), 16, 17, 20, 22, 25, 27, 29, 30, 31, 32. Drop: 2 (uncited; says "no two rows comparable"), 3 (restates Chapter 3), 5 (toy probe), 19 (repeats Table 16). Merge: 40 with 41, 42 with 43. Other duplicates: Table 18 repeats the Table 13 row; Fig 7 overlaps Tables 21, 23, 25; Table 36 repeats the Table 35 row; Table 38 repeats Table 26. Net: 14 figures to about 7, 45 tables to about 24 in the main text.

## 5. Length limit

Sources: the Word template (read through COM on a copy) and the reference report PDF (41 pages, 6 chapters; no stated limit).
- Template page 3, last bullet under "Some important instructions": "The report should be of at least 30-60 pages but no one can make a rule for number of pages of thesis".
- Page 1: "Expected number of references is 10 - 15" and "Literature survey should be around 2 to 5 pages". Page 3: "Number of chapters may change from thesis to thesis". Chapter 6 note: "brief description at least 8 to 10 lines followed by important conclusions highlighted in point form".
- The 30 to 60 range is written as "at least", with the statement that no page rule can be made; there is no hard maximum.
Recommendation of the audit: a 60-page budget for Chapters 1 to 5 plus references (about 54 to 57 achieved), with about 14 appendix pages labelled supporting material. Against a hard cap: rules 2 and 9 keep failed runs and limits on record, so evidence should move to appendices, not be deleted.

## 6. Cut plan (about half a day; for the ORIGINAL report; the submission set is deliverables_v2/, D-090)

| Part | Now | Target main | To appendix |
|---|---|---|---|
| Front matter | 16 | 14 | |
| Ch1 / Ch2 / Ch3 | 5.0 / 7.0 / 12.0 | 3.5 / 5.0 / 5.5 | 0 / 0 / 0.5 |
| Ch4 total | 67 | 34 to 36 | about 16 |
| Ch5 / References | 6.0 / 4.0 | 2.5 / 3 to 4 | |
| Appendices | 2.0 | | 14 (new C to G; A and B merged) |

Expected: Chapters 1 to 5 about 51 to 53 pages; numbered main text with references 54 to 57; appendices about 14; front matter about 14; total PDF about 83 to 86. Plan B (everything in 60 numbered pages) gives about 74 PDF pages and deletes record; not recommended.

Moves to appendices: C (2 pages) Table 5, download speed, CUDA wheel, Table 9, throughput, Table 10; D (4) Tables 12 (merged with 45), 14 to 17, 20, 22, 25, 27; E (1) Table 29, Figures 9 and 11, framework criteria; F (5) Tables 30 to 32, dashboard, explanations, Evaluation Lab, Figure 13; G (2.5) DeepAction recipe history, lip-sync steps 1 and 2, FakeAVCeleb follow-up details.

One-paragraph summaries that replace removed subsections: 4.1 one Windows 10 desktop, RTX 5070 (12 GB), Python 3.13, PyTorch with CUDA 13.0 wheels; 4.3 YuNet chosen over UltraFace and SCRFD (all found a face in every sampled frame; YuNet small, MIT licensed, fast on the processor); 4.4 the harness passed its acceptance checks including bootstrap coverage 192 of 200; 4.5 cheapest tie-set members DINOv2 small (face video), OpenAI CLIP (generated images), HuBERT (speech); 4.7 each lever chosen after the battery of the previous step; 4.9 local Flask page, one structured response, stand-in banner; 4.10 one entry per module, six statuses, three sections, no global score, the Lab never writes the ledger; 4.14.2 to 4.14.5 header shortcuts removed by the third re-encode recipe but pixel statistics still reach 0.77 to 0.79, appearance head at chance on DF26 (0.4957) and flags 0.7682 of other real clips, DF26-trained head 0.7548 on commercial generators but flags 0.8023 of real clips, so nothing is installed.

Caveats: page fractions are from pdftotext line positions (about 0.1 page); Chapter 2 was assessed for length and template fit only, not claim by claim; Tables 35 to 43 were compared by eye against E-014, E-015, E-020, E-027, E-028 and E-029.
