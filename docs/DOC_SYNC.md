# DOC_SYNC

Written 2026-10-25. Maps each decided item to the place where it appears in the paper (`paper/main.tex`), the report (`report/DeepFake_Detection_Technical_Report.docx`) and the slides (`slides/DeepFake_Detection_Presentation.pptx`). The documents never print decision IDs in visible text: IDs live in LaTeX comments, Word comments and speaker notes, and this file is the index. Section names are those of the documents as built; paper sections are Roman numerals, report sections are chapter numbers, slide numbers are in the order of the deck (1 to 16 main, 17 to 24 backup).

Placeholders: every `[pending]` carries its experiment ID (or, for a cell with no registered experiment yet, `E-000`) in a comment of the same line (paper), a Word comment (report) or the speaker notes (slides). The E-001 ranking exists in docs/EXPERIMENTS.md but is exploratory and still being re-run, so it is shown as `[pending]` in all three documents (DOC_FACTS section 1). Nothing else carries a measured number.

| Item (decision or open question) | Paper | Report | Slides |
|---|---|---|---|
| Whole system from zero, scope images, video, audio (D-001, D-002, D-018, D-019) | I Introduction; III Method (tasks) | Ch 1 Problem Statement and Scope, Objectives | 2 Problem and scope |
| Deliverables, method MVP first then iterations, goal is comparability not best model (D-003 to D-005, D-016) | I Introduction (contributions); V Iteration Plan | Ch 1 Objectives, Approach and Constraints | 2, 10 |
| rPPG a candidate, not adopted (D-006, O-010) | III Method G Pulse evidence; V Iteration Plan (I-7) | Ch 2 Physiological Signals; Ch 4 The Iteration Ladder (I-7 pulse experiment) | 11 rPPG: test the signal first |
| Local hardware, environment (D-007, D-017; measured in EXPERIMENTS) | IV Protocol A Setup | Ch 4 Environment | 24 Environment (measured) |
| Hosting, timeline to 30 Oct (D-008, D-009) | not in the paper | Ch 4 Summary; Ch 5 Further Work | 13 Timeline to 30 October |
| Output of the system, response structure (D-011, D-027) | III Method; VI Software | Ch 4 The Web Application | 5 Method, 12 Web demo |
| Split and evaluation policy, group-disjoint (D-012, D-026, D-035) | IV Protocol B Data and splits | Ch 4 Datasets, Splits and Licences (Group-disjoint splits, Leakage check) | 4 Protocol, 7 Data and splits |
| Data selection and access, licences (D-013, D-020 to D-022, D-039; DATA_LOG) | IV Protocol B | Ch 4 Datasets, Splits and Licences; Appendix B | 7, 22 Data licences and terms |
| Celeb-DF++ family rule, de-duplication, test balance (D-045) | IV Protocol C Celeb-DF++ de-duplication and test balance | Ch 4 Datasets, Splits and Licences (Celeb-DF++ balance) | 23 Celeb-DF++ test balance |
| Evaluation protocol and metrics (D-024, O-005) | IV Protocol E Metrics and statistics | Ch 3 Mathematical Background (ROC-AUC, EER, Brier, log loss, ECE, temperature scaling); Ch 4 Evaluation Harness | 6 Method and the mathematics, 18 Statistics and decision rules |
| "Comparable with published detectors" as non-inferiority (D-025) | Abstract; IV Protocol E | Ch 3 (paired non-inferiority); Ch 4 Staircase Test Protocol | 6, 18, 19 |
| Shared preprocessing, checkpoint metadata, mock rules (D-028, D-029, D-044) | III Method (preprocessing); VI Software | Ch 4 Shared Preprocessing; The Web Application | 20 Preprocessing, 12 |
| Workstreams, subagent method (D-015, D-030) | not in the paper | not in the report body | notes only |
| Software form, framework (D-031, D-043; O-011, O-013) | VI Software | Ch 4 The Web Application | 14 Web demo (real checkpoint of the working version) |
| MVP: frozen features and linear head (D-032, D-040; O-006) | III Method; V Iteration Plan (I-0, Table) | Ch 4 The Working Version | 9 I-0: the working-version rule |
| Image input: face crop or whole image (D-033; O-002) | III Method | Ch 4 Shared Preprocessing | 5 Method |
| Face detector and preprocessing parameters (D-034) | III Method; IV Protocol | Ch 4 Shared Preprocessing | 20 |
| Encoder sweep, selection rule (tie set, cheapest wins), cost record (D-036, O-007) | IV Protocol (sweep and cost record) | Ch 4 Design of the Encoder Comparison | 8 Encoder shortlist, 21 Cost record |
| Shortlist, conditional licences, DINOv3 exclusion (D-038) | III Method (candidate table) | Ch 4 Design of the Encoder Comparison; Appendix B | 8, 22 |
| Iteration ladder I-0 to I-8 (D-041; O-008) | V Iteration Plan | Ch 4 The Iteration Ladder | 10 Iteration ladder |
| I-1 budget and seed policy (D-042) | IV Protocol (seed policy); V | Ch 4 The Iteration Ladder (I-1 budget) | 21 |
| Staircase and confirmatory claims (D-012, D-025, D-035) | IV Protocol D Sealed tests and the staircase | Ch 4 The Staircase Test Protocol | 19 Staircase and confirmatory claims |
| Mixing of audio and video evidence, fusion (O-009) | III Method; V (late fusion family) | Ch 4 The Iteration Ladder | 10 |
| Metric, calibration and bootstrap mathematics (MATH_NOTES 10 to 17) | IV Protocol E | Ch 3 Mathematical Background (3.1 to 3.19, 42 numbered equations) | 6 |
| Limitations | VIII Limitations | Ch 5 Conclusions and Further Work | 14 Limitations |
| Conclusion (results pending) | IX Conclusion | Ch 5 | 15 Conclusion |
| Deliverable corrections, identity, references (D-037, D-046) | Authors block; References | Front matter; References | 1 Title slide; 16 References |

## Not mapped on purpose

D-010 (documentation method), D-014 and D-023 (start of writing and of early stages), D-015 and D-030 (working method) concern how the project was run and appear nowhere in the three documents.

## Identity fields

All three documents use the values of docs/IDENTITY.md; `scripts/check_docs.py` compares them.
