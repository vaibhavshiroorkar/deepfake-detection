# FINAL SUBMISSION AUDIT

Written 2026-10-30 (Phase 3 of the final-submission preparation). Read-only audit of `submission/` and the supporting folders against the four Google Classroom tasks (28 to 31). Nothing in `submission/` was changed. Owners are left as "to assign" until the three members approve the work allocation (docs/TEAM_WORK_ALLOCATION.md, after the interview); this audit does not decide who does what.

Status vocabulary: **complete**; **incomplete** (exists but misses a stated requirement); **missing**; **requires human action** (only a person can do it: signing, uploading, recording, approving).

## 0. Package integrity

| Check | Result | Evidence |
|---|---|---|
| Hashes of the 3 PDFs, the DOCX and the PPTX against submission/MANIFEST.txt | all match | sha256sum on 2026-10-30 |
| Text records (.md) against the manifest | match when hashed with LF line endings, as the manifest states | same |
| `git status` shows submission/records/PROTOCOL.md modified | line endings only; `git diff --numstat` is empty; content identical to HEAD and to docs/PROTOCOL.md | git, 2026-10-30 |
| Tags | `submission-final` (e27de9a), `submission-v2` (7a61675) | git tag |
| Archived originals | paper/, report/, slides/ and "Document Template/" hold older versions; they must not be submitted (D-090) | directory listing |

## 1. Task 28: signed final technical report

Artifact: `submission/Multimodal_Deepfake_Detection_Project_Report.pdf` (37 pages, A4, Times New Roman) and its editable source `submission/sources/report/Multimodal_Deepfake_Detection_Project_Report.docx`.

| Requirement | Existing artifact (path, section) | Status | Evidence | Exact remaining action | Owner | Dependencies | Risk if not done |
|---|---|---|---|---|---|---|---|
| Signed PDF | report PDF, Certificate and Declaration pages | **requires human action** | both pages blank; the PDF has no signature image and no digital signature | the three students sign the Declaration and date it; the guide and the Head of Department sign the Certificate (examiner and director if the department requires); scan the signed pages or sign digitally; produce the final PDF; regenerate MANIFEST.txt | to assign (all three students; guide; HoD) | contributions section finished first, so the signed version is the final text | High: "signed" is in the task title |
| Editable source | DOCX | complete | hash matches the manifest | re-export the PDF from it after edits | to assign | none | Low |
| Abstract, introduction | front matter; chapter 1 | complete | present | none | | | Low |
| 40-paper review and comparison | chapter 2, Table 2 | **incomplete** | one page; a six-row table by research area drawing on about 33 sources; 39 references in total; no per-paper comparison | write a review of at least 40 papers with a comparison table (authors, year, venue, task, method, data, metric and result as reported, limitation, relation to this project). Only papers marked READ, or PARTIAL for the parts read, may be cited (CLAUDE.md rule 6). docs/RESEARCH.md has 86 entries: 19 READ and 67 PARTIAL, of which 10 are bibliographic records only, so 76 entries have content that was at least partly read: enough for 40 without new reading | to assign | the members check each row against the source; then insert into the DOCX and rebuild | High: an explicit numeric requirement |
| Methodology, diagrams, design, implementation | chapters 3 to 6, figures 1 to 5 | complete | present | none | | | Low |
| Tests | section 8.4, Table 16 | complete | 713 passed, 2 skipped, 13 deselected (2026-10-29) | none | | | Low |
| Results | chapter 7 | complete | Tables 12 to 14 | none | | | Low |
| Discussion | no section titled Discussion | incomplete (minor) | material spread over chapters 7 and 9 | add a short "Discussion" section (or retitle 7.5 and 9 accordingly), using section 6.5 of docs/PROJECT_MASTER_LEARNING_GUIDE.md as the source | to assign | none | Low to medium if marked against a checklist |
| Conclusion, references, appendices | chapter 10, 39 references, Appendix A | complete | present | none | | | Low |
| All group members | cover, certificate, declaration | complete | three names and roll numbers (author order Shivansh Gupta, Vaibhav Shiroorkar, Adithya Jithesh, D-089) | none | | | Low |
| Each member's contribution | none | **missing** | no contribution text; deliverables_v2/SPEC.md line 29 says "no contribution statements" | after the interview and approval: a "Member contributions" section stating verified past contributions and approved ownership separately; log a decision that overrides SPEC.md line 29 | all three members | Phase 2 interview and approval | High: explicit requirement; must not be invented |
| College template points | Motivation subsection, end-of-chapter summaries, "Publications" section | incomplete (minor) | absent | add a Motivation paragraph to 1.1; add a "Publications" line with the DOI once it exists | to assign | DOI | Low |

## 2. Task 29: IEEE-format research paper with DOI proof

Artifact: `submission/Multimodal_Deepfake_Detection_Research_Paper.pdf` (16 pages: 8 college front-matter pages, then 8 IEEE pages) and `submission/sources/paper/` (main.tex with `\documentclass[conference,a4paper]{IEEEtran_nodash}`, refs.bib, figures).

| Requirement | Existing artifact | Status | Evidence | Exact remaining action | Owner | Dependencies | Risk |
|---|---|---|---|---|---|---|---|
| Separate IEEE-format paper PDF | paper PDF | complete, with one caveat | two-column IEEEtran conference, 8 pages including 32 references | for the preprint upload, produce an IEEE-only PDF (the 8 IEEE pages without the college front matter), from the same source | to assign | none | Medium: a preprint server expects the paper alone |
| DOI link or screenshot proving an assigned DOI | none | **missing; requires human action** | no DOI, Zenodo, arXiv, TechRxiv, SSRN or OSF record anywhere in the repository | (1) the three authors agree to publish the paper publicly, with the author order of D-089; (2) choose a server: Zenodo mints a DOI at once on publication; arXiv and TechRxiv need moderation (days); (3) one author uploads the IEEE-only PDF with all three authors, a title matching the paper, and a licence (for example CC BY 4.0); (4) after publishing, open https://doi.org/<DOI> to confirm it resolves; (5) save the DOI URL and a screenshot of the record page; (6) add the DOI to the report and README | to assign (the uploader needs an account) | author consent; IEEE-only PDF | High: hard requirement. A published Zenodo record cannot be deleted, only versioned; the DOI is permanent |
| Dataset terms before public posting | paper content | to check | the paper shows aggregate numbers and figures; no dataset frames are expected, but the figures should be checked before posting (Celeb-DF, Celeb-DF++, FakeAVCeleb forbid publishing derived data) | look at every figure of the IEEE pages before upload | to assign | none | Medium |

## 3. Task 30: viva defence slides

Artifact: `submission/sources/slides/Multimodal_Deepfake_Detection_Presentation.pptx` and `submission/Multimodal_Deepfake_Detection_Presentation.pdf` (25 slides: 14 main, a divider, 10 backup; speaker notes on all 25; rehearsal script submission/REHEARSAL.md: about 40 s per slide, 9 to 10 minutes plus a 3-minute demo).

| Requirement | Existing slide | Status | Exact remaining action | Owner | Dependencies | Risk |
|---|---|---|---|---|---|---|
| Editable and PDF | both | complete | regenerate the PDF after any edit | to assign | | Low |
| Problem | slide 2 | complete | none | | | Low |
| Literature gap | slide 2 has one context line; slide 13 has published numbers | **incomplete** | add one slide "Related work and the gap" (what published detectors report, why cross-dataset generalisation and shortcuts remain open, what this project adds) | to assign | the 40-paper review (for consistency) | Medium |
| Design, implementation | slides 3, 4, 10, 11 | complete | none | | | Low |
| Measured results, comparison, limitations | slides 6, 7, 8, 9, 13 | complete | none | | | Low |
| Member contributions | slide 1 lists names only | **missing** | add one slide with the approved contribution text (same as the report) | all three | interview and approval | High |
| Presentation length | REHEARSAL.md | stated | two added slides add about 1.5 minutes; confirm the allotted viva time with the department | all three | | Low |

Fastest legitimate path for slides: insert two slides into the existing deck (do not rebuild the deck from scratch; the build script needs the college template and a local screenshot, SETUP.md "Deck rebuild limit"). The proposed `submission/final/Viva_Defense_Slides.pptx` would duplicate the existing deck; a single final deck in `submission/` with an updated manifest is less confusing (decision for the team).

## 4. Task 31: final end-to-end demonstration

| Requirement | Existing artifact | Status | Exact remaining action | Owner | Dependencies | Risk |
|---|---|---|---|---|---|---|
| Final MP4 recording | none (no .mp4, .webm, .mkv, .mov or .avi in the repository outside the demo inputs) | **missing; requires human action** | record the screen following docs/FINAL_DEMO_RUNBOOK.md (setup check, start the v2 website in live mode, image, video with sound, audio, explanations, Evaluation Lab, test run, measured-results evidence), export MP4, record its SHA-256 | to assign (presenter and recorder) | runbook; a quiet machine; local checkpoint (present) | High: hard requirement |
| Complete project files | private GitHub repository at tag `submission-final`; submission/ | **incomplete; requires human action** | decide how examiners get the code: invite them to the private repository, or attach a source archive of the tag without data and checkpoints. Data and weights stay excluded (licences) and the README says why | Shivansh Gupta (repository owner) with the team | D-093 decision pending | Medium |
| Shows the final version, setup, input-to-output, measured performance | screenshots and JSON logs in docs/test_evidence/; docs/DEMO.md | partial | covered by the recording | | | Medium |
| Do not show restricted media | demo inputs are FF++ validation, LAV-DF train, ASVspoof dev and synthetic files (docs/DEMO.md section 2) | to respect | never open Celeb-DF, Celeb-DF++ or FakeAVCeleb files, or any sealed split, on camera | presenter | | High if breached |

## 5. Fastest legitimate path, in order

1. **Phase 2 interview and approval** of contributions (blocks the report section, the slide and the signatures).
2. **Contributions text** into the report and one slide.
3. **40-paper comparison table** from docs/RESEARCH.md entries with READ or PARTIAL content status; each member checks rows against the sources; insert into chapter 2; add the literature-gap slide.
4. **Discussion** section and the Motivation paragraph (small).
5. **Rebuild** the report PDF and the slide PDF from the edited sources.
6. **Signatures** on the final report PDF.
7. **IEEE-only PDF** and the **DOI** upload (after author consent); add the DOI to the report "Publications" line and README; screenshot.
8. **MP4 recording** with the runbook.
9. **Repository access** decision.
10. **Regenerate MANIFEST.txt** (scripts/eval/make_submission_manifest.py) and re-check hashes.

Steps 3, 4 and 7 (IEEE-only PDF) can be drafted by the assistant for the members to check; steps 1, 2, 6, 7 (upload), 8 and 9 need people.

## 6. Risks outside the four tasks

| Risk | Detail | Handling |
|---|---|---|
| Licences | most datasets are non-commercial research only; CLIP ViT-L/14 and WavLM research-only; SyncNet weights licence not stated; FakeAVCeleb form terms not recorded (D-092 parks these) | keep the research-only sentences; do not distribute weights or data |
| Authorship evidence | all commits are under one account and an AI agent did much of the implementation | contribution statements must separate verified past work from presentation ownership |
| Preregistration strength | E-019 to E-030 have weaker ordering evidence than earlier experiments | say so if asked (docs/RESEARCH_INTEGRITY.md) |
| Stale documents | PLAN.md (last updated 2026-10-25), the DATA_LOG DF26 row, port 8001 in two docs | listed in docs/GROUND_TRUTH_INVENTORY.md section 6; not in the submitted PDFs as far as checked |
| Submission day | the deadline recorded in the project is Fri 2026-10-30 (D-009) | do the human actions first |
