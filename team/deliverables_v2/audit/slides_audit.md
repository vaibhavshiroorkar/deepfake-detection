# Slide deck audit (hostile examiner), 2026-10-28

Deck: slides/DeepFake_Detection_Presentation.pptx, 58 slides, built by slides/build_slides.py. Analysis only; no repository file touched. Slide text read with python-pptx (full dump: audit/slides_dump.txt); slides 1, 12, 20, 23, 39 rendered to PNG via PowerPoint COM, read-only.

## STOP flag status: NO STOP

Every decimal figure (3 or 4 decimals, signed or not) printed on slides 1 to 58 was checked by script against docs/EXPERIMENTS.md: no figure is absent from it. Spot checks of the headline numbers against their E-ID rows are in section 4; all match exactly. Non-matching items found are not number mismatches but (a) a stale statement (slide 53), (b) a stale status line (slide 36), (c) missing "post hoc" labels (slides 31, 35), (d) one unverified count word ("two GANs", slide 32). Literature numbers on slide 2 (0.945, 0.772) are in RESEARCH.md R-017 as 0.9450 and 0.7718, not in EXPERIMENTS.md, correctly labelled "their protocol".

## 1. Current state

- 58 slides. Structure by content: 1 to 40 main body; 41 references; 42 "Backup slides" divider with contents; 43 to 58 backup (16 content slides). So "main" = 40 slides + references; backup = 16 + divider.
- Words on slide (all text boxes and table cells): total 9,095. Main 1 to 40: 6,028 (mean 150.7 per slide). Slide 41 (references): 421. Backup 42 to 58: 2,646.
- Speaker notes: total 12,109 words. Main 1 to 40: 9,056 (mean 226 per slide). Backup 42 to 58: 2,941. Longest notes: slide 34 (461), 14 (372), 35 (337), 32 (335), 12 (334), 15 (321). At about 130 spoken words per minute, notes alone for slides 1 to 40 would take about 70 minutes to read. The notes are a report, not a script.
- Dense slides (over 180 words on slide) in main: 4 (260), 9 (284), 10 (303), 11 (218), 15 (195), 20 (198), 21 (201), 39 (204).
- Figures and tables: 9 pictures in the whole deck (slides 1 x2, 12, 17, 22 x3 screenshots, 23, 27), 47 native tables, 0 native charts. Per slide counts are in the table below (W = words on slide, N = notes words, P = pictures, T = tables).

| # | W | N | P | T | | # | W | N | P | T | | # | W | N | P | T |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 53 | 197 | 2 | 0 | | 21 | 201 | 233 | 0 | 2 | | 41 | 421 | 112 | 0 | 0 |
| 2 | 175 | 177 | 0 | 0 | | 22 | 106 | 231 | 3 | 0 | | 42 | 99 | 101 | 0 | 0 |
| 3 | 142 | 119 | 0 | 0 | | 23 | 69 | 147 | 1 | 0 | | 43 | 140 | 159 | 0 | 0 |
| 4 | 260 | 128 | 0 | 1 | | 24 | 118 | 142 | 0 | 2 | | 44 | 88 | 311 | 0 | 0 |
| 5 | 125 | 164 | 0 | 0 | | 25 | 143 | 146 | 0 | 1 | | 45 | 139 | 167 | 0 | 1 |
| 6 | 113 | 175 | 0 | 0 | | 26 | 154 | 226 | 0 | 0 | | 46 | 185 | 123 | 0 | 1 |
| 7 | 169 | 155 | 0 | 0 | | 27 | 77 | 213 | 1 | 0 | | 47 | 263 | 162 | 0 | 2 |
| 8 | 171 | 118 | 0 | 1 | | 28 | 134 | 230 | 0 | 1 | | 48 | 294 | 180 | 0 | 1 |
| 9 | 284 | 238 | 0 | 1 | | 29 | 112 | 152 | 0 | 1 | | 49 | 174 | 167 | 0 | 2 |
| 10 | 303 | 195 | 0 | 3 | | 30 | 133 | 243 | 0 | 2 | | 50 | 179 | 182 | 0 | 1 |
| 11 | 218 | 202 | 0 | 1 | | 31 | 120 | 208 | 0 | 2 | | 51 | 101 | 174 | 0 | 1 |
| 12 | 110 | 334 | 1 | 0 | | 32 | 138 | 335 | 0 | 2 | | 52 | 138 | 155 | 0 | 1 |
| 13 | 179 | 217 | 0 | 1 | | 33 | 127 | 317 | 0 | 2 | | 53 | 167 | 197 | 0 | 1 |
| 14 | 162 | 372 | 0 | 1 | | 34 | 142 | 461 | 0 | 1 | | 54 | 146 | 264 | 0 | 1 |
| 15 | 195 | 321 | 0 | 1 | | 35 | 135 | 337 | 0 | 1 | | 55 | 201 | 158 | 0 | 1 |
| 16 | 156 | 225 | 0 | 1 | | 36 | 87 | 171 | 0 | 0 | | 56 | 105 | 120 | 0 | 1 |
| 17 | 58 | 248 | 1 | 0 | | 37 | 177 | 254 | 0 | 0 | | 57 | 153 | 210 | 0 | 1 |
| 18 | 173 | 270 | 0 | 0 | | 38 | 136 | 189 | 0 | 0 | | 58 | 74 | 111 | 0 | 1 |
| 19 | 168 | 294 | 0 | 1 | | 39 | 204 | 232 | 0 | 1 | | | | | | |
| 20 | 198 | 284 | 0 | 1 | | 40 | 103 | 156 | 0 | 0 | | | | | | |

Note: pictures counted for slides 12, 17, 23 are charts or schematics rendered as images; slide 20/21 are tables only. The deck has almost no visual argument: the one real chart is on slide 12 and a dot chart for LAV-DF on slide 17.

Structural observations (what an examiner sees):
- The sealed headline results sit on slides 20 and 21, after 19 slides of method and iterations; the project's point (comparable to a published detector on our own protocol) arrives at about two thirds of a 15 minute slot if each slide gets 40 seconds.
- Slide 22 shows the I-0 working version screenshots; slides 23 to 28 describe a later platform. A viewer sees two different products.
- Slide 53 (backup) states "The web page still runs the checkpoint of the working version; it has not been switched to the frozen system" and "304 unit tests pass". Slide 28 reports 551 passed, slide 25 describes the frozen modules in the platform, slide 26 a live demo. This is a stale statement inside the deck (contradiction), not a number mismatch against EXPERIMENTS.md. Fix or delete.
- Slide 36 (timeline) says "Status on 25 October" and shows the plan as of 24 to 30 October; the deck is otherwise current to 27 and 28 October. Stale.

## 2. Length limit

Searched: Document Template/DeepFake_Detection_System_Presentation (2).pptx (9 slides; no notes; titles: Introduction & Problem Statement, Objectives & Scope, Literature Review & Existing Systems, System Architecture & Methodology, Cross-Modal Streams: Lip-Sync & Emotion, Progress So Far, Challenges & Solutions, Remaining Work & Timeline), the reference report PDF (Document Template/Deepfake_detection_system_report_ (1).pdf, 41 pages; text search for minute, slide, presentation, duration: only unrelated hits), the Word template (Document Template/67F_B Tech Semester VII Project Report template.doc), docs/ (DECISIONS, STATUS, PLAN, DOC_SYNC, WORKSTREAMS, _agents/A.md).

- Slide count or talk duration requirement: NR. No document states one.
- Only presentation-related instruction found, in the Word template, instructions page: "Prepare presentation in the same flow as per the thesis" (Document Template/67F_B Tech Semester VII Project Report template.doc, instruction list). That constrains order, not length. The reference deck is the 9-slide progress deck (Introduction, Objectives, Literature, Architecture, Cross-modal, Progress, Challenges, Remaining work and timeline), a mid-term shape, not a limit.
- Internal targets exist: D-089 (docs/DECISIONS.md line 1171, 2026-10-28) sets "deck of 13 to 14 slides" for deliverables_v2, and deliverables_v2/SPEC.md line 37 says keep the original deck's template, fonts and colours. deliverables_v2/presentation/build_deck_v2.py exists (25 KB; its charts folder has 4 PNGs). These are the project's own targets (user-decided), not the college's.

Argument on 15 to 18 main slides for a 15 to 20 minute talk plus at most 10 backup slides:
- For: at 1 minute per slide (a normal ceiling for table-heavy slides) 15 to 18 slides fills 15 to 18 minutes and leaves 2 to 3 minutes for questions or demo. The present 40 main slides (150 words each, 3 to 5 tables with 4-decimal intervals) need about 55 to 60 minutes. Examiners punish read-aloud tables, and the user's own CLAUDE.md audience is a mathematics professor who will go to a backup slide on demand, so a backup deck is the right place for the sweep tables, the battery and the licences (already 16 slides). Backup at most 10 forces further merging of 43 to 58.
- Against: D-089 already decided 13 to 14 slides for v2, so 15 to 18 is above the user's own target; if the v2 deck (13 to 14) is the one to be presented, the right main count is 14, and a live demo (slide 26) takes 2 to 3 minutes by itself. Also the narrative has two stories (detector iterations plus the platform plus post-freeze experiments); 15 slides is tight for three stories. Recommendation: 16 main slides is defensible only if the post-freeze experiments are compressed to one or two slides; 14 is the safer count for 15 minutes, 17 or 18 for 20 minutes. The list in section 5 gives 17 main plus a backup of 10, and marks 3 slides as droppable to reach 14 (note on D-089 conflict: needs user ruling, tag decided autonomously - needs user review).

## 3. Section audit of every slide

Verdicts: CORE (stays in main, tell it), SUPPORTING (main if time, else one line), BACKUP, DELETE. Reasoning is one line each. Keep case and remove case are given in brief for every slide.

| # | Title | Strongest case to keep | Strongest case to remove | Verdict | Reason |
|---|---|---|---|---|---|
| 1 | Title | Identity, names, guide required | none | CORE | Required opening. |
| 2 | Problem and scope | States why protocols matter (0.945 vs 0.772 is a good hook) | Three paragraphs of text; aim text repeats slide 3 | CORE | Examiner needs the problem; trim aim. |
| 3 | Objectives | Reports chapter 1; professors check objectives against results | 8 objectives, 142 words, read-aloud | SUPPORTING | Merge with slide 2 as one "Problem, scope, objectives" slide. |
| 4 | Literature | Required by the thesis flow; ties each source to a choice | 260-word table, unreadable at the back | SUPPORTING | Keep one slide with trimmed table; full detail to backup (refs). |
| 5 | Protocol: group-disjoint splits | The core methodological claim; guards against leakage | none strong | CORE | Distinctive, defended by rule 3. |
| 6 | Method: file to calibrated score | The system in one picture | Overlaps slide 23 (platform picture) | CORE | The one pipeline diagram. |
| 7 | Method and the mathematics | Mathematically strong professor will ask loss, AUC, bootstrap | Formula dump, 169 words | SUPPORTING | Keep as one slide or move equations to backup; talk to it in 1 minute. |
| 8 | Data and splits | Shows datasets and sealed sets | Dense table; slide 5 already shows the logic | SUPPORTING | Keep with dev/sealed table; licences stay in backup 56. |
| 9 | Encoder shortlist | Shows the "why this encoder" criteria | 12-row table of params/GFLOPs, no result | BACKUP | Duplicated by 48 (cost record); a table of candidates before results is a skim. |
| 10 | Encoder sweep on dev slice | The "why DINOv2 small, CLIP, HuBERT" answer | 303 words, 3 tables; same data as 46 and 47 | SUPPORTING | One slide with the winner per task; full tables stay in backup 46 and 47. |
| 11 | I-0: working-version rule | MVP definition; rule and winners | Repeats 10 winners and 13 train/val/dev numbers | SUPPORTING | Merge with 10 as "Working version and encoder choice". |
| 12 | Iteration ladder | Only visual summary of the whole 7-iteration story; has a chart | Dense colour code | CORE | Best single slide of the iteration path; anchor of the talk. |
| 13 | I-0 diagnosis | Shows reasoning from diagnostics to levers (why each iteration) | Exploratory, 179 words; battery already in backup 49 and 50 | BACKUP | Reason for lever choice can be one sentence on slide 12; full battery backup. |
| 14 | I-1 fine-tuning | Biggest dev gain (+0.1700) and the baseline comparison | Exploratory dev numbers; superseded by sealed slide 20 | SUPPORTING | One of the iteration slides; keep, trimmed. |
| 15 | I-1b and I-1c | Failed or non-entering iterations; honesty | Two levers that did not enter, run at team request with triggers not met | BACKUP | Negative results belong in backup; one line on slide 12 already. |
| 16 | I-2 to I-4 speech | Clear stepwise gain 0.6100 to 0.8130 | Exploratory, 12 speakers | SUPPORTING | Keep as speech iteration slide; sealed is on 21. |
| 17 | I-5 and I-6 LAV-DF | Honest negative: fused result driven by pipeline cue | 58 words, the claim is "no audio-visual claim" | BACKUP | Important integrity result but not the main line; detail in 54 and 55 backup. |
| 18 | rPPG: test signal first | Mandated by objective 5; the plan and the check | The check failed; nearly a negative result | SUPPORTING | Merge with 19. |
| 19 | rPPG: positive control | Makes the failed check informative | Detailed; the outcome is "not adopted" | BACKUP | Keep one combined rPPG slide in main (18 + one line of 19); full control to backup. |
| 20 | Sealed tests: three comparisons | The headline: three pre-declared tests, exact match to E-014 | Not strong against baseline (superior at 95 only) | CORE | The result slide. |
| 21 | Sealed staircase | Shows effects per iteration and per set; honesty about Xception not lower on DFD | 201 words, 2 tables | SUPPORTING | Keep one table; rest to backup. |
| 22 | Web demo (real checkpoint) | Software deliverable evidence | Shows I-0 checkpoint, not final system; duplicated by 26 and 27 | DELETE | Superseded by platform dashboard slides; conflicts with current product. |
| 23 | Platform: architecture | Single picture of the platform; "no global score" design | Platform is a post-freeze extension, not the research question | SUPPORTING | Keep as the software slide. |
| 24 | Report contract | Explains statuses and findings; defends no fused score | Software detail, two tables | BACKUP | Detail for a viva question. |
| 25 | Implemented capabilities | Maps each module to its measured evidence (tie to E-IDs) | Repeats numbers from 20, 29 to 33 | SUPPORTING | Keep as the capabilities and evidence-status slide; it is the honest summary of the platform. |
| 26 | Demonstration workflow | Supports a live demo | Not a result; steps, not evidence | BACKUP | Live demo replaces it; keep the doc as backup. |
| 27 | Dashboard and explanations | Screenshot proves the software exists | Caption says "Screenshot, not a result"; before verdict labels (stale layout noted in CONTENT_MAP) | BACKUP | Stale screenshot; use one live screenshot on slide 23 or 25 instead. |
| 28 | Testing | Evidence for software claims (551 passed) | Software behaviour, not accuracy; process | BACKUP | One line on slide 25; detail backup. |
| 29 | After the freeze: FakeAVCeleb | Sealed once, pre-stated rules; supports independence of modalities | Its own limit: codec and method coincide, AUC may be inflated; sensitivity/specificity do not transfer | SUPPORTING | Keep with the limit in the same slide (already there). |
| 30 | Generated video and lip sync | Pre-registered, honest negative results; shows why module is experimental | Two topics on one slide; failed (chance) results | BACKUP | Experimental negatives; one line on slide 25 says it. |
| 31 | Modern generators, published numbers | Contextual comparison with literature, honest class-B wording | Mixed content (video heads, speech EER); confuses A/B comparability | SUPPORTING | Split: speech EER row to benchmark slide 35; video row to slide 30 backup. |
| 32 | Modern generated images | Pre-registered audit of own product on 21 modern generators; exposes a real weakness | Heavy detail; the story is "our image path misses 74 percent" | SUPPORTING | Strong honest-limit slide; keep one, merge with 33. |
| 33 | Retraining on recent generators | Shows the fix works for ranking and fails for threshold (52 percent real flagged) | Not installed; an unapproved head | SUPPORTING | Merge with 32 as one "Modern generated images: audit and retrain" slide. |
| 34 | Verdicts and the Evaluation Lab | Product wording and safeguards (D-083 to D-088) | 461 notes words; product design, not research | BACKUP | Detail for a viva; one line on 25. |
| 35 | Benchmark, limits, unapproved heads | Class A rows are the only head-to-head statements; limits | Mixes three topics | SUPPORTING | Keep as the comparison-with-published slide (the "comparable" claim). |
| 36 | Timeline to 30 October | Shows schedule fidelity | Process not result; status text stale ("Status on 25 October"); bars are the original plan | DELETE | Process slide with stale status; examiner gains nothing. |
| 37 | Limitations | Needed; dev slices small, one baseline family, failed predictions | 177 words, read-aloud list | CORE | Required honesty slide; merge with 38 (see below). |
| 38 | Limitations of the later work | Genuine integrity items (registration order not provable in git, codec confound, licences) | Second limitations slide; reads as confession in the main talk | SUPPORTING | Merge into 37: take 3 or 4 items (registration order, codec confound, licences, speech no check); rest backup. |
| 39 | Conclusion | Claim-outcome table with exploratory/sealed labels | 204 words; mixes dev and sealed numbers | CORE | Required; trim to sealed rows and one summary line. |
| 40 | Future work | Required by thesis flow | Generic list | SUPPORTING | Fold into conclusion slide or keep as 3 bullets. |
| 41 | References | Required; verified entries | 421 words | SUPPORTING | Keep at end, not counted in the talk. |
| 42 | Backup divider | Helpful signpost | Its contents list is stale vs the real backup list | SUPPORTING | Keep; regenerate contents. |
| 43 | Statistics and decision rules | Answers "why bootstrap, why 3 seeds" | none | BACKUP | Keep. |
| 44 | Staircase and confirmatory claims | Explains the 74-entry ledger and 3 declared tests | Graphic is thin; notes (311 words) hold the content | BACKUP | Keep. |
| 45 | Preprocessing | Answers a leakage/reproducibility question | none | BACKUP | Keep. |
| 46 | Sweep tables, face frames | Evidence behind slide 10 | none | BACKUP | Keep. |
| 47 | Sweep tables, images and speech | Same | none | BACKUP | Keep (merge 46 and 47 for the 10-slide limit). |
| 48 | Cost record and fine-tuning fit | Defends "cheapest wins" | Duplicates slide 9 | BACKUP | Keep; delete 9. |
| 49 | I-0 battery: metrics | Calibration and gap numbers | none | BACKUP | Merge with 50. |
| 50 | I-0 battery: subsets and errors | Error analysis | none | BACKUP | Merge with 49. |
| 51 | Parity check and page check | Reproducibility evidence | Minor | BACKUP | Fold into slide 28/45 backup, or drop (low priority). |
| 52 | Audio diagnosis | Shows why the speech gap was a shift not a bug | Minor | BACKUP | Keep as one backup if time. |
| 53 | Fine-tuning grid and checkpoint v2 | Grid shows learning rate sensitivity | Contains the stale "web page still runs the working version" and "304 unit tests" statements | BACKUP | Keep only after fixing the stale paragraph (contradicts slides 25, 28). |
| 54 | LAV-DF control and subset | Defence of the "no audio-visual claim" | Post hoc control | BACKUP | Merge with 55. |
| 55 | Sealed LAV-DF subset | Full result of the control | Exploratory | BACKUP | Merge with 54. |
| 56 | Data licences and terms | Rule 7 | none | BACKUP | Keep (licences needed in viva). |
| 57 | Celeb-DF++ test balance | Defends the 44-group interval and 9.7 percent real | Niche | BACKUP | Keep or fold into 8's note. |
| 58 | Environment (measured) | Rule 6 | none | BACKUP | Keep. |

Specific judgements asked for:
- Platform slides (23 to 28): of six, keep two in main (23 architecture, 25 capabilities with evidence status), push 24, 26, 27, 28 to backup. The platform is real software but a viva question is about the detector evidence, and 24, 26, 28 contain no result. Slide 27's screenshot predates the verdict labels (stale layout, see CONTENT_MAP.md), so it should be replaced if used.
- Post-freeze experiment slides (29 to 33): five slides of exploratory or pre-registered post-freeze experiments, all labelled; none changes the frozen system. Keep 29 (FakeAVCeleb, with its limit), merge 32 and 33 into one, put 30 and 31 in backup (31's speech EER row moves to 35). This shrinks five to two main slides.
- Limitations slides (37 and 38): consolidate to one main slide, four items from 38 (registration order, codec/method confound, licences, speech module has no speech check). Keeping two limitation slides in the main talk costs about 2 minutes of a 15 minute slot.
- rPPG slides (18 and 19): one main slide (18, with one line on the control); 19 to backup. The rPPG result is "not adopted, pulse not recoverable", worth 1 minute, because objective 5 requires it to be reported.
- Timeline slide (36): process, not result; stale. DELETE. If a timeline is required by the college flow (reference deck has "Remaining Work & Timeline"), regenerate it as "Plan versus done" only if the user asks.
- Duplicates: 9 and 48 (shortlist vs cost record); 10 and 46/47 (sweep winners vs full tables); 10 and 11 (winners); 13 and 49/50 (diagnosis vs battery); 22, 26, 27 (demo, three slides); 20, 21, 39 (sealed numbers shown three times); 37 and 38 (limits); 6 and 23 (two pipeline pictures, but they show different things, so keep both if time); 14 and 15 and 16 (iteration slides).

## 4. Claim audit

Method: each numeric claim compared with docs/EXPERIMENTS.md (row and E-ID). "Pre-declared" means the hypothesis, rule or test was written in the log before the run (Hard rule 2); "sealed" means a set read only under the final guard. Exact match = every digit equals the logged value. All numbers below match.

### Title slide (1)
No result, comparison or performance adjective. Subtitle "A Hypothesis-Driven Study Under a Group-Disjoint Protocol" is descriptive (note D-089 drops the subtitle in v2). KEEP.

### Problem and scope (2, 3)
| Claim | E-ID | Logged | Match | Status | Verdict |
|---|---|---|---|---|---|
| "plain Xception trained on FF++ c23 reached a mean AUC of 0.945 within the domain and 0.772 over eight other datasets" (Yan et al.) | not ours (R-017) | RESEARCH.md: 0.9450 / 0.7718, Table 3 | yes (rounded to 3 decimals, "their protocol") | literature | KEEP (labelled as theirs) |
| "published detectors are measured under different protocols, so their numbers cannot be read side by side" | none | argument | n/a | claim of method | KEEP |
| Aim: "a result comparable with published detectors on a protocol we define and run ourselves" | E-014 T3 | non-inferior to re-run Xception, margin 0.0100 | yes | pre-declared, sealed | WEAKEN: "comparable (non-inferior to one re-run published baseline, Xception at 224 px, on one sealed set)". The unqualified word comparable invites "to which detector". |
| Slide 3 obj. 6 "Define comparable as non-inferior to a re-run published detector" | E-014 | same | yes | pre-declared | KEEP |
| "No temporal localisation" | n/a | design | n/a | n/a | KEEP |

### Results slides (10 to 21, 29 to 33, 35)
| Slide | Claim | E-ID | Logged | Match | Pre-declared / post hoc | Exploratory / sealed | Verdict |
|---|---|---|---|---|---|---|---|
| 10 | Face dev AUC 0.7464 CLIP OpenAI, 0.6933 DINOv2 small; images 0.9935; speech 0.6100 HuBERT | E-001, E-002, E-003 | E-001 L335, E-002 L377, M-004 | yes | rule fixed beforehand | exploratory, dev slice | KEEP (labelled "Exploratory") |
| 11 | Working version dev AUC 0.693 / 0.994 / 0.610 | E-004 | E-004 L459 | yes | rule fixed beforehand | exploratory | KEEP |
| 12 | "Fine-tuning +0.1700 over the probe", "Speech 0.6100 to 0.8130" | E-005, E-009 | E-005 L526, E-009 L736 | yes | prediction logged (found larger than range, E-005 L533) | exploratory | KEEP; add "development slice" next to the numbers on the ladder text |
| 14 | SigLIP fine-tuned 0.9096 vs Xception 0.8716, +0.0380 [+0.0172, +0.0592] | E-005, E-013b | E-013b L944 | yes | criteria written before | exploratory, 15 identities | KEEP |
| 15 | I-1b 0.9315, I-1c 0.9192; neither enters | E-016, E-017 | L980, L1010 | yes | run on request, triggers not met (stated) | exploratory | KEEP; MOVE to backup |
| 16 | 0.6723, 0.7455, 0.8130 | E-007, E-008, E-009 | logged | yes | pre-declared | exploratory, 12 speakers | KEEP |
| 17 | 0.3838, 0.99 vs originals, 0.5070, 0.5169, 0.8574 | E-011, E-012 | L833 to L873 | yes | control is post hoc, stated on slide 54 | exploratory | KEEP; add "post hoc control" to slide 17 itself (currently only on 54) |
| 18 | rPPG check fails: 130 of 432 (0.301 [0.260, 0.346]) vs 0.6 | E-006 | L673 | yes | threshold written before | exploratory, no ground truth | KEEP |
| 19 | injection recovery 0.008 gives 0.873 etc. | M-005 | L1043 to L1056 | yes | requested in D-062 d, written before injected clip analysed | exploratory | KEEP |
| 20 | Video sealed 0.8218 vs 0.6238, +0.1980 [+0.1688, +0.2242], 98.33 percent [+0.1627, +0.2303]; speech 0.9148 vs 0.7516, +0.1633; baseline 0.8042, +0.0176 [+0.0025, +0.0327], 98.33 percent [-0.0011, +0.0355] | E-014 | L1160 to L1162 | yes | three tests declared before sealed read, 98.33 percent line written in amendment before sealed read (D-062 b) | sealed, one read per frozen iteration | KEEP. Verdict wording "improved, robust" fine; "superior at 95 percent; not robust at 98.33" matches the log wording. Add "one baseline family" in the headline cell (it is in the right box). |
| 21 | Staircase values (0.6238 to 0.8218, speech 0.7516 to 0.9148; ASVspoof 0.9835; images 1.0000, 0.9316, 0.9354) | E-014 | L1080 to L1135 | yes | only three rows are declared tests | sealed, exploratory except marked rows | KEEP (labelled "exploratory") |
| 29 | FakeAVCeleb sealed 0.9552, 0.9263, 0.5613, 0.5011; sens 0.480, spec 0.637; dev 0.9732 to 0.8772; codec-matched 0.8856 vs 0.8763 | E-015, E-021, E-023, E-025 | L1205 to L1207; E-021 L1265; E-025 L1409 | yes | H1 to H3 pre-stated rules supported; E-021/E-023 follow-ups on dev half only | sealed half once; dev half exploratory | KEEP. The adjective "supported" refers to rules written beforehand: OK. Add that the "audio specificity 0.637" row is at the frozen speech threshold (the slide implies it). |
| 30 | Generated video: C1 0.5073/0.5077, C2 0.8281/0.4957, C3 0.7613/0.5604; lip-sync 15.0 and 13.3 percent, 85 to 93 percent, 1.3 percent [0.4, 3.8], 0 of 160; "338 of 440 above threshold" | E-019/E-020, E-024, E-025, E-026 | E-019 L1380 to L1391; E-024; E-025 L1415; E-026 L1438 | yes (338 of 440 = 0.7682 is the installed head, E-026) | preregistered (E-026 header: before any clip scored; E-025 preregistered) | development / held-out; exploratory | MOVE to backup. Note: lip-sync "band of -6 to +2" was fitted on one pool and validated on a disjoint pool; fine. |
| 31 | Open-held-out 0.7453 [0.7174, 0.7710]; commercial 0.7548 [0.7189, 0.7885]; 0.8023 [0.7625, 0.8368]; speech EER 6.94 percent vs AASIST 1.13 percent; In-the-Wild 15.8 percent vs 33.9 and 7.2 | E-027, E-030 | E-027 L1470, L1480, L1488; E-030 L1510 | yes (6.94 = 0.0694 and 15.8 = 0.1578, rounded) | E-027 preregistered; E-030 is a **post hoc** metric (header says "post-hoc metric, no model run") | exploratory / sealed reuse | WEAKEN: add "post hoc, computed from stored scores" to the speech EER rows (slide 31 and 35 do not say it). "In-the-Wild 15.8 percent" vs "33.9 and 7.2 percent, full set" is class B, partially comparable (42 of 54 speakers); the footer says differences only where matched, good. |
| 32 | Macro AUC 0.7039 [0.6855, 0.7229]; missed 0.7395 [0.7300, 0.7500]; GAN 0.9580 / commercial 0.7351; PNG only 0.6123; face 0.7859 vs 0.6020, +0.1839 [+0.1596, +0.2074] | E-028 | L1533, L1534, L1537, L1538 | yes; "two GANs" count not verified in the log (log says "GAN 0.9580" per architecture) | preregistered, nothing fitted | exploratory, one look | KEEP; WEAKEN "two GANs" to "GAN architecture". Phrase "Inadequate by the line fixed in advance" matches a pre-declared line. |
| 33 | Validation 0.9823 [0.9662, 0.9969]; macro 0.8839 [0.8732, 0.8942]; +0.1800 [+0.1643, +0.1959]; +0.1675 [+0.1540, +0.1803]; flags real 0.5213 / 0.5885 / 0.0328; missed 0.0545 [0.0495, 0.0596] | E-029 | L1566 to L1571 | yes | preregistered before any feature extracted; H-29 supported | exploratory, scored once | KEEP; slide already says not usable, not installed. |
| 35 | EER 6.94 percent [5.95, 7.82] vs AASIST 1.13 and RawNet2 5.9 percent; zero-shot DF26 0.4957 vs nine detectors 44.0 to 69.7; Celeb-DF 0.8218 vs 0.8042 | E-030, E-020, E-014; PERFORMANCE_BENCHMARK | E-030 L1510 (0.0694 [0.0595, 0.0782]; slide rounds [5.95, 7.82] from 0.0595 and 0.0782: matches to 2 decimals); NUMBERS_LIT 44.0, 69.7 | yes | EER is post hoc; class A per PERFORMANCE_BENCHMARK | exploratory | WEAKEN: label EER "post hoc" and "class A (same test items), published model trained on LA train, ours AUC-selected"; the speech EER is worse than AASIST: said on slide (good). Zero-shot DF26 comparison: PERFORMANCE_BENCHMARK class C/D, not a difference; slide shows the two columns side by side: acceptable but mark class. |

### Conclusion (39)
| Claim | E-ID | Match | Verdict |
|---|---|---|---|
| "Group-disjoint protocol; sealed sets read once per frozen iteration; three tests declared beforehand" | E-014, D-062 b | yes (ledger 74 entries) | KEEP |
| "Face development AUC 0.7396 to 0.9096 by fine-tuning" | E-005 | yes (L526) | KEEP (exploratory label is on the heading) |
| "Speech 0.6100 to 0.8130 on 12 speakers" | E-009 | yes | KEEP |
| "Video and speech improved over the working version, at both levels." | E-014 T1, T2 | yes (lower bounds above zero at 95 and 98.33) | KEEP |
| "Superior to the re-run baseline at 95 percent only." | E-014 T3 | yes | KEEP; add "non-inferior at both". The word "superior" is the declared rule's wording but the robust statement is non-inferior; leading with "superior" will draw the question "after Bonferroni?". |
| "LAV-DF fusion driven by a pipeline cue." | E-012, E-014 | yes (0.7491 and fused from the cue "original or produced") | KEEP |
| "One finding per module, no global score; generated video experimental." | D-069, D-084 | yes | KEEP |
| Table row "Two more video levers I-1b and I-1c: 0.9315 and 0.9192; neither enters" | E-016, E-017 | yes | MOVE to backup (does not support the central claim) |
| Table row "rPPG: adopt or not" | E-006 | yes | KEEP |
| The conclusion has no sentence on the post-freeze findings (image 0.7039 missed 74 percent, video head chance 0.4957, FakeAVCeleb) | E-028, E-020, E-015 | n/a | WEAKEN omission: add one line "Post-freeze audits found the image and generated-video checks inadequate; they stay experimental". An examiner reading slides 29 to 33 expects the conclusion to say it. |

Other claim-wording flags (adjectives):
- Slide 20 verdict words "improved, robust", "superior", "non-inferior": used as the logged declared rules. KEEP, but "robust" should be glossed once ("survives the 98.33 percent Bonferroni interval").
- Slide 25 "sealed CNN subset 0.9316, diffusion set 0.9354" (E-014 exploratory, single frozen run, not a declared test): the cell says "sealed", add "exploratory".
- Slide 22 and 53 stale claims (see section 1). Slide 36 stale status line.
- No "real-time", "state of the art", "better than" found anywhere in slide text. (Searched the dump: none.)
- "Possible manipulation" label wording (slide 34) matches the D-087 addendum of 2026-10-28; "Likely fake" absent.

## 5. Proposed lists

### Main (17 slides; at 1 minute each, 15 to 18 minutes with a 2 minute live demo cut as needed)
| New # | Title | From existing |
|---|---|---|
| 1 | Title | 1 |
| 2 | Problem, scope and objectives | 2 and 3 (merged) |
| 3 | Literature and what we take from it | 4 (trimmed) |
| 4 | Protocol: group-disjoint splits and data | 5 and 8 (merged; table of splits trimmed) |
| 5 | Method: file to calibrated score | 6 |
| 6 | Method and the mathematics | 7 |
| 7 | Working version: encoder choice by rule | 10 and 11 (merged; winners only) |
| 8 | Iteration ladder | 12 |
| 9 | Video iteration: fine-tuning | 14 |
| 10 | Speech iterations | 16 |
| 11 | rPPG: tested and not adopted | 18 (one line of 19) |
| 12 | Sealed tests: three declared comparisons | 20 |
| 13 | Sealed staircase | 21 (one table) |
| 14 | Platform: architecture and capabilities with evidence status | 23 and 25 (merged) |
| 15 | After the freeze: FakeAVCeleb and modern image generators | 29 and 32/33 (merged to one; or two slides) |
| 16 | Comparison with published results (class A) and limitations | 35 and 37/38 (merged limitations) |
| 17 | Conclusion and future work | 39 and 40 |
Plus references (41) after the last slide, not counted in time. Expected main count: 17 (plus references). To reach 14 for D-089, drop 9, 13/21 table and 11: user decision.

### Backup (10)
| New # | Title | From existing |
|---|---|---|
| B1 | Statistics and decision rules; staircase ledger | 43, 44 |
| B2 | Preprocessing; parity and page check | 45, 51 |
| B3 | Encoder shortlist and cost record | 9, 48 |
| B4 | Sweep tables (face frames, images, speech) | 46, 47 |
| B5 | I-0 diagnosis and battery | 13, 49, 50, 52 |
| B6 | I-1b and I-1c; fine-tuning grid | 15, 53 (after fixing the stale paragraph) |
| B7 | LAV-DF control and sealed subset; rPPG positive control | 17, 54, 55, 19 |
| B8 | Generated video, lip sync and modern generators | 30, 31 |
| B9 | Platform detail: report contract, verdicts, Evaluation Lab, testing, demo workflow | 24, 26, 27, 28, 34 |
| B10 | Licences, Celeb-DF++ balance, environment | 56, 57, 58 |
Deleted outright: 22 (superseded I-0 page screenshots), 36 (stale timeline). Expected total: 17 main + references + backup divider + 10 backup = 29 slides (from 58). Merges that put two or more slides on one need layout work; tables of more than 120 words should be cut before merging, not shrunk.

Open items needing the user: (1) main count 14 (D-089) vs 17 here; (2) delete 36 and 22; (3) fix stale paragraph on slide 53 and status line on 36 if kept; (4) add "post hoc" to EER rows (31, 35); (5) revise conclusion to include one post-freeze sentence. All proposals tagged decided autonomously - needs user review. No repository file was edited.
