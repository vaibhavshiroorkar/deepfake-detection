# Paper audit: paper/main.tex and paper/main.pdf (hostile examiner pass)

Date of audit: 2026-10-28. Analysis only, no repository file was changed. Sources checked against: docs/EXPERIMENTS.md (the record), docs/DOC_FACTS.md, docs/PROTOCOL.md, docs/DECISIONS.md (D-049, D-087), docs/PERFORMANCE_BENCHMARK.md (where the paper cites it).

## STOP flag (scoped)

**STOP, scoped to one item. No logged number in the paper was found to differ from EXPERIMENTS.md** (every number of the abstract, Sections VII-A to VII-H, the conclusion and Tables V to XII was checked; rounded abstract values such as 0.693 [0.664, 0.731] round the logged 0.6933 [0.6639, 0.7314] correctly). **But one result pair is not in EXPERIMENTS.md at all**: Section VI-A, "computed afterwards from the stored sealed scores, the frozen threshold flagged about 0.11 of the real videos of the Celeb-DF family and 0.14 of those of DFD". Its only sources are DECISIONS.md D-087, PERFORMANCE_BENCHMARK.md 12.4 and VIVA_PREP.md, which itself says "not an experiment entry". Under CLAUDE.md rule 1 this is a measured result without an E-ID. Either log it (a post-hoc entry like E-030) or delete the sentence. The paper also cites "(Section VIII)" for these numbers, but Section VIII does not contain them.

Items that are not mismatches but that an examiner will catch (details in section 2):
- Section VII-G says DF26 was "read once and used for nothing else" (E-020). E-027 later trained heads on DF26's open generators and made a second counted look at its closed subset ("counted look 2; E-020 was look 1"). The paragraph after it also says "In one counted look". As written, the two sentences contradict the record.
- Section VII-H says the speech EER was "computed afterwards from the stored sealed scores". The E-030 correction of 2026-10-27 23:10 says the EER was already in the freeze report data/_final/eval/I-4.json, so it is not a post-hoc metric.
- Section VII-G puts 0.9732 to 0.8772 (E-021, Q3, 1,000 clips) next to 0.8856 against 0.8763 (E-023, B baseline). These are two logged values of what reads like the same canonical baseline (0.8772 and 0.8763), and the text does not say why they differ.
- Section VII-G says "the visual prediction (0.55 to 0.80) was wrong" for FakeAVCeleb. The audio prediction (0.40 to 0.90, observed 0.9263) was also wrong (E-015) and is not mentioned.

## 1. Current state

| Item | Value | How measured |
|---|---|---|
| Total pages | 31 (A4) | pdfinfo |
| Front matter | PDF pages 1 to 8 (8 pages): cover 1, certificate 2, declaration 3, acknowledgements 4, contents 5, list of figures 6, list of tables 7, abbreviations 8 | pdftotext per page |
| Main content | PDF pages 9 to 31 = paper pages 1 to 23 (offset 8): text and floats on PDF 9 to 30, references from about the middle of PDF 30 to the end of PDF 31 (about 1.3 pages, 42 entries) | headings found per page |
| Words, main content (pdftotext, PDF 9 to 30 up to REFERENCES) | 17,227 (includes table cells, captions and the college footers) | pdftotext, wc |
| Words, prose only (LaTeX source, floats removed) | 12,949, of which abstract and keywords 635 | script over main.tex |
| Words in floats (tables and captions) | 3,029 in 20 floats | same script |
| Words, references | 1,327 | pdftotext |
| Abstract | **625 words** (pdftotext). IEEE conference abstracts are normally 150 to 250 words, so this is 2.5 times the upper limit | pdftotext |
| Figures | 8 (Fig. 1 pipeline, 2 splits, 3 staircase, 4 ladder, 5 web screenshots, 6 platform, 7 ladder results, 8 LAV-DF cells), all full width (figure*) | main.lof |
| Tables | 12 (I to XII), 7 of them full width (table*) | main.lot |
| Numbered equations | 5 (clip mean, temperature scaling, AUC, ECE, non-inferiority), plus inline maths in Section V | main.tex |
| References | 42 bibitems, 66 \cite commands | main.bbl |

### Pages per section and subsection

The start page comes from the PDF (heading found by pdftotext). The length is an estimate: prose words from the source at about 1,000 words per two-column page, plus float areas measured on the rendered pages. The total comes to the 22.7 pages of PDF 9 to 31.

| Section | Starts (PDF page) | Approx. pages (prose + floats) | Prose words |
|---|---|---|---|
| Title, authors, abstract, keywords | 9 | 0.6 | 635 |
| I Introduction (with Contributions) | 9 | 0.6 | 450 |
| II Related Work | 10 | 0.85 | 693 |
| II-A Datasets for face forgery | 10 | 0.1 | 115 |
| II-B Visual detectors and generalisation | 10 | 0.2 | 183 |
| II-C Physiological signals | 10 | 0.15 | 152 |
| II-D Audio and audio-visual detection | 10 | 0.15 | 154 |
| II-E Frozen encoders, adapters and calibration | 11 | 0.1 | 89 |
| III Method | 11 | 1.3 | 929 |
| III-A Problem and output (+ Fig. 1) | 11 | 0.2 + 0.35 | 169 |
| III-B Preprocessing | 11 | 0.2 | 211 |
| III-C Frozen encoder and linear head | 11 | 0.2 | 204 |
| III-D Clip score, calibration and threshold | 11 | 0.15 | 115 |
| III-E Candidate encoders (+ Table I) | 11 | 0.1 + 0.3 | 96 |
| III-F Cost record | 12 | 0.06 | 54 |
| III-G Pulse evidence | 12 | 0.08 | 80 |
| IV Experimental Protocol | 12 | 2.3 | 1,399 |
| IV-A Setup | 13 | 0.05 | 52 |
| IV-B Data and splits (+ Fig. 2, Table II) | 13 | 0.25 + 0.65 | 230 |
| IV-C Celeb-DF++ de-duplication and test balance | 13 | 0.17 | 163 |
| IV-D Sealed tests and the staircase (+ Fig. 3) | 13 | 0.25 + 0.3 | 247 |
| IV-E Metrics and statistics | 13 | 0.55 | 520 |
| IV-F Seed policy and fine-tuning budget | 15 | 0.2 | 187 |
| V Iterations (+ Fig. 4, Tables III and IV) | 15 | 0.8 + 0.75 + 2.0 = 3.5 | 760 |
| VI Software (+ Fig. 5) | 16 | 0.55 + 0.35 | 525 |
| VI-A Analysis platform (+ Fig. 6) | 17 | 1.5 + 0.45 | 1,481 |
| VII Results (lead paragraph) | 21 | 0.1 | 114 |
| VII-A Encoder sweep (+ Table V) | 21 | 0.45 + 0.45 | 448 |
| VII-B Working version (+ Table VI) | 21 | 0.15 + 0.2 | 154 |
| VII-C Iterations, incl. re-run baseline (+ Table VII, Fig. 7) | 23 | 0.95 + 0.3 + 0.4 | 928 |
| VII-D LAV-DF, incl. pulse check (+ Table VIII, Fig. 8) | 23 | 0.65 + 0.1 + 0.45 | 649 |
| VII-E Staircase and confirmatory tests (+ Tables IX, X, XI) | 25 | 0.85 + 0.9 | 834 |
| VII-F Calibration and per-generator behaviour | 27 | 0.15 | 136 |
| VII-G Experiments after the freeze (+ Table XII) | 27 | 1.15 + 0.4 | 1,141 |
| VII-H Comparison with published results | 28 | 0.35 | 344 |
| VIII Limitations | 29 | 1.0 | 965 |
| IX Conclusion | 30 | 0.4 | 364 |
| References | 30 | 1.3 | 1,327 |

Observations from the rendered pages: PDF 17 to 20 are almost all floats (the Fig. 4 ladder takes most of PDF 17, Tables III and IV take PDF 18 and 19, and the screenshots take the top of PDF 20). Fig. 4 repeats Tables III, IV and VII. Software and the platform (VI, VI-A) take more space (about 2.8 pages) than the confirmatory results (VII-E, about 1.75 pages).

## 2. Claim audit

Columns: Loc. = location. E-ID = experiment. Logged = value in EXPERIMENTS.md. Match = exact match (rounding counted as a match when correct). Decl. = pre-declared (P), pre-declared lever with a post-hoc composite or reading (P/ph), or post hoc (ph). Data = dev (development slice or validation, exploratory), sealed-C (sealed and confirmatory), sealed-E (sealed but exploratory), post-freeze (registered after the freeze, own protocol). Verdict = KEEP, WEAKEN (with new wording), MOVE or DELETE.

### 2.1 Ranked by examiner damage (worst first)

| # | Loc. | Claim (short) | E-ID | Logged | Match | Decl. | Data | Damage reason | Verdict and new wording |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Abstract, VII-E, Conclusion, Limitations | T3: final video system "superior by the declared 95 % rule", +0.0176 [+0.0025, +0.0327]; 98.33 %: [-0.0011, +0.0355] | E-014 | same | yes | P (H2 superiority as the stronger statement; delta amended to 0.0100 before the read, D-066) | sealed-C | Multiplicity: it does not survive Bonferroni. One baseline family at 224 px whose learning rate was again at the edge of its grid. The margin was set by its floor. Leading with "superior" invites the "you chose the rule that gives the word" attack | WEAKEN: "non-inferior to the re-run Xception baseline with a margin of 0.0100 at both levels; the 95 % interval of the difference, +0.0176 [+0.0025, +0.0327], excludes zero but the 98.33 % interval [-0.0011, +0.0355] does not, so the two are not distinguished after correction." Put non-inferiority first everywhere, and never write "superior" without that sentence |
| 2 | Abstract, Conclusion | "fine-tuning the image encoder raised the AUC on Celeb-DF v2 identities from 0.693 [0.664, 0.731] to 0.910 [0.878, 0.939]" | E-004, E-005 | 0.6933 [0.6639, 0.7314] (DINOv2 small probe), 0.9096 [0.8776, 0.9388] (SigLIP fine-tuned) | yes | P | dev | Development-slice number in the abstract. The comparison changes the encoder (DINOv2 small to SigLIP base) as well as the training, so "fine-tuning raised" claims more than the run shows. The same-encoder effect is 0.7396 to 0.9096 | MOVE out of the abstract. In the body: "fine-tuning SigLIP base raised its development AUC from 0.7396 (probe) to 0.9096, +0.1700 [+0.1228, +0.2114] (exploratory, 15 identity groups)" |
| 3 | Abstract, Conclusion | FakeAVCeleb: "the frozen system ranks well (visual AUC 0.9552, audio 0.9263)" | E-015 (with E-021, E-023) | 0.9552 [0.9442, 0.9657], 0.9263 [0.9149, 0.9366] | yes | P (registered post freeze) | post-freeze sealed half | The codec alone separates real from fake video (metadata model 0.9985 on the development half, E-021). Codec and method are fully confounded (E-023). "Ranks well" is a performance adjective on a data set with a known shortcut | WEAKEN or DELETE from the abstract. Body: "the pre-stated ranking criteria were met (0.9552, 0.9263), but codec and manipulation method coincide in this data set, so the number is not evidence of manipulation detection; at the frozen thresholds visual sensitivity was 0.480 and audio specificity 0.637" |
| 4 | Abstract | "On LAV-DF, a fused score trained in the same dataset reaches an AUC of 0.99" | E-011 | 0.9935 [0.9905, 0.9962] | yes | P (mechanical success); the control that undoes it is ph | dev (LAV-DF val) | Known shortcut: the visual head separates original from produced videos (0.9912), and the fusion gain is +0.0023. Putting 0.99 in the abstract and then retracting it wastes the reader's first impression | DELETE the 0.99 from the abstract. Keep the negative: "the visual head is at chance within the produced videos (0.5070 on validation, 0.5032 on the sealed subset), so no audio-visual detection is claimed" |
| 5 | Abstract, VII-E, Conclusion | T2: speech 0.9148 [0.8896, 0.9383] against 0.7516 [0.6802, 0.8035], +0.1633 [+0.1177, +0.2266], robust | E-014 | same | yes | P | sealed-C | 42 speakers. Sealed values are above the development values (0.8130 and 0.6100) and both speech predictions failed upward, with no explanation given. Single-feature recording cues reverse sign between corpora (leading silence, RMS), and RMS still reaches 0.135 on In-the-Wild after trimming (E-007). An examiner will ask whether T2 measures removal of a shortcut | KEEP the number. WEAKEN the reading: "improved by the declared rule and at the Bonferroni level, on 42 speakers; the values lie above both the development values and the predicted range, and a recording-level cue remained after trimming, so how much of the gain is speech evidence is not established" |
| 6 | Abstract, Conclusion | speech changes "raised the AUC on In-the-Wild speakers from 0.610 to 0.813, with wide intervals from 12 speakers" | E-003, E-007 to E-009 | 0.6100 [0.5434, 0.7256], 0.8130 [0.7523, 0.8808] | yes | P/ph (each step declared; the composite I-0 to I-4 difference was never a declared test) | dev | Development-slice number in the abstract, 12 speakers | MOVE to the body (it is already in VII-C and Table VII) |
| 7 | VI-A | frozen threshold "flagged about 0.11 of the real videos of the Celeb-DF family and 0.14 of those of DFD" | none (D-087, PERFORMANCE_BENCHMARK 12.4) | not in EXPERIMENTS.md | n/a | ph | sealed-E | Rule 1: a measured result without an E-ID. The cross-reference to Section VIII is wrong | DELETE, or log it as a post-hoc entry first and then keep it, labelled post hoc |
| 8 | VII-H | speech EER 6.94 % [5.95, 7.82] "against 1.13 % for AASIST, 5.81 points worse"; "computed afterwards" | E-030 | 0.0694 [0.0595, 0.0782] | yes | ph (but see the E-030 correction) | sealed-E | Computes a difference against a quoted number with no interval and from a different training run, although the paper's own rule says only directly comparable rows get a difference. "Computed afterwards" is outdated (the EER is in the freeze report) | WEAKEN: "the EER of the frozen speech system on the official evaluation partition is 6.94 % [5.95, 7.82] (freeze report); AASIST reports 1.13 % under its own training run, so ours is higher in point estimate; no tested difference" |
| 9 | VII-G | DF26 "read once and used for nothing else"; E-027 "In one counted look" | E-020, E-027 | 0.4957 [0.4755, 0.5168]; 0.7548 [0.7189, 0.7885] | yes | P | post-freeze held out | Wording contradicts the record (DF26 open generators used for training in E-027; closed subset was look 2) | WEAKEN: "read once in E-020 for the DeepAction-trained head; E-027 later trained on the open generators of DF26 and made the second counted look, at its closed subset only" |
| 10 | VII-A | T-I: leader "separated from every other candidate by the paired bootstrap" (next -0.037 [-0.204, -0.001]) | E-002 | same | yes | P | dev (2 groups) | A bootstrap over 2 groups is not a population interval. The paper says so, then uses it anyway | WEAKEN: "leads on the two development units; with two groups the paired interval does not support a separation" |
| 11 | Abstract, VII-E, Limitations | "344 groups" as the independent units of the Celeb-DF family test | E-014 | 7,650 / 344 | yes | n/a | sealed | 300 of the 344 groups are single real YouTube videos; all fakes come from 44 identity groups. The fake side of the interval rests on 44 clusters. The Celeb-DF v2 only row also shows 344 groups | WEAKEN: "344 groups (44 identity groups that carry every fake, and 300 real YouTube videos)" in the abstract or the table caption |
| 12 | VII-A | T-V hypothesis "supported on this slice", best non-CNN minus best CNN +0.0994 [+0.0228, +0.1723] | E-001 | same | yes | P | dev | 7 candidates and 15 groups, and the permutation p for EfficientNet-B4 is 0.110. Supported only for the declared contrast | KEEP, adding "exploratory; the permutation check does not reach 0.05 for EfficientNet-B4" next to the verdict (it is already a sentence earlier) |
| 13 | VII-G | FakeAVCeleb: "the visual prediction (0.55 to 0.80) was wrong" | E-015 | H2 prediction 0.40 to 0.90, observed 0.926 | partial | P | post-freeze | Incomplete: the audio prediction also failed | WEAKEN: "both predictions were wrong (visual 0.55 to 0.80, audio 0.40 to 0.90)" |
| 14 | VII-G | "a common re-encode lowers the frozen visual AUC from 0.9732 to 0.8772, and a codec-matched control did not lower it further (0.8856 against 0.8763)" | E-021, E-023 | 0.9732 [0.9642, 0.9816] to 0.8772 [0.8562, 0.8981]; P1 0.8856, B 0.8763 | yes | P | post-freeze dev half | Two values for what reads as one canonical baseline | WEAKEN: name the bases ("on 1,000 clips" and "on the E-023 sample") or quote only E-023 |
| 15 | VII-H | DF26: our head 0.4957 "lies within the range of FF++-trained detectors reported by its authors (44.0 to 69.7)" | E-020; R-083 | 0.4957 | yes | ph comparison | cross-protocol | Different training data (DeepAction against FF++), so this is not a like-for-like range. Low damage because it is a negative result | MOVE to the report, or keep as one context clause |
| 16 | VII-E | DFD: final 0.8797 against baseline 0.8847, "a difference that is small against the interval of either" | E-014 | same | yes | ph reading | sealed-E | No paired test was run, and the sentence reads like an inference | WEAKEN: "point estimates 0.8797 and 0.8847; no paired test was declared" |
| 17 | VII-D | fusion "higher than the better single head by +0.0023 [+0.0009, +0.0040]", "mechanical success" | E-011 | same | yes | P | dev | Gain on a shortcut, but the paper labels it and gives the control | KEEP |
| 18 | VII-C | I-1c post hoc two-member average "lower bound of +0.0009 ... does not change the verdict" | E-017 | +0.0083 [+0.0009, +0.0176] | yes | ph, labelled | dev | Honest disclosure | KEEP (shorten in a cut) |
| 19 | Intro | DeepfakeBench Xception 0.9450 within and 0.7718 cross domain | R-017 (quoted) | quoted from the paper, own protocol named | n/a | n/a | literature | Context only. Correctly labelled with its protocol | KEEP |
| 20 | Abstract, VII-E | "Two of the five written predictions, both for speech, failed in the optimistic direction" | E-014 | same | yes | P | sealed | Good practice; "five" counts the E-014 predictions only (E-005, E-009 and E-015 also missed their ranges) | KEEP in the body: "two of the five predictions written for the sealed run". Drop from the abstract for space |
| 21 | Abstract, VII-D, Conclusion | pulse check failed: 130 of 432 (0.301 [0.260, 0.346]), 0.6 required | E-006 (+ M-005) | same | yes | P | dev | Negative result. Sound | KEEP (one sentence) |
| 22 | VII-E, Table IX | T1: +0.1980 [+0.1688, +0.2242], 98.33 % [+0.1627, +0.2303], improved and robust | E-014 | same | yes | P | sealed-C | The strongest claim. It confounds the encoder switch with fine-tuning, which the paper discloses (Xception minus probe +0.1783 on dev) | KEEP, adding "the final system differs from the working version in encoder and in training" |
| 23 | VII-E | sealed video below dev, speech above dev | E-014 | 0.8218 vs 0.9096; 0.8042 vs 0.8716; 0.9148 vs 0.8130; 0.7516 vs 0.6100 | yes | ph reading | sealed-E | Honest. Strengthens credibility | KEEP |
| 24 | VII-E, Table X | staircase values (Celeb-DF v2 only 0.8999, Celeb-DF++ with v2 real 0.7105, DFD, FF++, ASVspoof, image sets) | E-014 | same | yes | P (staircase declared), values exploratory | sealed-E | Labelled exploratory. Fine | KEEP (compact) |
| 25 | VII-E, Table XI | LAV-DF sealed: 0.3226, 0.9921, 0.9945, 0.5032, 0.5203, 0.8480 | E-014 | same | yes | P scoring, readings exploratory | sealed-E | Supports the main negative result | KEEP (merge with Table VIII) |
| 26 | VII-A to VII-C, Tables V to VII | every sweep, working-version and iteration number (E-001 to E-013b, E-016, E-017) | E-001 to E-017 | checked one by one | yes | P | dev | Labelled exploratory | KEEP in compact form |
| 27 | VII-F | calibration: ECE 0.2048 to 0.2217, 0.3293 to 0.4052, 0.0462 to 0.0050; after fine-tuning 0.0522 to 0.0331; speech average 0.0692 to 0.1670 | E-004, E-005, E-009 | same | yes | P (battery item) | dev | Fine | KEEP (two sentences) |
| 28 | VII-G | E-028: macro AUC 0.7039 [0.6855, 0.7229], pooled FNR 0.7395, format shortcut 0.9222, PNG-only 0.6123, face-image +0.1839, misses 80.6 % (logged FNR 0.8063) | E-028 | same | yes | P | post-freeze | Format shortcut partly controlled; a negative result for the image headline | KEEP the FNR 0.7395 and the macro AUC in a one-paragraph summary; MOVE the rest |
| 29 | VII-G, VII-H | E-029 retraining: 0.8839, +0.1800 [+0.1643, +0.1959], subgroup gains (+0.32, +0.12, -0.01, +0.17 [+0.15, +0.18]); false-positive rate 0.5213; IMG-15 0.9142 | E-029 | same (0.1675 [0.1540, 0.1803] rounds to the quoted +0.17 [+0.15, +0.18]) | yes | P (subgroups partly ph) | post-freeze | Off the core story; the subgroup reading is a multiplicity risk | MOVE to the report |
| 30 | VII-G | E-026, E-027 (0.7682, 0.7453, 0.7548, 0.8023, -0.0404) | E-026, E-027 | same | yes | P | post-freeze | Off the core story (generated video is not one of the three tasks) | MOVE to the report |
| 31 | VII-G | lip sync 15.0 % [8.1, 26.1], 13.3 % [6.9, 24.2], band 1.3 % [0.4, 3.8], 0 of 160 | E-024, E-025 | same | yes | P | post-freeze dev | Platform module, not the main result | MOVE to the report |
| 32 | VI, VI-A | parity 0.0019 on 11 files; 710 passed, 2 skipped; 71 pass; 304 unit tests; Evaluation Lab reproduced 0.8218 | none (TESTING_RECORD.md, E-004 parity) | software checks | n/a | n/a | engineering | Not detection results, so rule 1 is not strictly at issue, but no E-ID | MOVE to the report (keep the one sentence "a parity check against training-time scores passed, largest difference 0.0019") |

### 2.2 What the ranking means

- Damage classes, in order: (a) multiplicity and the single baseline (row 1); (b) development-slice numbers in the abstract (rows 2, 6); (c) known shortcuts quoted as performance (rows 3, 4, 5); (d) unregistered or mis-described numbers (rows 7, 8, 9); (e) small-cluster inference (rows 10, 11, 12).
- Nothing in the confirmatory set (T1, T2, T3) is numerically wrong. The risk is the wording around T3 and the reading of T2.
- The abstract currently carries 10 development-slice or shortcut-affected numbers (0.693, 0.664, 0.731, 0.910, 0.878, 0.939, 0.872, 0.610, 0.813, 0.99), and also 0.507, 0.9552, 0.9263 and 0.4957. The proposed abstract in section 5 keeps only sealed numbers plus clearly labelled exploratory or post-freeze ones.

## 3. Section audit

Question per row: if this part were deleted, would the examiner lose anything needed to judge the main result? The core story is the protocol, the sweep, the ladder, the sealed staircase, the confirmatory comparisons, the main limitations, and every negative result the conclusion relies on. Target pages are for the 8-page plan of section 4.

| Section | Pages now | Strongest case to keep | Strongest case to remove | Verdict | Target pages |
|---|---|---|---|---|---|
| Abstract | 0.6 | Required | 625 words, 13 development or shortcut numbers | CORE, rewrite (section 5) | 0.35 |
| I Introduction + Contributions | 0.6 | Motivation; DeepfakeBench gap justifies the cross-dataset headline | Contribution 4 (web page, platform) is not research; the road-map paragraph is filler | CORE, shortened; contribution 4 cut to one clause | 0.5 |
| II-A to II-E Related work | 0.85 | Shows the choices (identity leakage, frozen CLIP features, In-the-Wild degradation, pulse caution) were informed | Five subsections for a conference paper; II-C pulse and II-E calibration overlap with the method | SUPPORTING (short): three paragraphs (face forgery and generalisation; audio and audio-visual; pulse and frozen encoders) | 0.55 |
| III-A Problem and output + Fig. 1 | 0.55 | Defines what is claimed (separate scores, no fusion, calibration scope) | Fig. 1 is full width for a simple pipeline | CORE; Fig. 1 as single column | 0.35 |
| III-B Preprocessing | 0.2 | Needed to reproduce (16 frames, YuNet, 224 px, 64,600 samples, trim rule) | Parameter detail (1.3 enlargement, JPEG 95, Whisper window) | SUPPORTING (short): two sentences; parameters to the report | 0.1 |
| III-C, III-D Head, clip score, calibration, threshold (eq. 1, 2) | 0.35 | The method of the working version, and threshold-on-validation matters for the transfer failures | Optimiser detail | CORE (compact) | 0.25 |
| III-E Candidate encoders + Table I | 0.4 | Families explain the sweep hypothesis | Table I repeats the rows of Table V | APPENDIX for Table I; one sentence listing the families stays | 0.05 |
| III-F Cost record | 0.06 | Cost decides ties (cheapest wins) | Definition fits in one clause of IV-E | APPENDIX (one clause in IV-E) | 0 |
| III-G Pulse evidence | 0.08 | Describes I-7 | Duplicated in V and VII-D | Merge into the I-7 row | 0 |
| IV-A Setup | 0.05 | Hardware explains the budget | Tiny | SUPPORTING (one sentence) | 0.05 |
| IV-B Data and splits + Fig. 2 + Table II | 0.9 | The group-disjoint protocol is the paper's main contribution | Fig. 2 is a schematic that Table II already conveys | CORE: keep Table II (compressed); merge Figs. 2 and 3 into one single-column figure | 0.5 |
| IV-C Celeb-DF++ de-duplication arithmetic | 0.17 | Explains why the family test has 744 real videos, 344 groups and a mixture AUC | The 2,079 and 150 arithmetic is bookkeeping | SUPPORTING (short): one sentence ("development identities and byte-identical duplicates removed; 744 real, 6,906 fake, fakes in 44 identity groups"); arithmetic to the report | 0.07 |
| IV-D Sealed tests and staircase + Fig. 3 | 0.55 | Guard, ledger, three declared tests and the Bonferroni line are essential to judge T1 to T3 | Fig. 3 is a schematic | CORE (figure merged with Fig. 2) | 0.3 |
| IV-E Metrics and statistics (eq. 3 to 5) | 0.55 | Paired group bootstrap, tie set, non-inferiority and the margin are needed to judge every verdict | ECE equation, the list of secondary metrics, permutation detail | CORE (drop the ECE equation; keep AUC, non-inferiority, margin) | 0.45 |
| IV-F Seed policy and fine-tuning budget | 0.2 | Explains one seed per learning rate and the 10 GiB fit rule | Detail of the schedule | SUPPORTING (short): two sentences | 0.1 |
| V Iterations prose | 0.8 | Explains why each lever was chosen (battery to lever) | Long; repeats the tables | CORE, cut to half | 0.4 |
| V Fig. 4 ladder | 0.75 | Visual overview | Fully redundant with Tables III, IV and VII | DELETE (keep in the slides and the report) | 0 |
| V Tables III and IV (hypotheses and outcomes) | 2.0 | Shows the hypotheses were written first: the core of the method | Two full-width pages of text | CORE, compressed: one table with lever, success line (short) and outcome, one row per slot; full text to the report | 0.6 |
| VI Software + Fig. 5 screenshots | 0.9 | Shows a working product (a deliverable of the project) | Not evidence for any result; the screenshots are unreadable at that size and show training items | APPENDIX (technical report); one sentence in the conclusion | 0 |
| VI-A Analysis platform + Fig. 6 | 1.95 | The verdict-wording rules respond to measured threshold failures | No result depends on it; it holds the unlogged 0.11 and 0.14; longest subsection of the paper | APPENDIX | 0 |
| VII lead paragraph | 0.1 | States exploratory against confirmatory | n/a | CORE | 0.05 |
| VII-A Encoder sweep + Table V | 0.9 | Selection rule in action; the T-A inconclusive verdict motivates I-2 to I-4 | Full candidate table, 19 rows | CORE, with Table V compressed (T-V all rows, T-I top 2 and CNNs, T-A all rows; drop the validation column or merge it) | 0.6 |
| VII-B Working version + Table VI | 0.35 | Diagnostic gaps (0.105, 0.384) justify the levers | EER and ECE columns repeat VII-F | SUPPORTING: merge the numbers into Table VII as the I-0 rows; drop Table VI | 0.1 |
| VII-C Iterations (I-1, I-2 to I-4, I-1b, I-1c, re-run baseline) + Table VII + Fig. 7 | 1.65 | The ladder's measured effects; the baseline's tuning history matters for T3 | Fig. 7 repeats Table VII | CORE: keep Table VII, drop Fig. 7, compress the I-1 stage-1 detail and the I-1b and I-1c detail | 0.8 |
| VII-D LAV-DF + Table VIII + Fig. 8 | 1.2 | The main negative result (no audio-visual claim) | Fig. 8 repeats numbers in the text | CORE: merge Tables VIII and XI into one table (validation and sealed); drop Fig. 8 | 0.5 |
| VII-D Pulse check and positive control | 0.3 | Negative result the conclusion relies on | Control detail | CORE (S-1, 4 lines) and SUPPORTING (M-005, one sentence) | 0.12 |
| VII-E Staircase and confirmatory + Tables IX, X | 1.5 | The main result | Table X repeats I-1 values across I-2 to I-4 | CORE: Table IX kept; Table X reduced to I-0, I-1 (video), I-0, I-2, I-3, I-4 (speech), image row once, baseline column | 1.0 |
| VII-E Table XI LAV-DF sealed | 0.25 | Sealed confirmation of the negative | n/a | CORE (merged with Table VIII above) | counted above |
| VII-F Calibration | 0.15 | Explains why the scores are not probabilities off-validation | Small | SUPPORTING (two sentences) | 0.1 |
| VII-G Experiments after the freeze + Table XII | 1.55 | FakeAVCeleb and E-028 are the negative results the conclusion relies on (thresholds do not transfer; image misses modern generators) | Generated video, lip sync and retraining are off the three tasks; many registration subtleties | APPENDIX, with the one-paragraph replacement below | 0.2 |
| VII-H Comparison with published results | 0.35 | Examiner will ask "how do you compare with the literature" | Only the Xception re-run is directly comparable, and it is already in VII-E; the EER difference breaks the paper's own rule | SUPPORTING (short): three sentences (the rule; the Xception re-run is the only directly comparable row; the EER is given as context) | 0.15 |
| VIII Limitations | 1.0 | The main limits are core | 14 bullets, half of them about the platform and post-freeze work | CORE, cut to 6 bullets | 0.45 |
| IX Conclusion | 0.4 | Required | Repeats the abstract with dev numbers | CORE, rewritten without dev numbers in the first sentences | 0.25 |
| References | 1.3 | Required | 42 entries; several only support moved sections (C2PA, ReStraV, DeepAction, DF26, Community Forensics, SyncNet, YuNet) | Keep about 30 | about 1.0 (outside the count) |

### Replacement paragraphs for removed subsections that later text depends on

**Replacement for VII-G (Experiments after the freeze), to sit at the end of Results, about 0.2 page:**
"After the freeze, further tests were registered and run without changing the frozen system; the technical report gives them in full. On the sealed identity half of FakeAVCeleb (5,013 clips) the pre-stated ranking criteria were met (visual AUC 0.9552 [0.9442, 0.9657], audio 0.9263 [0.9149, 0.9366]), but codec and manipulation method coincide in that data set (a metadata model separates real from fake video with AUC 0.9985 on its development half), so these numbers are not evidence of manipulation detection, and the frozen thresholds did not transfer (visual sensitivity 0.480 at specificity 0.996, audio specificity 0.637). The generated-image probe, unchanged, reached a macro AUC of 0.7039 [0.6855, 0.7229] over 21 recent generators and missed 0.7395 of their images at its threshold, part of its separation following the file format. A generated-video head trained on DeepAction was at chance on DF26 (0.4957 [0.4755, 0.5168]) and was not installed."

**Replacement for VI and VI-A (Software and platform), one sentence in the conclusion or at the end of III-A:**
"The frozen detectors are served by a local web page and a modular analysis platform that report one score per modality, with its tested scope and no overall verdict; parity with the training-time scores was checked (largest difference 0.0019 on 11 validation items), and the design and tests are in the technical report."

**Replacement for III-F and IV-C (cost record, de-duplication), inside IV-B and IV-E:** "Cost is the end-to-end extraction time per 1,000 inputs, then the parameter count." and "Celeb-DF++ samples of the development identities and byte-identical duplicates of Celeb-DF v2 were removed, leaving 744 real and 6,906 fake videos in 344 groups (fakes in 44 identity groups)."

Limitations then needs only one bullet for the post-freeze work: "Post-freeze registrations entered version control together with their scripts, so their order rests on the project's own record."

## 4. Cut plan

### Target and what counts

- **Argument for 6 to 8 pages:** the IEEE conference format (IEEEtran conference, two columns) is normally 6 pages with paid extra pages up to 8, plus references. The current 23 main pages read as a report in IEEE dress; an examiner will see a 625-word abstract and two pages of screenshots and platform text as a sign that the authors cannot separate evidence from engineering. The project already has a technical report (deliverable 3), so nothing has to be lost: it moves there.
- **Why 8, not 6:** three modalities, a written protocol, a ladder of hypotheses and a negative audio-visual result. At 6 pages the hypothesis table or the LAV-DF negative would have to go, and both are core. 8 pages keeps every core element with compact tables.
- **College front matter:** keep it, but outside the count. The main.tex switch (`\collegestyletrue` or `\collegestylefalse`) already separates it. State explicitly: "page limit = main content from the title to the end of the conclusion; front matter (cover, certificate, declaration, acknowledgements, contents, lists, abbreviations) and references are not counted." Submit both builds: the college build for the department and the plain IEEEtran build for any venue. The front matter could also shrink by one page: the list of figures and the list of tables (PDF 6 and 7) will each drop to a few lines after the cut and can share a page.
- **Expected page count after the cut:** about 8.0 pages of main content plus about 1.0 page of references, so 9 PDF pages in the plain build and about 16 to 17 in the college build (8 front matter pages, possibly 7).

### Page budget

| Section (new numbering) | Content | Budget |
|---|---|---|
| Title, abstract (248 words), keywords | Section 5 | 0.35 |
| I Introduction | Problem, protocol gap (DeepfakeBench numbers), 3 contributions (protocol; selection rule and ladder; sealed confirmatory evaluation with negative results) | 0.5 |
| II Related work | Three paragraphs | 0.55 |
| III Method | Output definition, preprocessing in two sentences, frozen encoder and head, clip mean and temperature (eq. 1, 2), threshold, Fig. 1 single column | 0.7 |
| IV Protocol | Splits and groups, Table II compressed, de-duplication sentence, guard and ledger, three declared tests and the Bonferroni line, AUC, group bootstrap, tie set and cheapest wins, non-inferiority (eq. 3, 4), margin derivation, seed and budget in two sentences, one merged splits and staircase figure (single column) | 1.4 |
| V Iterations | Battery to lever logic (half the current prose), one compact hypothesis and outcome table (I-0 to I-7, I-1b, I-1c) | 1.0 |
| VI Results | | 2.85 |
| VI-A Sweep | Compressed Table V, tie sets, T-A inconclusive, sign-reversed recording cues | 0.6 |
| VI-B Ladder on the development slice | Table VII with I-0 rows, I-1 same-encoder gain, speech steps, I-1b and I-1c in two sentences, re-run baseline and the grid extension | 0.75 |
| VI-C LAV-DF and pulse | Merged validation and sealed LAV-DF table, the post-hoc control, E-012, S-1 and one sentence on M-005 | 0.6 |
| VI-D Sealed staircase and confirmatory tests | Table IX, reduced Table X, prediction check, sealed against dev | 0.9 |
| (VI-E) After the freeze | Replacement paragraph of section 3 | inside VI-D or 0.2 extra, taken from VI-B |
| VII Limitations | 6 bullets: small dev slices; Celeb-DF family is a 44-identity mixture; one baseline family at 224 px with the margin at its floor and T3 not robust; speech: 42 speakers and residual recording cues; thresholds and calibration do not transfer (FakeAVCeleb, modern images); research-only data and weights | 0.45 |
| VIII Conclusion | Protocol, T1 to T3 as worded in section 2, the negatives, pointer to the report | 0.25 |
| Total main content | | **about 8.1** (trim related work or Table II by 0.1 if the build runs over) |
| References | about 30 entries | about 1.0, not counted |

### What moves to "see the technical report"

- Section VI Software and VI-A Analysis platform in full, with Fig. 5 (screenshots) and Fig. 6 (platform), the verdict-wording layer, the Evaluation Lab and the software test counts.
- Section VII-G in full except the replacement paragraph: FakeAVCeleb details and the codec controls (E-021, E-023), DeepAction and DF26 (E-018 to E-020, E-026, E-027), modern images in detail (E-028), retraining (E-029), lip sync (E-024, E-025), and Table XII.
- VII-H except three sentences; the full comparison table stays in PERFORMANCE_BENCHMARK.md and the report.
- Table I (full candidate list), the cost record, the preprocessing parameters, the Celeb-DF++ de-duplication arithmetic, the fine-tuning schedule details, the I-1 stage-1 learning-rate grid, the I-1b and I-1c full numbers, the M-005 amplitude table, Table VI (working-version EER and ECE) and the per-manipulation AUCs.
- Fig. 4 (ladder), Fig. 7 (ladder results) and Fig. 8 (LAV-DF cells): these remain in the report and the slides, since the tables carry the numbers.
- Limitations bullets about the platform, the Evaluation Lab, registration order of post-freeze work, the lip-sync pools, and Community Forensics subset selection.

### Cut order (whole subsections first)

1. Delete VI and VI-A, Fig. 5 and Fig. 6 (about 2.8 pages saved).
2. Replace VII-G with its paragraph and drop Table XII (about 1.35 pages).
3. Delete Fig. 4, Fig. 7 and Fig. 8 (about 1.6 pages).
4. Merge Tables III and IV into one compact table (about 1.4 pages).
5. Drop Table I, Table VI, III-F and III-G; merge Tables VIII and XI; reduce Table X (about 1.1 pages).
6. Merge Figs. 2 and 3 into one single-column figure; make Fig. 1 single column (about 0.6 page).
7. Cut VIII to 6 bullets, VII-H to 3 sentences, II to 3 paragraphs, the abstract to 248 words (about 1.6 pages).
8. Trim the remaining prose of IV, V and VII-C by about a third (about 1.5 pages).

Total saved: about 12 pages, from 21.4 pages of main content (excluding references) to about 8.1 pages, then trim the last 0.1.

## 5. Proposed abstract (248 words by wc; 222 words excluding pure number and bracket tokens)

Published deepfake detectors are usually compared under different protocols, which makes their numbers hard to compare. We build a detector for images, videos and the speech in videos as a documented path: one frozen pretrained encoder with a linear head per task, then iterations whose hypotheses and pass lines were written before each run. Splits are group-disjoint, all tuning uses a development slice, the sealed test sets are read once per frozen iteration, and three paired group-bootstrap tests were declared before that read. On the sealed Celeb-DF family test (7,650 videos, 344 groups, fakes from 44 identities) the final video system, a fine-tuned SigLIP encoder, reaches an AUC of 0.8218 against 0.6238 for the working version (+0.1980 [+0.1688, +0.2242]). It is non-inferior, with a margin of 0.01, to an Xception baseline re-run with the same recipe (0.8042); its superiority at the 95 % level (+0.0176 [+0.0025, +0.0327]) does not hold after Bonferroni correction over the three tests. On the sealed In-the-Wild test (42 speakers), silence trimming, noise augmentation and an average of three speech encoders reach 0.9148 against 0.7516 (+0.1633 [+0.1177, +0.2266]), above the predicted range, with recording-level cues not fully removed. The image probe, trained on ProGAN only, reaches 0.9316 and 0.9354 on held-out GAN and diffusion test sets (exploratory) but misses 0.7395 of images from 21 recent generators at its threshold. On LAV-DF the visual head is at chance within the produced videos (0.5032), so no audio-visual detection is claimed. Thresholds did not transfer to FakeAVCeleb.

Sources of every number: T1, T2, T3, 0.8042, 7,650, 344, 42, 0.9316, 0.9354 and 0.5032 from E-014 (0.9316, 0.9354 and 0.5032 are sealed but exploratory, and labelled or worded as such); the 44 identity groups from DOC_FACTS 3 and paper IV-C (data count); margin 0.01 (logged 0.0100) from E-013b and D-066; 0.7395 from E-028 (post-freeze, registered); "above the predicted range" from the E-014 prediction check; "recording-level cues not fully removed" from E-007 (RMS single-feature AUC 0.135 on In-the-Wild after trimming); "thresholds did not transfer" from E-015 (visual sensitivity 0.480). Headlines per modality: video (T1 and T3), speech (T2), image (sealed exploratory and E-028), audio-visual (LAV-DF negative). Method in one sentence: the second sentence. Main limitation: T3 not robust, residual speech cues, thresholds do not transfer. No development-slice number appears.
