# Claims stress test (hostile examiner), 2026-10-28

Scope: `paper/main.tex` (built `paper/main.pdf`, 20:01, newer than the source at 19:10), report sources `report/*.py` (built preview PDF 20:01), slides `slides/build_slides.py` (built pptx 20:01; slide 31 "Modern generators, published numbers", slide 35 "Benchmark, limits, unapproved heads", slide 39 "Conclusion" were confirmed in the built pptx). Record: `CLAUDE.md`, `docs/DOC_FACTS.md`, `docs/PROTOCOL.md`, `docs/DECISIONS.md`, `docs/EXPERIMENTS.md`, `docs/PERFORMANCE_BENCHMARK.md`, `ledger/final_ledger.jsonl` (read only), `data/_logs/freeze_run_all_final.log`, read-only `git log`. Nothing in the repository was changed.

## STOP-condition results (read first)

**STOP 1 (a comparison called confirmatory that was not declared before its sealed read): NOT TRIGGERED.** All three confirmatory tests (T1 video vs I-0, T2 speech vs I-0, T3 video vs Xception) and the 98.33 percent line were declared before the first sealed read. Timeline (all +0530, 2026-10-25):

| Event | Evidence | Time |
|---|---|---|
| Two confirmatory claims declared ("final vs I-0", "final vs re-run baseline") | PROTOCOL.md addendum lines 85-89 (D-041, 2026-10-24) | 2026-10-24 |
| Comparison (1) defined on two blocks (T-V `celebdf_family`, T-A `in_the_wild`); comparison (2) with "superior if lower bound > 0" | D-061 (DECISIONS.md line 905), "written before any sealed split is read" | 2026-10-25 |
| "The confirmatory tests are three" plus the 98.33 percent Bonferroni robustness line | D-062 (b), DECISIONS.md line 918; commit `5fbb859` | 18:44:17 |
| E-014 amendment (three tests, Bonferroni wording rule) | EXPERIMENTS.md E-014 "Amendment of 2026-10-25 evening"; commit `b5d1144` | 19:00:51 |
| Delta 0.0100 for T3 | D-066, "before any sealed read" | evening |
| Dry run (non-sealed: FF++ val, dev_x) | `data/_final_dry/DONE` | 19:54 |
| First sealed read: crops of the sealed test videos, then scoring I-0 | `data/_logs/freeze_run_all_final.log` lines "20:16:50 crops of the sealed test videos", "20:16:52 scoring I-0"; E-014 "run 2026-10-25 20:16 to 22:30" | 20:16:50 |
| Ledger (74 E-014 entries, date only, no time) first committed | commit `0552441` | 22:16:49 |

Note: the "third" test is not the Xception comparison (that was comparison (2) of D-041 from the start); the count rose from two to three because comparison (1) was split into a video test and a speech test (D-061, D-062 b), both before the sealed read. Residual weaknesses, not STOP: (a) `docs/PROTOCOL.md` was never amended and still says "The confirmatory claims are exactly two" (line 89), and DOC_FACTS section 7 still says "two pre-declared paired comparisons"; the slide 39 speaker note cites "docs/PROTOCOL.md addendum" as the evidence for three tests, which an examiner opening that file will find contradicted. (b) The ledger stores dates only; the ordering rests on git commit times and the freeze log, which are consistent but author-controlled.

**STOP 2 (a number in an abstract, results section or conclusion that does not match docs/EXPERIMENTS.md): NOT TRIGGERED.** A script extracted every decimal number from the three abstracts, the three conclusions, the slide 39 text and notes, the paper Results section (lines 1067 to 1760), and report `ch4_sealed.py`, `ch4_iter.py`, `ch4_results.py`, `ch4_platform.py` lines 280 to 440, and slides 712 to 1095 and 1281 to 1470. Every 4-decimal value was found verbatim in EXPERIMENTS.md, and every rounded 3-decimal value in the abstracts (0.693 [0.664, 0.731], 0.910 [0.878, 0.939], 0.872 [0.836, 0.908], 0.610, 0.813, 0.507) is the correct rounding of E-004, E-005, E-013b, E-009, E-011. Three values come from outside EXPERIMENTS.md, none in an abstract or conclusion:
- 0.0390 (paper Results; I-1b validation to dev_x gap) is arithmetic in D-065.
- 1.732 s per 1,000 inputs (report `ch4_results.py`) is in D-049 / `cost_record.json`.
- **0.11 and 0.14** (false-alarm share of real videos at the frozen threshold, Celeb-DF family and DFD): paper line 1001 (Software section), report `ch4_platform.py` line 166, slide 34 notes (build_slides.py line 1411). Source: PERFORMANCE_BENCHMARK.md section 12.4 (dagger values computed 2026-10-27) and the D-087 addendum. They have no E-ID in EXPERIMENTS.md, which strictly breaks CLAUDE.md rule 1 ("Numbers in any document come from a logged experiment (E-xxx)"). Not in a results or conclusion block, so not a STOP, but they should get an E-ID (or an M-ID) entry, labelled post hoc.

---

## Item 1. Confirmatory count

- Evidence: see the STOP 1 table. Paper line 544: "The confirmatory claims are exactly three tests, declared before any final run". Paper abstract line 115: "three confirmatory tests were declared beforehand". Report abstract (front.py line 131): "three tests were declared before the run". Slide 1 notes and slide 39: "three tests declared beforehand".
- For keeping: D-062 (b) at 18:44 and the E-014 amendment at 19:00 come before the first sealed crop at 20:16:50, and the change only adds a stricter robustness line.
- Against: the binding protocol file still says two. An examiner who reads PROTOCOL.md first will think the documents inflate the count after the fact.
- **Verdict: KEEP** in all three documents. Recommended precision (paper, protocol subsection, line 544): "The confirmatory claims are three tests, fixed before the first sealed read: the two comparisons of the protocol, with the comparison against the working version split into a video test and a speech test, and a Bonferroni robustness line added (decision recorded before the run)." Record-side fix, outside the deliverables: add a dated pointer addendum to PROTOCOL.md ("amended by D-062 b, three tests") and correct DOC_FACTS section 7. Until then, change the slide 39 note's evidence pointer from "docs/PROTOCOL.md addendum" to "D-062 (b), E-014 amendment".

## Item 2. Video vs Xception: "superior"

- Evidence: E-014 T3 +0.0176 [+0.0025, +0.0327]; 98.33 percent [-0.0011, +0.0355]; delta 0.0100. Declared rule (D-061, line 905; E-014 H2): "superior" if the lower bound of the 95 percent interval > 0. The 98.33 percent line was declared in D-062 (b) before the run, as a robustness line that "does not change any declared verdict". The E-014 amendment declared the wording: a verdict the 98.33 percent interval does not support is written "not robust to the Bonferroni level". D-068 fixed the document wording after the run: "superior at the 95 percent level, not robust to the Bonferroni level; non-inferior (above -0.0100) at both levels".
- For keeping: the per-test rule was declared, and "superior" is its literal output. The documents qualify it in every occurrence I found (paper abstract, table, results and conclusion; report abstract, ch4, ch4_sealed, ch5; slides 20, 35, 37, 39).
- Against:
  1. The project itself declared the three tests as a family and declared the Bonferroni line because of that. Under family-wise control the result is "not distinguished".
  2. The lower bound (+0.0025) is about a third of the final system's own seed SD on this set (0.0082).
  3. The family set is a mixture, and on its Celeb-DF++ part the baseline is higher (Xception 0.7351 against 0.7105, E-014 staircase), so the sign depends on composition.
  4. The baseline runs at 224 px instead of Xception's native 299 px, and its learning rate sits at the edge of the extended grid (D-066). Both handicap the baseline.
  5. The dev_x gap of +0.0380 shrank to +0.0176 on the sealed set.
  6. Paper limitations, lines 1819-1820, gives a wrong rationale: "The three confirmatory tests share one bootstrap, so the 98.33 % line is a robustness check and not a second decision rule." Bonferroni is valid under any dependence, so sharing a bootstrap is no reason to demote it. It is a robustness line because D-062 (b) declared it so.
- **Verdict: WEAKEN**: lead with non-inferiority, which is robust; say the superiority rule was met per test only, and do not claim "better". Exact wording:
  - Paper abstract (lines 123-128): "Against the re-run Xception baseline (AUC 0.8042 [0.7841, 0.8223]) the final video system is non-inferior with a margin of 0.0100 at both the declared 95 % level and the Bonferroni 98.33 % level. Its difference of +0.0176 [+0.0025, +0.0327] meets the declared per-test superiority rule, but the 98.33 % interval [−0.0011, +0.0355] contains zero, so no claim that it is better than the baseline is made." Use the same sentence in the paper conclusion (lines 1872-1876) and in the Table II verdict cell: "non-inferior at both; superiority rule met at 95 % only (not claimed)".
  - Paper lines 1819-1820, replace with: "The 98.33 % line was declared as a robustness check beside the 95 % decision rule; Bonferroni control holds under any dependence between the three tests, so a reader who wants family-wise control should read the 98.33 % column."
  - Report (front.py line 133, ch4.py line 33, ch4_sealed.py lines 23-25, ch5.py lines 27-28 and 54): the same wording as the paper. In the ch5 bullet, change "a superiority over the re-run Xception baseline that holds at the 95 percent level" to "non-inferiority to the re-run Xception baseline at both levels, with the per-test superiority rule met at 95 percent only".
  - Slides: slide 20 table verdict cell and slide 39 table cell become "non-inferior at both levels; superiority rule met at 95 percent only, not claimed". The slide 39 bullet "Superior to the re-run baseline at 95 percent only." becomes "Non-inferior to the re-run baseline; not shown better at the Bonferroni level." Add one sentence to the slide 20 notes: "On the Celeb-DF++ part alone the baseline is higher (0.7351 against 0.7105, exploratory)."
  - Process note: D-068 fixed the current wording, so this change needs a new D-entry (CLAUDE.md rule 10 and section 3).

## Item 3. Speech gain on sealed In-the-Wild (0.7516 to 0.9148)

- Evidence. M-004 (b), measured on train, validation and dev_x only, as single-feature fake-scores:

| Feature | ASVspoof dev | In-the-Wild dev_x (12 speakers) |
|---|---|---|
| Leading silence | 0.375 [0.299, 0.453] | 0.719 [0.425, 0.933] |
| RMS energy | 0.594 [0.537, 0.649] | 0.248 [0.073, 0.546] |
| Clip duration | 0.467 | 0.622 [0.532, 0.670] |

  The from-scratch reference is inverted at 0.246, and M-004 (a) excludes a label bug. M-004's own reading: "supported in sign, not established in size" (the silence and RMS intervals on In-the-Wild include 0.5).
  E-007, the I-2 lever (trim silence, peak-normalise): the manipulation check passed (leading-silence AUC 0.500 on all splits); the dev_x gain was +0.0623 [+0.0114, +0.1048]. RMS on In-the-Wild after the lever is still 0.135 [0.054, 0.382], so a level cue remains in the target domain.
  Sealed staircase (exploratory, no paired interval): I-0 0.7516, I-2 0.8125 (+0.061), I-3 0.8672 (+0.055, noise, low-pass, reverberation and MP3 augmentation, itself a nuisance-invariance lever), I-4 0.9148 (+0.048, three-encoder average). T2 = +0.1633 [+0.1177, +0.2266], robust. The E-014 predictions failed upward for both I-0 and I-4.
- Does M-004 support "the I-0 audio baseline was partly a shortcut artefact"? **Yes, in sign.** The I-0 head was trained where silence and level carry a weak class signal that reverses on In-the-Wild, and removing exactly those cues raised the out-of-corpus AUC as E-007 predicted. The size is not established (12 speakers), and the mechanism inside the I-0 head was not measured directly (no ablation of the frozen head's dependence on these features).
- For keeping the plain "improved": T2 is a declared test, and the final system really ranks sealed In-the-Wild clips better.
- Against: an examiner will say that part of the "improvement" is the I-0 baseline being handicapped by a corpus artefact. Removing a handicap is not better detection of synthesis, and the documents never say this next to T2.
- **Verdict: KEEP the T2 numbers and verdict; ADD this sentence** (paper Results after line 1474 and the Limitations; report ch4_sealed.py after line 22 and ch5.py; slide 20 card "What it does not show"):
  > "Part of the speech gain is the removal of a shortcut rather than better detection of synthesis: in the training corpus the leading silence and the level of a clip carry a weak class signal that points the other way on In-the-Wild (single-feature AUC of leading silence 0.375 on ASVspoof development data against 0.719 on the In-the-Wild development slice; RMS energy 0.594 against 0.248; supported in sign, not in size, with 12 speakers), and the first speech lever, which removes these cues, accounts for 0.061 of the 0.163 sealed gain on the exploratory staircase; the augmentation lever is an invariance lever of the same kind, and a level cue remains on In-the-Wild after normalisation (RMS AUC 0.135). Test 2 shows that the final speech system ranks In-the-Wild clips better than the working version; it does not show how much of that is better detection of synthesis."
  - Slide version: "Part of the speech gain removes a silence and level shortcut that reverses between corpora (M-004); not all of it is better detection."

## Item 4. FakeAVCeleb (E-015): "ranks well"

- Evidence: E-015 sealed half, visual 0.9552 [0.9442, 0.9657], audio 0.9263 [0.9149, 0.9366]; the pre-stated criteria (lower bound > 0.70) were met; at the frozen thresholds visual sensitivity is 0.480 and audio specificity 0.637. E-015's own interpretation: "a finding, not a claim of general detection"; real audio is VoxCeleb interview speech and fake audio is clean RTVC output (channel confound). E-021 (development half): codec metadata alone 0.9985, pixel statistics alone 0.9585; a common re-encode lowers the visual AUC from 0.9732 to 0.8772. E-023: codec fully confounded with method; codec-matched reals 0.8856 against 0.8763. The overnight note says "Unresolved: whether the sealed E-015 0.955 is inflated".
- "ranks well" appears in: the paper abstract (line 134) and conclusion (line 1882); report front.py line 136, ch4.py line 37, ch5.py line 36 ("ranks fake from real well"). The slides do not use the phrase (slide 29 shows "supported" plus limits; acceptable).
- For keeping: AUC is a ranking metric, and the pre-stated criteria were met.
- Against: "well" is an evaluative word for a number that a metadata-only model beats (0.9985). Placing it before "but" invites the reader to keep the praise and drop the caveat.
- **Verdict: WEAKEN.**
  - Paper abstract: "On the sealed half of FakeAVCeleb the frozen system's AUCs (visual 0.9552, audio 0.9263) met their pre-stated criteria, but they are not evidence of detection quality: in this data set the video codec coincides with the manipulation method (codec metadata alone separates real from fake video at 0.9985 on the development half), real and fake audio come from different recording channels, and at the frozen thresholds about half of the fake videos are missed."
  - Paper conclusion: "...the frozen system's AUCs on the sealed half of FakeAVCeleb met their criteria but are confounded with the codec, and its thresholds do not transfer...".
  - Report: the same wording in front.py, ch4.py and ch5.py. In ch5 line 38, change "the sealed number is read as ranking quality on that data set" to "...as ranking on a data set with a codec shortcut, not as detection quality".
  - Slides: KEEP.

## Item 5. LAV-DF 0.99 in the abstract

- Evidence: E-011 fused 0.9935 [0.9905, 0.9962] (LAV-DF validation subset; mechanically SUPPORTED). The control, within produced videos 0.5070 [0.4771, 0.5361], is labelled "Post-hoc control (labelled as such)" in E-011 / DOC_FACTS. E-012 (registered): produced-only recipe 0.5169 [0.4945, 0.5392]. Sealed (exploratory): I-6 fused 0.9945, I-6p visual within produced 0.5032 [0.4746, 0.5320], I-6bp 0.5203.
- Paper abstract (lines 112-115): "a fused score trained in the same dataset reaches an AUC of 0.99, but a control shows ... at chance (0.507)".
- For keeping: the negative control is a methodological contribution, and the reader needs the 0.99 to see why the control mattered.
- Against: the abstract is quoted out of context, and "0.99" is the most quotable number in it. The 0.507 that cancels it is a post hoc control on validation data. The sealed, more authoritative 0.5032 is not used.
- **Verdict: WEAKEN in the paper; KEEP in the report and slides.**
  - Paper abstract replacement: "On LAV-DF, a fused audio-visual score trained in the same data set looked near perfect, but its visual part separates the data set's original videos from its produced ones and is at chance within the produced videos (0.507 on validation, 0.503 on the sealed subset), so no audio-visual detection is claimed."
  - The report abstract already says this without the 0.99 (front.py line 130): KEEP.
  - Report ch5.py lines 20-22 (0.9935, 0.5070): KEEP, but add "post hoc control on the validation subset; the sealed subset repeats it (0.5032)".
  - Slide 39 "LAV-DF fusion driven by a pipeline cue": KEEP.

## Item 6. Published-results comparisons

Rule (PERFORMANCE_BENCHMARK section 1 rule 3): a difference only for class A, and class A needs "the same declared training protocol"; rule 4 forbids "better than".

| Location | Statement | Class per benchmark | Problem | Verdict and new wording |
|---|---|---|---|---|
| Paper VII-H lines 1738-1742; report ch4_platform.py lines 418-420 | EER 6.94 % against AASIST 1.13 %, "5.81 points worse" | A | Fine as a difference. But "our system trades in-corpus accuracy for transfer" (paper) and "built to transfer pays for that in-corpus" (report) imply better transfer than AASIST, which was never measured on a common out-of-corpus protocol (the Muller and Pascu rows are B). The EER is also called "computed afterwards", although the E-030 correction says the freeze reports already held it | **WEAKEN**: "...5.81 points worse. Whether it transfers better than AASIST was not measured on a common protocol." Cite the freeze report's EER and drop "computed afterwards", or keep "computed afterwards" and add "(also stored in the freeze report)" |
| Paper lines 1742-1745; report lines 420-421; slide 31 table row "In-the-Wild (42 of 54 speakers) / 15.8 percent / 33.9 and 7.2 percent, full set" | Published B values side by side with ours | B | Paper and report say "no gap is computed": acceptable. The slide 31 table has columns "Ours / Published" with no class column, and 15.8 sitting between 7.2 and 33.9 reads as a ranking | **WEAKEN (slides)**: add a column "Comparability" with "A, same protocol" and "B, different test set; no ranking", or move the In-the-Wild row to notes. Paper and report: KEEP |
| Paper lines 1746-1748 ("(iii) Zero-shot DF26: our DeepAction-trained head (AUC 0.4957) lies within the range of FF++-trained detectors"); report lines 421-422 and 425-427 (lists DF26 among class A rows); slide 31 note; slide 35 table titled "Class A row" with "Zero-shot video, DF26" | Our chance-level head positioned "within the range" of nine detectors | Labelled A in PERFORMANCE_BENCHMARK, but the row itself states that the training data differ (DeepAction against FF++) and the frames differ (2 s, 7 fps, 224 px against 32 frames). That fails the A definition ("same declared training protocol"); the rule-3 exception sentence contradicts the definition | **WEAKEN / reclassify as B**. Paper: "(iii) On DF26 without DF26 training our head is at chance (0.4957 [0.4755, 0.5168]); the DF26 authors report 44.0 to 69.7 for FF++-trained detectors without intervals; training data and frame sampling differ, so no ordering is drawn." Report: remove DF26 from the list of class A rows. Slide 35: move the DF26 row out of the "Class A row" table, or rename the header "Rows with a stated comparison"; slide 31 note, replace "lies inside the 44.0 to 69.7 range" with the paper sentence |
| Slide 35 table row "Face video, Celeb-DF family / 0.8218 / re-run Xception 0.8042; paired +0.0176 [...]" | Our re-run baseline | A (own re-run) | Under the header "Published / baseline" it is not a published number; fine. The paired interval is shown without the 98.33 % qualifier in the cell | **WEAKEN**: append "; non-inferior; superiority rule met at 95 % only" |
| Report ch1_2.py lines 99-110, context table ("none is a result of this project and no two rows are comparable") | Published context values | D | Conservative, consistent | **KEEP** |
| Slide 31 bullet "In-domain training moves commercial generators from chance to 0.75" | E-027 0.7548 against the installed head 0.4880 on the closed subset (E-020) | Own cross-experiment, not a declared paired test; different real sets (200 scenario-matched against 271) | Implies a measured gain | **WEAKEN**: "A head trained on DF26's open generators reached 0.75 on the commercial generators (one counted look); the installed head was at chance there; the two used different real sets." |
| Not in the deliverables but likely in a viva: PERFORMANCE_BENCHMARK 3.1 "Our encoder transfers better to Celeb-DF v2 than the re-run Xception on this split" (point estimates, no paired interval); 3.3 B row "about half of the best value Muller et al. report"; 3.3 T-A row classing our own I-0 as "the recipe family of Pascu et al." under "Published metric" with class A | Ranking language on unpaired or B rows; our I-0 presented as a published recipe | | Breaks the file's own rule 4 | Fix in the record (not a deliverable): "higher in point estimate, no paired test"; remove "about half"; reclass the I-0 row as "own working version (not a published number)" |

Summary for item 6: no table cell in the paper or report sets a published number beside ours as if comparable. Slide 31's speech table does, implicitly, for the In-the-Wild row. The DF26 row is mislabelled class A in the report and on slide 35. The sentence "trades in-corpus accuracy for transfer" (paper) and its report twin imply an unmeasured ranking.

## Item 7. Wording audit (improved, improvement, robust, robustness, confirmed, confirm, generalise or generalize, generalisation, superior, outperform)

Method: regex over `paper/main.tex`, `report/*.py` and `slides/build_slides.py` (speaker notes included). The full per-occurrence list with file:line, word, class and context is in `occ_classified.tsv` in this folder. Counts exclude "confirmatory" (36 more: paper 14, report 12, slides 10). All 36 name the protocol class of T1 to T3 and are correct. "outperform" occurs 0 times.

| Class | Meaning | Paper | Report | Slides | Total | Verdict |
|---|---|---|---|---|---|---|
| T | "improved" or "robust" for T1 or T2 | 10 | 13 | 19 | 42 | KEEP (maps to the declared rule, D-061, lower bound > 0, and to the 98.33 % line); for T2 add the item 3 sentence once per document |
| S | "superior" or "superiority" for T3 | 9 | 11 | 10 | 30 | WEAKEN per item 2 (lead with non-inferiority; "superiority rule met at 95 % only, not claimed") |
| N | "not robust" qualifier | 5 | 7 | 7 | 19 | KEEP |
| D | Protocol definitions ("confirmed performance", "superior only when", "robustness line", "generalisation result") | 11 | 13 | 11 | 35 | KEEP |
| L | Literature descriptions | 3 | 3 | 2 | 8 | KEEP, except report ch1_2.py line 135 (see W) |
| P | Plans or goals ("improve in iterations", "candidate improvement", "not an improvement") | 0 | 5 | 3 | 8 | KEEP |
| R | Titles of cited works and citation keys | 2 | 11 | 5 | 18 | KEEP (not our claims) |
| K | Keyword "generalisation" | 1 | 0 | 0 | 1 | KEEP |
| O | Other measured statements with a registered rule | 3 | 5 | 2 | 10 | KEEP (details below) |
| W | Need rewording | 1 | 4 | 0 | 5 | WEAKEN (below) |
| **Total** | | **45** | **72** | **59** | **176** | |

W items, each with its exact replacement:
1. Paper lines 1819-1820, "The three confirmatory tests share one bootstrap, so the 98.33 % line is a robustness check and not a second decision rule." No rule maps to this, and the reasoning is wrong. Replace per item 2.
2. Report ch1_2.py line 135, "Cozzolino et al. obtained robust results with a linear support vector machine on CLIP features". This is an unqualified evaluative word about a paper. Replace with "Cozzolino et al. trained a linear support vector machine on CLIP features from few examples of one generator and evaluated it on many generators" (as in paper line 229; check the RESEARCH.md status of that row).
3. Report ch4.py line 37, "the generated-video head does not generalise to DF26". Maps to E-020 H-G1 "not supported"; the result is from one benchmark. Replace with "the generated-video head is at chance on DF26 (0.4957)".
4. Report ch4_platform.py line 124, "The head did not generalise to a held-out benchmark". Replace with "The head was at chance on one held-out benchmark (DF26, 0.4957 [0.4755, 0.5168])".
5. Report ch4_platform.py line 316, "this head, trained on this corpus, does not generalise to talking-head video from other generators". This generalises a negative from one benchmark. Replace with "...was at chance on DF26's talking-head clips from ten other generators".

O items, all KEEP:
- Paper line 1577 and report ch4_results.py line 152, "did not improve the calibration": measured, E-004.
- Paper line 1675, report ch4_platform.py line 404 and slide 33 note (line 1385), "hypothesis of improvement is supported": E-029, registered, lower bound > 0. It is already followed by the threshold failure; keep the word "ranking", which the report has as "gain ... in ranking".
- Report ch4_results.py line 79, "generalisation gap": validation minus dev_x, defined.
- Report ch4_results.py line 95 and slide 13 note (line 820), "improved / improves" for the T-I ECE from 0.0462 to 0.0050: add "on two generator units" on slide 13.
- Report ch5.py line 75, future-work "before any claim of robustness".
- Paper line 1819, "not a confirmed gain" (I-1b): correct.

Related claim outside the word list (report ch5.py line 49): "most of that gain comes from fine-tuning, since the Xception baseline gains +0.1783 ... over the working version as well". This is an untested causal inference: Xception fine-tuned against a frozen DINOv2 probe changes the encoder too. **WEAKEN**: "The gain over the frozen probe of the same encoder (+0.1700 [+0.1228, +0.2114]) isolates fine-tuning; the Xception gain over the working version (+0.1783) is consistent with it but changes the encoder as well."

## Item 8. Development-slice numbers in abstracts and conclusions

| Document, place | dev_x number | Small group | Current label | Recommendation |
|---|---|---|---|---|
| Paper abstract lines 99-102 | 0.693 [0.664, 0.731] to 0.910 [0.878, 0.939]; Xception 0.872 [0.836, 0.908] | 15 identities | "Exploratory results on the development slices" | Abstract space: **REMOVE** the intervals and keep one clause ("on the development slice (15 identities, exploratory) fine-tuning raised the AUC from 0.69 to 0.91"), or remove entirely; the sealed T1 carries the claim |
| Paper abstract lines 106-108 | speech 0.610 to 0.813 | 12 speakers | "wide intervals from 12 speakers" | **REMOVE** from the abstract (sealed T2 supersedes it) or keep with "12 speakers, exploratory" |
| Paper conclusion lines 1859-1863 | 0.693 to 0.910; 0.872; 0.610 to 0.813 | 15 / 12 | "on the development slices and exploratory" | KEEP, but add "(15 identities; 12 speakers)" inline |
| Report abstract front.py line 129 | 0.6933 to 0.9096; 0.9315; 0.9192; 0.6100 to 0.8130 | 15 / 12 | "exploratory, on validation data and on development slices" | **LABEL** inline "(15 identities)" and "(12 speakers)"; drop 0.9315 and 0.9192 from the abstract |
| Report conclusions ch5.py lines 15-20 | 0.6933 [...], 0.9096 [...], 0.8716 [...], +0.0380 [...], 0.9315, 0.9192, 0.5338, 0.6100 to 0.8130, **0.9935 (generated images, 2 generator units)** | 15 / 12 / **2** | "The results are exploratory" | KEEP with inline group counts. **REMOVE 0.9935** (DOC_FACTS 14: with 2 groups "the interval is not interpretable as a population interval") or add "on two generator units". The dev_x +0.0380 against Xception sits right before the sealed T3 (+0.0176) paragraph; add "development slice, 15 identities" |
| Report conclusion bullets ch5.py lines 49-50 | +0.1700 [...], +0.1783 [...], 0.8130 | 15 / 12 | partly | Label "(development slice, 15 identities)"; see item 7 for the causal claim |
| Slide 39 left column | "Face development AUC 0.7396 to 0.9096"; "Speech 0.6100 to 0.8130 on 12 speakers" | 15 / 12 | heading "What the exploratory runs show" | Add "(15 identities)". Note the inconsistency: slide 39 starts from 0.7396 (the SigLIP probe) while the paper and report start from 0.6933 (I-0). Use one baseline (I-0, 0.6933) everywhere |
| Slide 39 table | "0.693; 0.994; 0.610"; +0.0380; 0.9315 and 0.9192 | 15 / **2** / 12 | "(exploratory)" | **REMOVE 0.994** or label "(2 generator units)" |

## Item 9. Pre-declared or post hoc, for every number in the abstracts and conclusions

| Number(s) | Source | Status |
|---|---|---|
| Sweep values, I-0 0.6933, 0.6100, 0.9935 | E-001 to E-004 | Pre-declared rule (D-036), exploratory |
| I-1 0.9096, +0.1700 | E-005 | Pre-declared hypothesis, exploratory |
| Xception dev_x 0.8716, +0.0380 | E-013b, D-066 | Pre-declared run; the +0.0380 paired difference is a descriptive comparison, not a registered test |
| Xception minus I-0 +0.1783 | DOC_FACTS 15 / E-013 | Post hoc comparison (not registered); used in the report for a causal claim |
| I-1b 0.9315, I-1c 0.9192 | E-016, E-017 | Pre-declared (D-063, D-064), run with triggers not met to the letter |
| Speech 0.6100 to 0.8130 | E-007, E-008, E-009 | Each lever pre-declared against its predecessor; the cumulative 0.610 to 0.813 is descriptive |
| LAV-DF fused 0.99 / 0.9935 | E-011 | Pre-declared (mechanical) |
| LAV-DF 0.507 / 0.5070 control | E-011 | **Post hoc** (labelled so in E-011); E-012 0.5169 is pre-declared; sealed 0.5032 is exploratory |
| LAV-DF frozen visual 0.3838, audio-within-produced 0.8574 | E-010 / E-011 | E-010 pre-declared pre-check; 0.8574 is part of the post hoc control |
| Pulse 0.301 against 0.6 | E-006 | Pre-declared |
| Positive control 0.008, 377 of 432 | M-005 | Declared in D-062 (d) before the run |
| T1, T2, T3 point values and 95 % intervals, delta 0.0100 | E-014 | Pre-declared (D-061, D-062 b, D-066) |
| 98.33 % intervals | E-014 | Pre-declared as a robustness line (D-062 b), not a decision rule |
| Predictions (0.75 to 0.92, etc.) | E-014 | Pre-declared |
| 344 groups, 42 speakers, 7,650, 24,788 | E-014 | Descriptive |
| FakeAVCeleb 0.9552, 0.9263, criteria | E-015 | Pre-declared |
| FakeAVCeleb sensitivity 0.480, specificity 0.637 | E-015 | Pre-declared as secondary metrics, no success line (descriptive) |
| Codec shortcut 0.9985, 0.9732 to 0.8772, 0.8856 against 0.8763 | E-021, E-023 | Registered follow-ups, but **post hoc relative to E-015** (motivated by its result); development half |
| DF26 0.4957 | E-020 | Pre-declared by the record; registration order not provable from git (RESEARCH_INTEGRITY) |
| Lip-sync 1.3 % [0.4, 3.8], 0 of 160 | E-025 | Pre-declared (same caveat on git ordering) |
| EER 6.94 %, 15.8 % (paper VII-H, report, slides 31 and 35; not in the abstracts) | E-030 | Registered before computing (15:14:09) but after the sealed scores were known: **post hoc relative to the freeze**; the E-030 correction says the freeze report already held the EER. The AASIST, RawNet2 and Pascu comparisons are post hoc, not declared tests |
| 0.11 and 0.14 false-alarm shares (paper line 1001, report ch4_platform.py line 166, slide 34 notes; not in the abstracts) | PERFORMANCE_BENCHMARK 12.4, D-087 addendum | **Post hoc**, computed 2026-10-27 from stored sealed scores; **no E-ID** in EXPERIMENTS.md |
| PERFORMANCE_BENCHMARK 12.4 values: official-list 0.8478 / 0.7615, AP 0.9387, CNN AP 0.9714 | 12.4 | Post hoc; **not used in any deliverable** (grep: 0 hits). Good |
| E-026 0.7682; E-027 0.7548 / 0.8023; E-028 0.7039 / 0.7395; E-029 0.8839 / +0.1800 / 0.5213 (report and slide conclusions and later-work sections) | E-026 to E-029 | Pre-declared (registered before their runs) |

## Item 10. Number mismatches against EXPERIMENTS.md

No mismatch found (see STOP 2). Out-of-log numbers: 0.11 and 0.14 (needs an E-ID or M-ID), 0.0390 (D-065 arithmetic), 1.732 s (D-049). One wording inconsistency of fact: the paper and report call the EER "computed afterwards" while the E-030 correction (2026-10-27 23:10) says it was already in the freeze reports. A second: report ch5.py bullet line 45 says the family set "has about 44 independent identity groups" while the bootstrap resampled 344 groups (44 identity groups plus 300 single YouTube reals). Both statements are true, but give the 344 figure with its split, as the paper does (line 1772).

---

## Three highest-risk claims an examiner would attack

1. **"The final video system is superior to the re-run Xception baseline" (T3).**
   - Attack: "You declared three tests, added Bonferroni yourselves, and the corrected interval contains zero. On the Celeb-DF++ part the baseline is higher. The baseline ran at 224 px, not 299, and at the edge of its grid. Why say superior at all?"
   - Recommended answer: "We do not claim it is better. The robust result is non-inferiority with a margin of 0.0100 at both levels. The +0.0176 [+0.0025, +0.0327] meets the per-test rule we declared before the run, and we report that mechanically, but the family-wise interval [−0.0011, +0.0355] includes zero, the margin is small against the seed SD (0.0082), the sign reverses on Celeb-DF++ fakes (0.7105 against 0.7351, exploratory), and the baseline handicaps (224 px, grid edge) favour us. What we can defend: a fine-tuned SigLIP base under the same recipe is not worse than Xception on this sealed family." Change the documents per item 2 so they say this first.

2. **"The speech system improved from 0.7516 to 0.9148 on sealed In-the-Wild" (T2).**
   - Attack: "Your own M-004 shows the working version learned silence and loudness cues that reverse on In-the-Wild. Isn't the gain mostly removing a self-inflicted shortcut? And your predictions failed upward, so you did not understand this system."
   - Recommended answer: "Partly, yes. M-004 supports a sign reversal of leading silence and level between ASVspoof and In-the-Wild (supported in sign, not in size, with 12 speakers). The first lever removes exactly those cues and accounts for about 0.06 of the 0.16 sealed gain on the exploratory staircase. The augmentation lever is an invariance lever of the same kind. T2 shows that the final system ranks In-the-Wild clips better than the working version; it does not show how much is better synthesis detection, and a level cue remains (RMS AUC 0.135 after normalisation). Both upward prediction failures are logged and stated." Add the item 3 sentence to all three documents.

3. **"Comparable with published detectors" (the project's stated aim) and the published-results section.**
   - Attack: "You have no class A row for face video against any published detector. Your DF26 'class A' row compares a DeepAction-trained head with FF++-trained detectors. Your speech system is 5.81 EER points worse than AASIST in-corpus, and you excuse it with 'trades in-corpus accuracy for transfer', which you never measured against AASIST."
   - Recommended answer: "The only comparison on our protocol is the re-run Xception (non-inferior). Every published face-video number uses other subsets, frame counts and training compression, so we computed no differences. The DF26 row is not like-for-like in training data and should be class B. In-corpus our speech system is worse than AASIST by 5.81 EER points, and we have no common out-of-corpus test against AASIST, so we make no transfer claim against it." Apply the item 6 wording changes: delete "trades in-corpus accuracy for transfer" and its report twin, reclassify DF26, and add a comparability column to slide 31.

Runner-up: "the frozen system ranks FakeAVCeleb well" (item 4). An examiner will answer that the codec metadata alone scores 0.9985. Use the item 4 wording.
