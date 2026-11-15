# Project log: decisions and discussions, 2026-11-13 to 2026-11-15

This file records what was discussed and decided while turning the study into a paper, so that no decision lives only in a chat. It is the "why" behind the commits.

- [HANDBOOK.md](HANDBOOK.md) holds the full technical record, including every mistake.
- [FINAL.md](FINAL.md) is the short version: only what the final project uses.
- The binding research plan is the preregistration, [P-001](../results/preregistration/P-001-shortcut-neutral-av-study.md), with its amendments A1 to A6.

## 1. What the project is asked to be

The goals, as stated by Vaibhav on 2026-11-13:

- Keep the core framework: multimodal audio-visual deepfake detection, with preprocessing, several paths, and fusion at the end.
- Test the work and improve the results. Find every mistake from start to finish, and do not hesitate to suggest changes.
- Aim for work worthy of an MS application to Stanford: humane, careful, not padded.
- Store results properly.
- Look for objectives that are new enough for a conference paper, backed by real related-work research.
- Keep one handbook that explains everything and says what could be done better.
- Document everything, including changes to the maths, so the work can be defended even if the results are not better than existing methods.

## 2. Decisions, in order

| Date | Decision | Why | Where recorded |
|---|---|---|---|
| 11-13 | The paper builds on the root study, not the team project | The root study had the cleaner design. The team's sealed test sets had been looked at many times | Plan; HANDBOOK section 9 |
| 11-13 | The new contribution is modality prediction plus false-alarm control, inside a shortcut-controlled protocol | Fusion alone is not new (AVFF, AVH-Align, HAVIC) | P-001; [related work](research/related-work.md) |
| 11-13 | One handbook; stale documents deleted, not archived | Git keeps the history | HANDBOOK; commit `8366d12` |
| 11-13 | GPU budget: about 3 days | Enough for three seeds on every path | Plan |
| 11-13 | Fold `team/` into one project and delete it | Shivansh gave permission; he is deleting his repository | Commit `fb463ba`; [legacy records](../results/legacy-team/README.md) |
| 11-13 | Equal credit, one third each, for Vaibhav, Shivansh and Adithya | Agreed by the team | [AUTHORS.md](../AUTHORS.md) |
| 11-13 | Preregister hypotheses H1 to H4 before any run | So git order proves what was decided first | P-001, commit `dc7fa6f` |
| 11-13 | Primary models train on FakeAVCeleb only | The first draft of P-001 used LAV-DF for both training and testing | Amendment A2 |
| 11-13 | Sample LAV-DF by original recording | Each real clip stays next to its own edits | Amendment A3 |
| 11-13 | Neutralizer v4: one codec, fixed 3.00 s clips, dither | Version 1 put the shortcut back (H1 attempt 1 failed) | Amendment A4 |
| 11-14 | Restart the pipeline with lighter settings | Claude Code stopped jobs when RAM ran low | Commits `78dc2e3`, `033e3a9` |
| 11-15 | Push to GitHub after each result | The work had existed only on one PC | Pushed from `9b02d6e` on |
| 11-15 | Skip KoDF and HiFi-AVDF, then reverse both | HiFi-AVDF turned out to be public. Vaibhav chose to keep KoDF until refused | Amendments A5, A6 |
| 11-15 | Request AV-Deepfake1M and AV-Deepfake1M++ | A third corpus with per-modality labels; AVH-Align's own data | Amendment A6; [datasets](research/datasets.md) |
| 11-15 | Use the 2-day plan, not the 10-day plan | There is no time for 10 days | Section 5 below |

## 3. What we found, in plain words

1. **FakeAVCeleb's labels can be read from its files.** A classifier that never sees a face or hears a voice reaches 0.994 AUC. Generated voices are MP3 where real ones are AAC. Most face swaps are MPEG-4 where real clips are H.264. Generated speech starts and ends with digital silence. LAV-DF has no codec cue, but weak cues add up to 0.936. DFDC is close to clean (0.588).
2. **Naive cleaning put the shortcut back.** Re-muxing exposed hidden AAC padding differently per source, giving 0.933. Neutralizer v4 brought it down to 0.632. That still misses the preregistered 0.60. The main residual is video bitrate, which follows image content.
3. **Cleaning has a cost we must report.** 1,260 short clips were refused, all generated speech and none real.
4. **The real-only lip-sync path works for the right reason.** In-domain test AUC is 0.775 to 0.809 over three seeds. By method: voice cloning 0.94, Wav2Lip 0.84 to 0.86, face swap alone 0.63 to 0.68. The old sync branch was below chance.
5. **False alarms are the weak spot.** At the validation threshold, 27 to 39% of real clips are flagged. H4 tests whether real-only calibration fixes this.

Every number has its files in `results/`. The handbook, section 5, has the tables.

## 4. Questions asked, and the answers given

**Is the work good enough?** It is cautiously promising. It is strong on what was discovered (the shortcut census, the cleaning protocol, a sync path that works for a stated reason). It is weaker on what is built end to end: fusion has not run on clean data, and cross-corpus tests had not run on 11-15.

**Can we claim to beat existing methods?** Only by running them on our protocol. A number quoted from another paper is not a fair comparison, because those numbers come from shortcut-carrying data. HAVIC's released weights can be tested on HiFi-AVDF, DFDC and LAV-DF. They cannot be tested on our FakeAVCeleb test, because HAVIC fine-tuned on a random, ungrouped FakeAVCeleb split.

**Are we altering the maths?** Yes, in documented ways, and never after seeing the result each change affects. The EER is now interpolated. Thresholds come from validation, with a record of which clips chose them. Splits group identities properly, and the input data is neutralized. Old and new numbers are not directly comparable. HANDBOOK section 5.3 lists every change and its effect.

**Is it worthy of a Stanford MS CS application?** It can be a strong part of one, but no project gets anyone in alone. What impresses research faculty here is the method: finding the leak, failing to fix it the first time, fixing it, and preregistering. What would make it count:
1. Finish the cross-corpus results.
2. Post a preprint and submit to a workshop.
3. Get a professor's letter that speaks to this work.
4. Be able to explain every number without notes.
5. Describe your own role accurately within the equal-credit team.

**Is it good enough for the BTech final-year presentation?** Yes, and above the usual bar on depth and rigor. Today it is below the bar on completeness, because the full multimodal system has not run end to end on clean data. Before presenting:
- Say which parts were submitted on 2026-10-30 and which came after.
- Check the college's rules on AI assistance and disclose it if they require it. A large share of the recent code, analysis and documentation was written with Claude's help.
- Own every number.

**How complete is it?** About 32% on 11-15:
- Protocol and tooling: done.
- Shortcut audit: about 60%.
- Models: about 30%.
- Cross-corpus tests: about 5%.
- Baselines: 0%.
- Paper: about 5%.

New datasets add work rather than completing it.

**How much does the hardware limit us?** It limits speed, not the science.
- **CPU:** a Ryzen 5 5600X, 6 cores, at about 91% load. Preprocessing is CPU-bound.
- **RAM:** 32 GB caps the number of workers, and caused one stop.
- **GPU:** an RTX 5070 Ti at about 12%, mostly waiting on the CPU.

64 GB of RAM would save about 2 days. AV-Deepfake1M will be sampled, not fully processed.

**Do we have enough datasets?** Enough for all four preregistered tests. The limits are that modality prediction has only two labelled corpora, and that speakers may overlap with AV-HuBERT's pretraining data (VoxCeleb2, LRS3).

**AV-Deepfake1M or AV-Deepfake1M++?** AV-Deepfake1M first. It makes the AVH-Align comparison fair, gives a second corpus for testing the neutralizer, and has an unseen-speaker test set. ++ is for H4, because its built-in degradations are the shift that test needs. Both were requested.

**Will we lose progress if the session stops?** No. Finished work is in git and on GitHub. Large intermediate files stay on disk. Every stage resumes: finished clips are skipped, and receipts verify finished stages.

## 5. The plans

**The 10-day plan (deferred).** This is the full preregistered study:
- every hypothesis on FakeAVCeleb, LAV-DF, DFDC, HiFi-AVDF and MNW
- fusion as preregistered, with out-of-fold fold models
- H3 in both directions
- AVH-Align and HAVIC baselines
- a complete paper and a demo

AV-Deepfake1M adds 2 to 3 days. 64 GB of RAM saves about 2.

**The 2-day plan (chosen 2026-11-15):**

| Hours | Work |
|---|---|
| 0 to 3 | Raw HiFi-AVDF audit; neutral copies of HiFi-AVDF, DFDC and the LAV-DF test sample |
| 3 to 10 | AV-HuBERT features on those copies |
| 10 to 12 | The frozen sync path scored on all three corpora, per generator |
| 12 to 14 | H4: real-only thresholds against the source threshold |
| meanwhile | Visual and audio models train at low priority, if the cache finishes |
| 24 to 48 | Paper draft; Vaibhav reads and challenges every claim |

The 2-day paper claims four things:
1. A shortcut census on four corpora, including six commercial generators.
2. A neutral protocol, with its residual reported.
3. A real-only sync detector evaluated across corpora.
4. False-alarm control from real clips.

Fusion against visual-only (H2), modality prediction (H3) and the baselines become future work, stated as limitations.

## 6. Waiting on others

| Dataset | State |
|---|---|
| AV-Deepfake1M, AV-Deepfake1M++ | Requested 11-15 through the challenge and both gated Hugging Face repos |
| DeepSpeak v2 | Requested; send the reminder email around 11-22 if there is no answer |
| KoDF | Requested 11-15; included until refused |
| HiFi-AVDF | Public; downloaded 11-15, revision `c5a6aff6` |

## 7. Working rules agreed in this chat

- **Commits** are made as the user, with no AI co-author line.
- **Preregister first:** every change of plan is a dated amendment, committed before the result it could affect. Failed tests are reported, never dropped.
- **Locked test sets** never train, select or calibrate anything. A dataset that arrives late is scored by frozen models only.
- **Receipts:** every stage leaves one (command, input hashes, code hashes, output hashes).
- **GitHub:** push after each committed result.
- **Memory stops:** a job stopped for low memory is restarted only when Vaibhav agrees.
