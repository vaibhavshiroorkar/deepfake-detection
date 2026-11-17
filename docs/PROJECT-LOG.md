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
| 11-15 | Add HiFi-AVDF back; one frame size per corpus | Public data; generated clips are 1920x1080 while real ones are 1280x720 | Amendments A5, A7 |
| 11-15 | Build the black book, research paper, slides and poster alongside the 2-day results; every number filled from `results/numbers.json` | Deadline within 2 days; no number typed by hand | [deliverables/](../deliverables/README.md) |
| 11-15 | After the 2-day deadline, run the full project (the 10-day plan) | The 2-day version cuts fusion (H2), modality prediction (H3) and the baselines | Section 5 below |
| 11-16 | Pipelines run as detached processes, outside Claude Code | Both chains died when a Claude Code session ended. Detached, they survive; each resumes from its logs | `runs/p001-2day.log`, `runs/p001-resume2.log` |
| 11-16 | Only the 2-day chain neutralizes DFDC and the LAV-DF test sample | Two writers on one output corrupted HiFi-AVDF copies on 11-15 | Mistakes log, HANDBOOK section 6 |
| 11-16 | Three days (11-16 to 11-18) instead of two, so H2 (fusion against visual-only, cross-corpus) runs now | Vaibhav asked whether H2 fits in 3 days; the estimate was about 25 hours of compute plus a day of buffer | P-001 amendment A8, commit `7c81b89`; `runs/p001-h2.sh` |
| 11-16 | Guide is an Assistant Professor; academic year 2026-27; batch 2023-27, to confirm with the guide | Vaibhav confirmed the title and graduates in 2027 | `deliverables/report/report.tex` |
| 11-16 | Report and paper compile on Overleaf | Vaibhav uses Overleaf; no LaTeX on this PC | `deliverables/overleaf.py` |
| 11-16 | Slides and poster built from the college slide master in the October deck | The `Document Template` folder is not in the repo; the submitted deck still carries its master | [deliverables/](../deliverables/README.md) |
| 11-16 | Label LAV-DF clips by their kept window as well as by file (A9) | 46% of LAV-DF fake clips had no edited time in the 3 s window. Found after the file-label result was read, so both are reported; H2 uses the window label | P-001 A9; handbook mistake 68 |
| 11-16 | Locate small faces with YuNet before landmarking, on the targets (A10) | The landmarker found a face in only 35% of DFDC windows. Found after the DFDC result was read, so both are reported | P-001 A10; handbook mistake 69 |
| 11-16 | Train on one corpus for this submission; plan leave-one-corpus-out for the next study | Vaibhav asked whether training on one corpus is right. It is right for measuring transfer, but it limits how strong the detector can be | `docs/research/P-002-design.md` |
| 11-16 | Write the plan to rebuild for maximum performance after the deadline | Vaibhav asked for an unbiased redesign, even a remake | `docs/research/P-002-design.md` |
| 11-17 | Report H2 as passing its rule but weaker than its verdict, with two post-hoc checks beside it | Vaibhav asked whether the result was optimistic. It was: the FakeAVCeleb gain is only the cloned-voice clips, and the LAV-DF gain is over a visual path inverted by a file difference. The preregistered verdict is kept and reported as is | `results/p001-h2/README.md`; `h2_checks` in `scripts/build_numbers.py` |
| 11-17 | Run HiFi-AVDF caching, then H2 with HiFi-AVDF, then A11, strictly one after another | Both PC hangs came with CLIP training beside target caching; four cache shards alone ran stable for seven hours. The serial order still fits the A11 stop rule (11-17, 23:59) if the estimates hold | Handbook mistake 71; `runs/p001-h2.sh`, `runs/p001-a11.sh` |

## 3. What we found, in plain words

1. **FakeAVCeleb's labels can be read from its files.** A classifier that never sees a face or hears a voice reaches 0.994 AUC. Generated voices are MP3 where real ones are AAC. Most face swaps are MPEG-4 where real clips are H.264. Generated speech starts and ends with digital silence. LAV-DF has no codec cue, but weak cues add up to 0.936. DFDC is close to clean (0.588).
2. **Naive cleaning put the shortcut back.** Re-muxing exposed hidden AAC padding differently per source, giving 0.933. Neutralizer v4 brought it down to 0.632. That still misses the preregistered 0.60. The main residual is video bitrate, which follows image content.
3. **Cleaning has a cost we must report.** 1,260 short clips were refused, all generated speech and none real.
4. **The real-only lip-sync path works for the right reason.** In-domain test AUC is 0.775 to 0.809 over three seeds. By method: voice cloning 0.94, Wav2Lip 0.84 to 0.86, face swap alone 0.63 to 0.68. The old sync branch was below chance.
5. **False alarms are the weak spot.** At the validation threshold, 27 to 39% of real clips are flagged. H4 tests whether real-only calibration fixes this.
6. **Cleaning works on the new corpora too.** HiFi-AVDF falls from 0.9985 to 0.629, LAV-DF from 0.936 to 0.557, and DFDC from 0.588 to 0.526.
7. **The lip-sync path transfers poorly.** Its cross-corpus AUC is 0.55 to 0.66. On HiFi-AVDF it falls below the file-cue ceiling, because generators that make sound and picture together leave little lip mismatch. On Sora 2 it scores below chance.
8. **Real-only calibration controls false alarms, at a cost.** The carried-over threshold flags 31 to 48% of real clips. A threshold from 200 real clips holds about 5%, and H4 is met on HiFi-AVDF and on LAV-DF (file label). It then catches only 5 to 11% of fakes.
9. **Checks found three protocol flaws after results were read.** Short video, LAV-DF windows that missed the edit (A9), and small faces (A10). Each was fixed in the open, and the earlier results are kept.

Every number has its files in `results/`. The handbook, sections 5.1 to 5.1c, has the tables.

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

**The 3-day plan (chosen 2026-11-16, replaces the 2-day plan).** The deadline window became 16 to 18 November, which fits H2 as preregistered (amendment A8):

| When | Work |
|---|---|
| Mon to about 09:00 | The 2-day chain finishes: sync path on the three targets, H4, audits of the neutral copies |
| Mon to about 17:00 | FakeAVCeleb visual and audio cache, with the full CPU after 09:00 |
| Mon evening, overnight | Visual and audio training, 3 seeds each, on the GPU; the three targets cached on the CPU at the same time |
| Tue about midday | Targets scored by the frozen visual and audio models; the fusion test; H2 result |
| Tue afternoon to Wed | Deliverables updated with H2; Wednesday is buffer |

The paper then claims five things: the four above plus H2, whichever way it falls. Modality prediction (H3) and the baselines stay future work.

**After the deadline: the full run.** Once the deliverables are submitted, the rest of the 10-day plan runs:

1. Speed up the cache build.
2. LAV-DF reverse models and H3.
3. AVH-Align and HAVIC baselines.
4. The StreamFusion MLP, the secondary fusion.
5. AV-Deepfake1M and KoDF if access arrives.
6. The full paper.

The results so far stay valid: the full run reuses the same frozen models, neutral copies and splits. The deliverables are then rebuilt from the larger `results/numbers.json`. Open items are tracked in [PENDING.md](PENDING.md).

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
