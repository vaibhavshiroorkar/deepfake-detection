# STATUS

Last updated: 2026-10-24 (Session 1, night)

NEXT STEP: Start the I-0 sweep (E-001 to E-003, protocol D-036) by extracting frozen-encoder features for the E-001 candidates on the FF++ train and val crops and the Celeb-DF dev_x crops in a background job, using the loaders in src/dfd/data/loaders.py; in parallel accepts the LAV-DF terms on https://huggingface.co/datasets/ControlNet/LAV-DF (HF account gr8SHivANsh), and pastes the DFDC Kaggle rules text.

## Where the project stands

- Done on 2026-10-24 (night), after the usage limit: loaders for face clips, images and audio with a final-only split guard, 9 tests (52bcae7); the pre-declared encoder sweep protocol, E-001 to E-003 hypotheses, D-036 proposed and the audio-visual risk and fallback claim (764519a); the D-034 face-detector comparison M-002 on FF++ val and Celeb-DF dev_x frames (b2245a1); the audio encoder candidate verification, no choice made (docs/_agents/O7AUD.md, 93d80b7). The O-007 message of the user (sweep design, D-034 check, audio finding) was therefore handled before the limit; nothing needs resending.
- User review on 2026-10-24 (night): D-022 (fallback), D-024, D-025 (amended to non-inferiority), D-026, D-034, D-035 (amended, with a sign-flip permutation sensitivity check still to implement in dfd.eval) and D-036 (tie set against the leader, cheapest wins) are Decided (user); D-037 records the housekeeping: CLAUDE.md rule 6 now allows PARTIAL papers for the parts read, `Document Template/` stays ignored, the D-017 to D-020 and D-018 to D-021 mapping is confirmed, GitHub contradiction closed.
- Real-data loader check on FF++ val (throughput against the prep speed): started, result not yet recorded here.

- Session 1 decisions are logged: D-011 and D-012 confirmed; D-013 (open datasets only) superseded by D-020 (gated datasets allowed, user downloads them); D-014 to D-016; D-017 (Python 3.13 and PyTorch 2.14.1+cu130); D-018 (images: face-manipulated and fully generated); D-019 (audio-only is a separate mode); D-021 (requirements-driven data selection, approved by the user). D-009 now has the deadline 2026-10-30. Reasons stated by the user are filled in for D-001 to D-010.
- Literature: 67 papers in docs/RESEARCH.md (R-001 to R-067), 19 READ and 48 PARTIAL, plus an NR table. R-001 to R-018 were all re-marked PARTIAL.
- Deliverable scaffolds exist and are committed: paper (compiles, 4 pages), report (26 pages, built from the supplied Word template), 19-slide deck (supplied master, logos, colours, bars). They contain no results.
- Data: the open batch is downloading on this machine (see the Downloaded datasets table). The gated datasets (FF++, DFD, Celeb-DF v2, Celeb-DF++ narrowed to the 21 new generators, FakeAVCeleb, WildDeepfake test shards) arrive by drive on Monday 2026-10-26 (D-022; checklist in docs/DATA_LOG.md). LAV-DF waits for the user's Hugging Face token.
- No model, training or detection code exists and no experiment has been run.
- GitHub: `gh` is logged in and `main` plus the `setup` tag were pushed to the private repository https://github.com/Shivansh-gupta-2005/deepfake-detection on 2026-10-24 (local and remote both at 3b4e14d at the time). Later commits are pushed at the end of each work block.

## Blockers

1. GitHub: resolved. Logged in, remote set, main pushed (see above).
2. LAV-DF (25.6 GB): the user supplied a read token (account gr8SHivANsh), which was used only as an environment variable of the download process and written to no file. The server answers 403: the account is not on LAV-DF's authorized list, so the terms still have to be accepted on the dataset page. The token appeared in the chat and should be revoked at https://huggingface.co/settings/tokens when the download is done; a new one can be made then.
3. Gated datasets: approvals for FF++ and Celeb-DF arrived on 2026-10-24. By D-022 the user downloads them on Monday 2026-10-26 on a fast connection and carries them over on an exFAT or NTFS drive (checklist in docs/DATA_LOG.md). The fallback for the video MVP is LAV-DF if FF++ has not passed the completeness checks by Monday 18:00 (decided autonomously, needs the user's review).
4. DFDC on Kaggle: the facts in the DFDC row of docs/DATA_LOG.md are user-supplied (a paste of the Kaggle Data page on 2026-10-24) and are not verified against the official page, which the tools could not read. The rules text and licence have not been read, so DFDC may not be used. DFDC stays DEFER and is off the critical path.


## Open questions for the user

1. Template details that conflict across the three supplied files (found by stream A, full list of 12 in docs/_agents/A.md section 6): the department name (pptx "Engineering Electronics & Telecommunication", Word template "Electronics and Telecommunication Engineering", pdf "Electronics and Computer Engineering"; the newest, the pdf's, is used); semester and batch (the Word template says Semester VII and an old batch, the pdf says Semester VIII 2026-27; "Semester VII 2026-27 Batch" is used); student order, the title, the blank Director name, the Acknowledgement position and paper author details (no emails supplied).
2. The report has 18 references against the template's 10 to 15 and is 26 pages against the template's 30 to 60: trim references, and let the report grow with results?
3. (resolved 2026-10-24, D-037) Does the user accept D-020 and D-021 as the numbers for the access-policy change and the data selection (the user's message used D-017 and D-018, which were already taken)?
4. (resolved 2026-10-24, D-037) CLAUDE.md rule 6 says only `READ` papers may be cited; Session 1 added `PARTIAL` and allowed it for the parts read. Confirm or edit the rule.
5. (resolved 2026-10-24, D-037) Should the folder `Document Template/` (earlier project documents with names and roll numbers) stay out of git? It is currently ignored.
6. The submission needs the PDF form of the deliverables? The previews are ignored by git; say if they should be committed.

## Mistakes and corrections

| Date | Mistake | Correction |
|---|---|---|
| 2026-10-24 | CLAUDE.md was missing in Session 0; the first instruction (read it) could not be followed. The repository folder was empty. | Reported; the user added CLAUDE.md at the start of Session 1 and it was committed. The commit message format of CLAUDE.md (`type(scope): summary`) is used from Session 1; the two Session 0 commits kept their provisional format and were not rewritten. |
| 2026-10-24 | While adding the PyTorch dependency in Session 0, a duplicated section made pyproject.toml unparsable. | Duplicate removed, `uv sync` rerun, the GPU tests passed. |
| 2026-10-24 | An arXiv identifier (2009.14232) was written from memory in Session 0 and turned out to be an astrophysics paper. | Discarded; the right paper was found and its title page checked; every later identifier was checked against a title page, including those of the Session 1 streams. |
| 2026-10-24 | Session 0 said 17 papers were read; the matrix actually had 18 entries (R-001 to R-018). Also, all 18 were read selectively but were marked READ. | Counted correctly now; all 18 re-marked PARTIAL with the sections read (R-010 had only abstract and introduction read). |
| 2026-10-24 | The CPU was recorded from the prompt as "7900K" in CLAUDE.md and the prompt; the machine has a Ryzen 9 7900X. | The measured value stands; CLAUDE.md was not edited by the implementer (DECISIONS.md, contradictions). |
| 2026-10-24 | The download-speed measurement in EXPERIMENTS.md was first written "about 18:00 local time", a guess. The clock read about 17:30 when the downloads started. | Corrected to "between about 16:55 and 17:10", reconstructed from file timestamps. |
| 2026-10-24 | The Session 1 data list (Tier 1 and Tier 2) was built from availability under D-013, which the user then superseded. | Withdrawn; rebuilt from requirements (D-021) and approved. |
| 2026-10-24 | The single-stream download of ASVspoof 2019 from Edinburgh ran at 90 to 190 KB/s, which would have taken about 11 hours. | Stopped; a multi-connection downloader (8 parts) was added to scripts/data/fetch_open.py and now runs at about 2 MB/s. |
| 2026-10-24 | `.gitignore` pattern `data/` also ignored `scripts/data/`. | Anchored to `/data/`. |

## What each subagent did and what the main agent verified (D-015)

| Stream | Output | Verification by the main agent |
|---|---|---|
| A, templates | paper/, report/, slides/, assets/branding/, docs/_agents/A.md | Paper rebuilt with latexmk (4 pages, no undefined references); report cover and a slide rendered and viewed (logos, red bars, campus pictures, group details present); bibliography matches the matrix; the only number in the paper is the DeepfakeBench Xception AUC under its own protocol. Not checked: the full report text and every slide, and the Word styles beyond the cover |
| B, datasets | docs/_agents/B.md | HF gating, licence tags and file counts re-checked through the Hugging Face API; FF++ and Celeb-DF forms read; ASVspoof 5 and DeepfakeTIMIT Zenodo licences and sizes re-checked; ForenSynths archives opened by range requests. Not re-checked: the other tables |
| C1, audio-visual and audio | docs/_agents/C1.md (14 papers) | All 13 downloaded title pages matched; six headline numbers re-checked in the paper text |
| C2, visual, fine-tuning, calibration | docs/_agents/C2.md (17 papers) | All 17 title pages matched; the Effort fine-tuning numbers, the file-format bias number and the oracle-threshold table re-checked |
| C3, rPPG and surveys | docs/_agents/C3.md (18 papers) | All title pages matched; McDuff, Yang et al., DeepRhythm, Harraq and Aldwairi (second paper) and POS numbers re-checked; the R-009 discrepancy confirmed in the later paper's text |

## Provisional items to confirm

- Department, semester and batch wording on the report and slides (open question 1).
- Python 3.13 and PyTorch 2.14.1+cu130 are accepted by the user (D-017).
