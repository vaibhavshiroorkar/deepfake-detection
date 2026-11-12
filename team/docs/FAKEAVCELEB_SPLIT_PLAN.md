# FakeAVCeleb v1.2: identity-aware development / sealed split plan (Option B)

Written 2026-10-26. Status: **PROPOSED, pending review and authorisation.** The user authorised this plan only. Not authorised and not done: extraction for development, creating or using a development split, predictions, tuning, E-015. This file amends nothing; docs/DECISIONS.md D-062 (e) and the ledger are unchanged. Everything below was computed from the archive's `meta_data.csv` read in memory, with no model and no extraction.

## 1. Archive state (checked 2026-10-26)

| Item | Value |
|---|---|
| File | `D:\Final Year Project\FakeAVCeleb_v1.2.zip`, 6,400,608,769 bytes, last write 2026-10-26 00:53:19; size and time unchanged since the first inspection |
| SHA-256 (computed earlier today, not recomputed) | `2bdf8734e81898cf7275edac62e614ee9acf68e206a46439410e3f8245cd8bff` |
| Integrity | CRC test of every member passed (earlier today) |
| Contents | 21,544 MP4, 501 text files, 1 CSV with 21,566 rows; 22 metadata paths occur twice |
| Extracted | No |
| Version | "v1.2" comes from the file name only; no version is stated by the source |
| Licence | "released under the FakeAVCeleb Request Forms"; the form's terms are **not yet recorded** (needed in DATA_LOG before use); D-020: derived data may not be published |

Category counts (rows of `meta_data.csv`): RealVideo-RealAudio 500, RealVideo-FakeAudio 500 (RTVC voice cloning), FakeVideo-RealAudio 9,709, FakeVideo-FakeAudio 10,857 (my earlier report of 10,835 was an arithmetic slip; the method counts 4,587 + 3,553 + 2,717 sum to 10,857). Fake clips total 21,066 against 500 real: one real clip for every 42 fakes.

## 2. Correction to my earlier report

The earlier statement "578 distinct target identities, so groups are connected components of source and target ids" was wrong. The `target1` and `target2` columns hold mixed tokens (numbers, `fake.mp4`, `faceswap`, YouTube-style ids), not only identities. Parsing the identity ids (`idNNNNN`) from the clip file names and the target columns instead gives:

- 500 source identities, 500 target identities, every target is also a source.
- The identity graph (source linked to each target of its clips) has **exactly 10 connected components of 50 identities each**. They are the ten race x gender cells (African, Asian East, Asian South, Caucasian American, Caucasian European, each men and women). In all 25,114 source-target pairs found, the target has the same race and gender as the source.
- Each source has 17 to 38 distinct targets (mean 28.0).
- Clips with two target identities: 5,082; with one: 14,950; with none parsed: 1,534 (the 1,000 real-video clips plus 534 fake clips whose target could not be parsed; **these 534 must be inspected before any split is built**).
- Each of the 500 sources has exactly one real-video real-audio clip. Every fake derived from that clip shares its file-name stem and source id; these derivative clips must travel with the source identity.

Consequence: leakage-free splitting cannot be done below the cell level by identity alone, because inside a cell every identity is linked to many others.

## 3. Split options (retention figures are expectations from a seeded random simulation of identity assignment; no model was involved and no manifest was written)

| Option | Rule | Leakage | Retained | Drawback |
|---|---|---|---|---|
| A. Cell split | Whole race x gender cells go to development or sealed (for example 5 and 5) | None by construction | 100% of clips; 250 real clips per side | The two partitions differ in race and gender; a difference between development and sealed results cannot be separated from demographics |
| **B. Identity halves inside cells (recommended)** | Within each cell, assign identities to two halves by a salted hash; keep a clip only if its source and all its targets lie in the same half; mark the rest `dropped_cross` | None: no identity, as source or target, appears in both halves | Mean 46.6% of all clips (range 45.4% to 47.8% over 200 random assignments), about 4,500 fakes per half (the 1,000 real-video clips are always kept: 250 real-audio-real-video clips per half on average) | Loses about 57% of the fake clips (arithmetic from the 46.6% above, not separately simulated); the half sizes are random, not exactly equal |
| C. Single sealed cell | One or two cells sealed | None | High | Very few real clips (50 per cell); wide intervals |

Recommendation: **B.** It keeps both partitions demographically matched and the only cost is clips, of which there are plenty. A is the fallback if B's manifest checks fail.

Assignment must be fixed before anything is looked at: the salt is written to DATA_LOG, the assignment is a deterministic function of the identity id (hash parity), and the manifest records the rule so anyone can recompute it. Which half is sealed is decided by the same rule, not by examining outcomes.

## 4. Manifest (to be built only after authorisation; not built)

Path: `data/_manifests/fakeavceleb_v1.2.csv` (ignored by git; its SHA-256 recorded in DATA_LOG). One row per unique archive member:

`member_path, member_crc, category, type_code, method, source_id, target_ids, cell, partition (dev, sealed, dropped_cross), drop_reason, duplicate_of, has_audio, width, height, fps, duration_s, codec, bitrate` (the last six columns come from an ffprobe census after extraction).

## 5. Leakage and integrity checks (scripts, no model)

1. No identity appears in both partitions, as source or as target.
2. No file-name stem (derivative family) appears in both partitions.
3. Duplicate metadata paths (22) resolved to one row each and flagged.
4. Byte-level duplicates across partitions (hash of the extracted files).
5. Every member listed exactly once; counts per category, method and cell per partition printed and compared with this plan.
6. The 534 clips with no parsed target: classified and assigned by their source.
7. Cross-dataset: FakeAVCeleb reals are VoxCeleb2; LAV-DF is also built from VoxCeleb2 but its metadata has no identity ids, so overlap with LAV-DF cannot be verified by id. Overlap with FF++ and Celeb-DF celebrities is likewise unverifiable. Recorded as an unresolved limitation, not as "no overlap". The frozen D-061 heads were trained on FF++ and ASVspoof, so for them FakeAVCeleb is cross-dataset up to unverifiable celebrity overlap; the I-6 LAV-DF heads are **not** part of the frozen system and any use of them here must flag the VoxCeleb2 family.

## 6. Confounders to audit before any model result is trusted

- Class imbalance (1:42) and unequal category sizes: report each fake category against the real class separately, never pooled accuracy.
- Audio channel: RTVC output is clean synthetic speech; real clips are VoxCeleb interview audio. A detector may separate recording channel rather than synthesis (M-004 saw a sign reversal on another set). Metadata-only baseline first (container, codec, bit rate, sample rate, loudness) on the development half.
- Method confound: visual fakes are Wav2Lip (9,602 clips), FSGAN, FaceSwap, and FSGAN/FaceSwap + Wav2Lip combinations; Wav2Lip only edits the mouth region.
- Labels are video-level; no temporal segments; a "fake audio" clip is fake throughout.
- Clips derived from the same source clip are strongly correlated: all uncertainty intervals use a **bootstrap over identities (cell-stratified)**, never over clips.

## 7. Intended uses and what each can establish

| Use | Partition | Establishes | Does not establish |
|---|---|---|---|
| Frozen face detector on FaceSwap/FSGAN/Wav2Lip video (E-015 part 1) | sealed | Cross-dataset behaviour on two generators unseen in training; FaceSwap is close to FF++'s Deepfakes family, so it is not an unseen-method test | Generated video (Veo, Sora, Kling, Runway); detection after strong compression |
| Frozen speech detector on RTVC audio (E-015 part 2) | sealed | One 2019-era voice-cloning method against VoxCeleb audio | Modern TTS; robustness to channel |
| Independent modules on multimodal media (module behaviour: visual score on RVFA clips, audio score on FVRA clips) | sealed for E-015; dev for engineering | Whether each module responds only to its own modality | Anything about fusion; D-062 keeps video fusion "undetermined" |
| Lip-sync / cross-modal module development | dev | Sync response to Wav2Lip and to cloned audio, once preregistered | Real-world false-positive rate on dubbed content |

## 8. Preregistered procedure outline (E-015 option B; to be written into EXPERIMENTS.md before extraction, only if authorised)

1. Hypotheses and failure criteria written first (CLAUDE.md rule 2). Predictions written to a file and hashed before any scoring ("predictions first", D-062 e).
2. Extract once into `data/fakeavceleb/` (new directory); census; build the manifest; run all section-5 checks; stop if any fails.
3. Sealed half kept in a separate directory reachable only with the existing `--final` guard (conventions of `data/_final` and `src/dfd/eval` to be checked before use).
4. Score the sealed half **once** with the frozen checkpoint (I-5-freeze-candidate, thresholds unchanged). Metrics: per-category AUC against real (video score for categories with fake video; audio score for categories with fake audio), sensitivity and specificity at the frozen thresholds, all with identity-cell bootstrap intervals (2,000 resamples).
5. Report whatever results, including a null. No threshold, calibration or architecture decision uses the sealed half.
6. Development half: free for engineering and for new preregistered experiments, with the three-seed rule where models are trained.

## 9. Proposed amendment (NOT applied)

D-062 (e) now reads: "If FakeAVCeleb or WildDeepfake arrive before Wednesday 2026-10-28 12:00, E-015 (predictions first) scores each once as an extra sealed set after E-014; otherwise they are reported as not run."
Proposed addition for review: "FakeAVCeleb is scored on its identity-matched sealed half only (docs/FAKEAVCELEB_SPLIT_PLAN.md, option B); its development half is available for development. E-015 runs only after the user's explicit authorisation." This needs a new D-entry when approved.

## 10. Still needed before any step

1. User authorisation to extract and to execute E-015 (the deadline in D-062 e is Wed 2026-10-28 12:00; if not executed by then E-015 is reported as not run).
2. Request-form terms, source URL, version and hash entered in DATA_LOG.
3. Inspection of the 534 clips without a parsed target.
