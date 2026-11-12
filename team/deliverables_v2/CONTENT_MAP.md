# CONTENT MAP (evidence for the v2 paper, report, deck and website)

Written 2026-10-28 by the evidence workstream, read-only over the repository. Every number below is copied from the source named next to it.
Source keys: EXP = docs/EXPERIMENTS.md (entry E-xxx, RESULT block), DEC = docs/DECISIONS.md, PB = docs/PERFORMANCE_BENCHMARK.md,
DL = docs/DATA_LOG.md, TR = docs/TESTING_RECORD.md, RS = docs/RESEARCH.md, GF = docs/GEMINI_FEASIBILITY.md, RI = docs/RESEARCH_INTEGRITY.md,
FR = scripts/eval/benchmark_audit_2026-10-27/frozen_rates_output.txt, CK = checkpoints/I-5-freeze-candidate/metadata.json (git-ignored, read 2026-10-28).
Labels: **[SEALED-CONF]** sealed confirmatory (T1 to T3 only); **[SEALED-EXPL]** read once on a sealed or held-out split but exploratory by the
protocol; **[EXPL]** development slice, validation or development data; **[POST HOC]** computed after the fact from stored files, no model run;
**[ENG]** engineering verification, not accuracy. Authority rule used: an EXP RESULT block or a stored result file outranks prose (DOC_FACTS,
paper, VIVA_PREP). "not measured" means no logged number exists. IDs (E-, D-, R-) are for traceability only: DOC_FACTS section 1 says they do
not appear in visible text of the deliverables (comments, notes or appendix index only).

---

## 1. Identity

- **Title (all four deliverables):** Multimodal Deepfake Detection for Images, Video and Audio. No subtitle. (SPEC.md)
- **Never use** the old subtitle "A Hypothesis-Driven Study Under a Group-Disjoint Protocol" (still in paper/main.tex line 50, `\papersubtitle`).
- **Author order (SPEC.md, user instruction 2026-10-28):** Shivansh Gupta (16014023052), Vaibhav Shiroorkar (16014024805), third student
  (16014024803). SPEC.md leaves the third name as the placeholder `{THIRD_AUTHOR}` "set by the main agent". docs/IDENTITY.md spells the student
  with roll 16014024803 as **Adithya Jithesh**. Use the value the main agent writes into SPEC.md; if it stays a placeholder, IDENTITY.md is the
  only recorded spelling.
- **Conflict to know:** docs/IDENTITY.md (2026-10-25) and the current paper list Vaibhav Shiroorkar first; SPEC.md (2026-10-28, later user
  instruction) puts Shivansh Gupta first. SPEC.md governs v2. No "first author" labels, no e-mails, no contribution statements.
- **Institutional fields (IDENTITY.md, exact):** University: Somaiya Vidyavihar University. School: K. J. Somaiya School of Engineering.
  Department: Electronics and Computer Engineering. Degree: Bachelor of Technology (Electronics and Computer Engineering). Cover wording:
  "Submitted In Partial Fulfillment of Requirements / For the Degree Of / Bachelor of Technology / (Electronics and Computer Engineering) / By".
  Guide: Dr. Tejashree Phatak, Assistant Professor. Head of the Department: Dr. Jagannath Nirmal. Examiner and Director: labels only, no names.
  Address: Vidyavihar, Mumbai - 400 077. Place (certificate, declaration): Mumbai-77. Date: left blank. Year on cover: 2026-2027.
  Report footer: "Department of Electronics and Computer Engineering Semester VII 2026-27 Batch  Page <n>". Certificate class wording:
  "LY B. Tech, Semester VII". Slides: no footer wording (department name only where a department line is needed).
- Omitted everywhere (exist in no template): contact details, e-mails, examiner and director names, submission date.

## 2. Project story (one paragraph)

We set out to build one local system that checks images, videos and the speech inside videos for manipulation, and to show a documented path
in which every change starts from a written hypothesis and is judged on data split by source identity. The first working version (I-0) used
one frozen pretrained encoder per task with a linear head, temperature scaling and a validation-chosen threshold, chosen by a measured encoder
sweep in which the cheapest candidate statistically tied with the leader wins. Its diagnostics showed that the frozen face-video feature
underfits even its training data and that the speech model collapses out of corpus. We therefore fine-tuned the video encoder (SigLIP base)
and, for speech, trimmed silence, augmented the training clips and averaged three encoders; two further video levers did not enter, the
audio-visual fusion on LAV-DF turned out to rest on a production-pipeline cue, and the pulse (rPPG) check failed. Read once on sealed tests,
the final system improved over the first version on the Celeb-DF family (AUC 0.8218 against 0.6238) and on In-the-Wild speech (0.9148
against 0.7516), and against a re-run Xception it is superior at the declared 95 percent level but not robust at the Bonferroni level
(non-inferior at both). After the freeze we wrapped the frozen detectors in a modular platform with per-module findings, cautious per-module
labels, no overall score, a dashboard and an Evaluation Lab, and ran further studies that added limits rather than claims: FakeAVCeleb carries
a codec shortcut and the frozen thresholds do not transfer; generated-video heads are confounded by video source; the generated-image head is
inadequate on modern generators and a retrained head ranks better but flags about half of real images; lip sync is weak evidence. Unsolved:
reliable detection of modern generated video and images across real sources, thresholds that transfer, and any Gemini-specific study (none
was run).

## 3. Objectives and scope

**Claimed (with intervals, on the stated data only):**
- A reproducible protocol: group-disjoint splits, development slices separate from sealed tests, sealed tests read once per frozen iteration
  through a guarded, ledger-logged command, three seeds, 95 percent group-bootstrap intervals, hypotheses and lines written before runs.
- Face-video (T-V) and speech (T-A) detectors with sealed results and three pre-declared confirmatory tests (section 9).
- A generated-image detector (T-I) with sealed exploratory results on ProGAN-era test sets.
- A local analysis platform (8 modules) that reports per-module results with tested scope and limitations, no fused score.

**Not claimed:** audio-visual detection or a fused verdict (LAV-DF fusion is a pipeline cue); pulse-based detection (S-1 failed); generated-video
detection (all heads experimental, APPROVED_HEADS empty); generated-image detection on modern generators (E-028 INADEQUATE); lip-sync
deepfake detection; temporal localisation; real-time operation; state of the art; "better than" any published detector except where a
category A comparison exists, and there it is stated as a difference; any Gemini capability; robustness to social-media processing or edited
media (not tested). Emotion analysis was removed from scope by the user (D-069). Visible-logo or watermark detection is excluded by the user
(D-077) and must not appear even as future work.

## 4. Architecture

### 4.1 Frozen detector pipeline (final system, tag I-4 of E-014; definition D-061, unchanged by D-065)

| Task | Input and preprocessing | Encoder | Head | Temperature (fitted on) | Threshold (max balanced accuracy on calibrated validation probabilities, D-024) |
|---|---|---|---|---|---|
| T-V face video | 16 uniformly spaced frames, YuNet detector (MIT), largest face enlarged 1.3, 224 x 224 crop, JPEG 95; clip score = mean of frame probabilities; frames without a face left out | SigLIP base (google/siglip-base-patch16-224, IMG-17), fully fine-tuned (E-005: lr 1e-5, 1,800 steps, batch 32, AdamW, wd 0.05) | linear | 0.8321 (FF++ c23 validation, 700 clips) | 0.6138830927259571 (shown as 0.614 or 0.61) |
| T-I generated image | whole image through the encoder's own processor (face crop instead if a face is found, see 4.3) | CLIP ViT-L/14 OpenAI (IMG-13), frozen | linear probe | 0.6360 (progan_val) | 0.2072609853917272 (0.2073) |
| T-A speech | first 64,600 samples at 16 kHz after trimming leading and trailing silence and peak normalisation, repeat-padded | mean of logits of HuBERT base (AU-8), WavLM base plus (AU-4), Whisper small encoder (AU-6), each frozen | linear head per encoder | 0.4543 (ASVspoof 2019 LA dev, 24,844 clips) | 0.6954489212423093 (0.695) |

Source: CK; E-004, E-005, E-009, E-015 ("temperature 0.832, threshold 0.614", "threshold 0.695"). The checkpoint folder is named
`I-5-freeze-candidate` (checkpoint_id I-5-freeze-candidate-20261025-1608, metadata field iteration "I-5") although E-014 calls the final system
tag I-4; it holds the D-061 system (T-V from I-1, T-I from I-0, T-A from I-4). In E-014, tag "I-5" means the LAV-DF pre-check. Write "the
frozen final system", not "I-5", in visible text. No fusion in the checkpoint (fusion: null).

### 4.2 v1 inference library and web page (frozen, protected)

Local Flask page, 127.0.0.1 only, 512 MB upload limit, files deleted after each request; fixed response structure (score per modality,
status, warnings, errors, model block, a fixed sentence); a development stand-in always shows a MOCK banner; the real adapter refuses a
checkpoint whose metadata or preprocessing id does not match (DOC_FACTS 10, D-029, D-044). The v1 page shows a per-file label "Likely fake",
"Likely real" or "Undetermined" (src/dfd/web/static/app.js line 56) and the sentence "Calibrated probability that this content is fake"
(dfd.infer.schema.SCORE_MEANING). Both are history only (frozen path, recorded as a known inconsistency in D-087). The v1 screenshots were
taken against the I-0 checkpoint on training items (E-004); the page was checked once with the freeze checkpoint on three training files
(DOC_FACTS 16, data/_results/freeze/web_freeze_check.json).

### 4.3 Post-freeze analysis platform (src/dfd/platform)

- **Module interface:** `applicability(probe)` gives applicable, not_applicable or unsupported; `run(ctx)` returns one module result.
- **Coordinator:** modules run one at a time (shared GPU), each with a time limit; exception gives failed, no answer in time gives timed_out,
  a contract-breaking result gives failed (INVALID_RESULT); a timed-out thread is abandoned (late result discarded); a GPU module is skipped
  while an abandoned GPU module runs; a total time limit skips modules not started; every module appears exactly once (DOC_FACTS 16).
- **Report schema 2.1** (schema.py SCHEMA_VERSION "2.1", accepts "2"): sections authenticity, cross_modal, provenance; statuses completed,
  not_applicable, unsupported, failed, timed_out, skipped. **Findings vocabulary:** authenticity {indicates_manipulation,
  no_indication_within_tested_scope, inconclusive}; cross_modal {consistent, potential_mismatch, inconclusive}; provenance {credentials_valid,
  credentials_invalid, no_credentials_found, inconclusive}. No value means "genuine". `FORBIDDEN_REPORT_KEYS = ("fused", "overall_score",
  "overall_probability", "probability_fake", "verdict")` are rejected by the contract checker. Fixed statement: "This report lists what each
  module found within its tested scope. A module that found nothing, did not run, was unsupported or was inconclusive does not show that the
  media is authentic. No single probability that the media is fake is given." 2.1 adds optional module `explanation`, `evidence` (offset_curve,
  frames) and report `summary`; explanations are fixed server-side templates over the module's own fields, validated (no number absent from cited
  fields, no forbidden wording, no experimental score as a key figure), no language model (D-076, D-082). They describe the stored result; they
  are not saliency or causal explanations.
- **Per-module verdict layer (D-087, verdict.py, assessment version v1):** an additive `assessment` per completed module: verdict code and
  label, tier, basis, raw {score, score_kind, threshold}, calibration {is_probability: always false, statement}, uncertainty {level, text},
  reasons, evidence ids (verdict_evidence.py, each checked against its result file by a test), not_established. It re-derives `score >=
  threshold`; a missing, non-finite or out-of-range score, a missing threshold (real modules) or a finding that disagrees gives "Inconclusive"
  with the reason. Failed, timed-out or skipped give "Analysis failed"; unsupported gives "Unsupported input". **No report-level verdict, no
  fused or overall score.** Tiers: detector, limited, experimental, provenance, context, none.
- **Labels in force (D-087 plus addendum of 2026-10-28):**

| Module and path | Tier | Flag label (uncertainty) | No-flag label (uncertainty) | Evidence behind the wording |
|---|---|---|---|---|
| face_video | detector | Possible manipulation (high) | No strong evidence of manipulation detected (high) | E-015 specificity 0.996, sensitivity 0.4804; post hoc FPR 0.1098 Celeb-DF family, 0.1405 DFD (FR) |
| image_detector, face_crop | limited | Possible manipulation (weak evidence) (high) | Inconclusive (high) | E-028: T-V on face photos FPR 0.2152, FNR 0.6621 |
| image_detector, whole_image | limited | Possible AI-generation (weak evidence) (high) | Inconclusive (high) | E-028: pooled FPR 0.0418, pooled FNR 0.7395 |
| speech | limited | Possible manipulation (weak evidence) (high) | No strong evidence of manipulation detected (moderate) | E-015 audio specificity 0.6367, sensitivity 0.9615; E-030 In-the-Wild EER 0.1578 |
| lip_sync (cross_modal) | limited | Possible audio-visual mismatch (weak evidence) (high) | No audio-visual mismatch found (high) | E-025 |
| generated_image, generated_video | experimental | always "Experimental signal observed, no reliable verdict available" (high), whatever the score | same | E-026, E-027, E-028, E-029 |
| media_metadata | context | "Information only" | | |
| content_credentials | provenance | "Content credentials verified" / "Content credentials invalid" / "No content credentials found" | | |

  "Likely AI-generated" exists in the vocabulary but no rule reaches it (a unit test pins this). "Likely manipulated" was removed from the
  vocabulary on 2026-10-28 (D-087 addendum, user decision) because it rested on the FakeAVCeleb specificity alone.
- **Dashboard (D-075, D-076, D-082):** Flask on 127.0.0.1; live mode with one worker, refusal at 4 queued or running jobs, 512 MB limit,
  uploads deleted; upload-first page; groups: merits attention, nothing found within tested scope, credential verified, inconclusive, file
  information, experimental, did not apply, not available, did not complete; one rounded number per check, raw fields under "Technical
  details"; no overall score; mock scenarios (11 fixtures) only on request and marked MOCK; light and dark palette with WCAG AA contrast computed
  in a test; lip-sync offset curve and the 8 generated-video input frames shown as evidence (display only).
- **Evaluation Lab (D-088, src/dfd/platform/evallab, separate dashboard page):** re-analyses recorded score files (dfd.eval scorefile schema)
  with the project's metrics and group bootstrap. Runs no model, trains nothing, writes no ledger entry, never reads or writes APPROVED_HEADS,
  thresholds, checkpoints or detector configuration. Safeguards: a final split (test_in, test_x) only as a re-analysis of an evaluation already
  in the ledger, only with purpose "final_reanalysis", and only if the score file SHA-256 equals the ledger entry (refusals FINAL_PURPOSE,
  FINAL_GUARD); thresholds only from validation rows of the same runs or given with provenance, never fitted on evaluated rows; leakage checks
  (groups shared with train or val refuse a final run, LEAKAGE; duplicate sample ids; duplicate file hashes when present); configuration,
  score-file SHA-256 and code state frozen into the run record before computing; comparisons only on identical samples. Known limit: it cannot
  prove training data excluded the evaluated items when the score file does not record them. A good lab result is not a product approval.
  Software check 2026-10-28: re-analysis of I-4 on the sealed Celeb-DF family reproduced AUC 0.8218, a development-purpose sealed run was
  refused, ledger bytes unchanged (TR 10).

### 4.4 Module inventory (8 modules; pipeline.default_modules)

| Module id | Section | What it does | Model | Tier | Score kind (verdict layer) | `calibrated` flag in result | Can it give a finding? |
|---|---|---|---|---|---|---|---|
| face_video | authenticity | Face manipulation in the largest face of a video, up to 16 frames | T-V fine-tuned SigLIP base | detector | model_output | True | Yes (score >= 0.6139) |
| image_detector | authenticity | Still image: if a face is found, the largest face crop by the T-V head (path=face_crop); else the whole image by T-I (path=whole_image) | T-V head / T-I CLIP ViT-L/14 + linear head | limited | model_output | face_crop: False; whole_image: True | Yes (T-V 0.6139 or T-I 0.2073) |
| speech | authenticity | Synthetic or converted speech in the first ~4 s of an audio file or a video's audio track | logit mean of HuBERT base, WavLM base plus, Whisper small | limited | model_output | False (exploratory calibration) | Yes (score >= 0.6954); no speech check of its own (a 440 Hz tone gets a finding, D-075) |
| generated_image (D-083) | authenticity | Only when image_detector took the face path: frozen T-I on the whole image | T-I (frozen) | experimental | ranking_score | False | No: always inconclusive, threshold null, warning EXPERIMENTAL_PORTRAIT_SCOPE |
| lip_sync | cross_modal | Audio-visual offset of one tracked face; consistent if offset in band (-6, +2) frames, potential_mismatch outside; inconclusive if confidence < 3.0, non-speech-like audio, < 2 s, several faces, offset at the +-15 boundary | SyncNet v2 port (syncnet-v2-port-3) | limited | none (score is SyncNet confidence scaled for display) | False, threshold null | Cross-modal finding only (weak evidence); never states manipulation |
| generated_video | authenticity | Canonical-v3 re-encode of the first 2 s, 8 of 14 frames, frozen DINOv2 small, mean class-token embedding, logistic regression (E-020 C2) | E-020 C2 head (appearance baseline, not ReStraV) | experimental | ranking_score | False, threshold not reported | No: APPROVED_HEADS is empty, always inconclusive, warning EXPERIMENTAL_UNVALIDATED |
| media_metadata | provenance | Container and stream facts | none | context | none | no score | finding always inconclusive; label "Information only" |
| content_credentials | provenance | C2PA manifest validation via c2pa-python 0.38.0 | none | provenance | none | no score | credentials_valid / invalid / no_credentials_found / inconclusive; untrusted signer is inconclusive; valid credential never means authentic content |

Sources: src/dfd/platform/modules/*.py, provenance/modules.py, verdict.py, pipeline.py. Note: earlier records (TR section 4, the paper's
software paragraph, DOC_FACTS 16) say "all seven modules"; that was before generated_image was added on 2026-10-27 (D-083). Current count: 8
(TR 10 browser run: "4 checks ran ... 4 did not apply" on a video).

## 5. Score semantics

| Concept | What it is in this project | Which outputs | Code / source |
|---|---|---|---|
| Ranking score | A number whose only reliable property is order; AUC uses only ranks | Experimental heads: generated_video ("Uncalibrated ranking score ... It is not a probability"), generated_image; verdict layer score_kind "ranking_score". On sources unlike the validation split every detector score is effectively a ranking score | generated_video.py DEFAULT_SCORE_MEANING; generated_image.py SCORE_MEANING; base.py SCORE_MEANING_V2 |
| Validation-fitted calibration | One scalar temperature T, p = sigmoid(logit(p)/T), fitted on the detector's own validation split; `calibrated: True` means only "a temperature was fitted on validation data" (D-087) | face_video (FF++ val), image_detector whole_image (progan_val) | CK calibration fields; base.py SCORE_MEANING_V2 |
| Exploratory calibration | Temperature fitted on ASVspoof dev, but out-of-corpus ECE rose after scaling (E-004: 0.3293 to 0.4052 for AU-8 on dev_x) | speech (`calibrated: False`, warning AUDIO_CALIBRATION_EXPLORATORY) | speech.py; base.py SCORE_MEANING_EXPLORATORY |
| Calibrated probability | Only for content like the validation split; never product semantics. The platform's `calibration.is_probability` is always false | none in the platform; the frozen v1 sentence "Calibrated probability that this content is fake" is history | verdict.py `_out`, PROBABILITY_STATEMENT; D-087 |
| Threshold classification | finding = indicates_manipulation if score >= threshold, else no_indication_within_tested_scope; threshold = max balanced accuracy on calibrated validation probabilities | face_video, image_detector, speech | base.py finding_for; verdict.py _score_problem re-derives it |
| Categorical finding | Schema finding vocabulary per section | every completed module | schema.py FINDINGS |
| Categorical verdict label | Module-specific label from the verdict layer (4.3) | every completed module; never report-level | verdict.py VERDICTS, _PROFILES |

Wording to use (SPEC.md): "temperature-scaled on each detector's validation split (a probability only for content like that split); the
speech score has an exploratory calibration; experimental heads give uncalibrated ranking scores." Evidence that scores are not probabilities
elsewhere: E-015 (visual sensitivity 0.4804 at specificity 0.996; audio specificity 0.6367) and E-028 (T-I misses 0.7395 of modern generated
images; T-V flags 0.2152 of real face photographs) (D-087).

## 6. Datasets

| Dataset (version) | Task | Role | Size as documented | Licence / terms status | Source |
|---|---|---|---|---|---|
| FaceForensics++ c23 (official split, Deepfakes, Face2Face, FaceSwap, NeuralTextures) | T-V | train 3,600 clips, val 700, test_in 700 (70 groups) sealed | 5,000 videos (1,000 originals, 1,000 per method), 9.04 GB, no audio | FaceForensics Terms of Use: non-commercial research and education only | DL Downloaded; E-004; E-014 |
| Celeb-DF v2 | T-V | dev_x 1,728 clips (146 real, 1,582 fake, 15 identity groups); test_x 4,801 (744 real incl. 300 YouTube-real, 4,057 fake) sealed | 6,529 videos (590, 300, 5,639), 59 identity groups | Celeb-DF terms: non-commercial research only, no reproduction or publication of any portion or derived data | DL; V-002 |
| Celeb-DF++ (21 new generators) | T-V | sealed test_x, fakes only: 2,849 (44 identity groups) after removing 2,079 dev_x-identity samples and 150 byte-identical reals | 5,078 files extracted of 54,087 | Celeb-DF++ terms "as it is"; derived data not published, no frame in any document | DL Celeb-DF++ sections; D-045 |
| Celeb-DF family (derived) | T-V | sealed test_x for T1 and T3: 7,650 videos (744 real, 6,906 fake), 344 groups (44 identities plus 300 YouTube videos); 9.7 percent real | derived | as above | D-045; E-014 |
| DFD (Google DeepFakeDetection, inside FF++) | T-V | sealed test_x only: 3,431 (363 real, 3,068 fake), 28 actors | c23 | FF++ Terms of Use | DL V2; V-002 |
| ForenSynths ProGAN validation set | T-I | train 14 classes (5,600 images), val 6 classes (2,400) | 8,000 images, 20 LSUN classes | CC BY-NC-SA 4.0 (HF card); LSUN terms not established | DL; E-004 |
| ForenSynths ProGAN test set | T-I | sealed test_in, 8,000 images (20 groups) | 0.83 GB | CC BY-NC-SA 4.0 | DL; E-014 |
| ForenSynths CNN-synthesis subset (8 generators downloaded) | T-I | dev_x 2 units (StarGAN 3,998, whichfaceisreal 2,000; 5,998 images); test_x 48,805 images (16 groups) sealed | 56,003 files | CC BY-NC-SA 4.0 | DL; E-002; E-014 |
| UnivFD diffusion test sets | T-I | sealed test_x 10,000 images (8,000 fake, 2,000 real; 10 groups) | 0.92 GB | image licence not established (code MIT) | DL; E-014 |
| ASVspoof 2019 LA | T-A | train 25,380; val (dev) 24,844; sealed test_in 71,237 (63,882 fake, 7,355 real; 67 speakers) | LA.zip 7.64 GB | ODC-By | DL; E-014 |
| In-the-Wild (Muller et al.) | T-A | dev_x 6,991 clips (12 speakers; 3,804 real, 3,187 fake); sealed test_x 24,788 (42 speakers; 16,159 real, 8,629 fake) | 31,779 clips, 54 speakers | CONFLICT unresolved: HF card CC BY-SA 4.0; a search summary says Apache 2.0 | DL A4; V-002; FR |
| LAV-DF | audio-visual | subset of official splits, 500 videos per cell (real, video-fake, audio-fake, both) per split: train 2,000, val 2,000, sealed test_in 2,000 (1,736 groups; 1,500 fake, 500 real); E-011 used 10,000 head-training videos; E-022/E-024/E-025 used official-train real clips | 136,304 videos (train 78,703, dev 31,501, test 26,100), all with audio | CC BY-NC 4.0; terms: non-commercial research and education only; VoxCeleb2 agreements apply (not found as text) | DL LAV-DF sections; DOC_FACTS 15 |
| FakeAVCeleb v1.2 | audio-visual | identity split fakeavceleb-79d3f7051972: dev_x 5,041 clips (250 identities), sealed test_x 5,013 (250 identities; 500 real video, 4,513 fake video; 2,725 real audio, 2,288 fake audio), 11,490 excluded | 21,544 clips | "released under the FakeAVCeleb Request Forms"; form terms NOT recorded; derived data not published | DL V6; D-070; E-015 |
| DeepAction v1 | generated video | development only (E-018 to E-020, E-026) | 2,605 clips: 100 real (Pexels), 2,505 generated by 6 generators | generated folders CC BY 4.0; real folder Pexels licence | DL G1 |
| DF26 | generated video | held-out evaluation (E-020, once); open-generator subset (6 generators, 1,626 clips + 271 real) for training under leave-one-generator-out (D-079, E-027); closed subset (Grok Imagine, Kling 3.0, Veo 3.1, Wan 2.6; 794 clips) evaluation only, one counted look per experiment | 2,691 clips (271 real, 2,420 fake, 10 generators) | df26-evaluation-only-license v1.0: non-commercial; model updates prohibited except open-subset LOGO training; no redistribution; no frames in documents | DL G2 |
| Community Forensics-Small (subset I6-S) | generated image | E-029 training 23,807 images (14,000 fake from 125 generators; 9,807 real COCO, FFHQ, VISION) and validation 5,333 (3,172 fake from 31 generators; 2,161 real) | 15 of 186 shards, 11.27 GB | CC BY-NC-SA 4.0, non-commercial research only | DL I6; E-029 |
| Community Forensics-Eval (subset I7-S) | generated image | test only (E-028, E-029): 11,550 images (5,759 fake, 5,791 real), 21 generators | 92 of 413 shards | CC BY-NC-SA 4.0 | DL I7; E-028 |

**Split protocol:** split once, group-disjoint by source identity (the group is the source video together with the person identity; speaker
for speech; generator unit for T-I dev_x). Splits: train, validation, development slice dev_x (ranks candidates, may be evaluated repeatedly,
exploratory), test_in (in-dataset sealed), test_x (cross-dataset sealed). Sealed tests are read once per frozen iteration by `python -m
dfd.eval.run --final`, which writes ledger/final_ledger.jsonl and refuses a second read without `--repeat` and a logged reason (E-014: 74
ledger entries, no `--repeat`). A script checks group disjointness including content-hash de-duplication (V-002: passes on 80,803 image files
and 41,739 Celeb-DF, In-the-Wild and DFD files). Celeb-DF++ samples of the 15 dev identities are excluded from test (D-045). Never used: DFDC,
WildDeepfake, FF++ c40, AV-Deepfake1M (PB 13.1).

## 7. Methodology

- **Encoder sweep (E-001 to E-003, D-036, D-049):** frozen encoders, features extracted once, linear head (class-balanced BCE, AdamW lr 1e-3,
  wd 1e-4, batch 1,024, up to 200 epochs, patience 10), 3 head seeds; rank by mean dev_x AUC; tie set = candidates whose paired 95 percent
  group-bootstrap interval against the leader includes zero; **cost rule:** cheapest in the tie set wins (end-to-end extraction time per 1,000
  inputs, then parameters); from-scratch reference reported and excluded; permutation test as sensitivity check. DINOv3 excluded on purpose
  (manual gate, acknowledgement duty). Shortlist: images EfficientNet-B4, Xception, DINOv2 small and base, CLIP ViT-L/14 OpenAI and LAION, SigLIP
  base; audio wav2vec 2.0 base, WavLM base plus, Whisper small, HuBERT base.
- **Metrics (PROTOCOL.md, MATH_NOTES.md, D-012, D-024, D-025):** headline ROC-AUC with 95 percent group-bootstrap interval (2,000 resamples,
  base seed 20261024); EER; balanced accuracy at the validation threshold; log loss; ECE (15 equal-width bins); Brier; temperature scaling.
  Seeds: 3 per head; fine-tuning 1 seed per learning rate then 2 more for the winner. Wilson intervals for proportions in post-freeze studies.
- **Staircase:** every frozen iteration (I-0, I-1 to I-4, BASE, I-5, I-6 and variants) scored once on the sealed sets; everything outside
  T1 to T3 is exploratory.
- **Three confirmatory tests (D-061 amended by D-062 b, D-066):** T1 T-V final minus I-0 on celebdf_family test_x; T2 T-A final minus I-0 on
  in_the_wild test_x ("improved" if the lower bound is above 0); T3 T-V final minus the Xception baseline on celebdf_family test_x: superior if
  lower bound > 0, non-inferior if > -delta, an interval containing 0 but not -delta is "not distinguished, non-inferior by the rule of D-025".
  Each also reports a 98.33 percent robustness line (Bonferroni over 3, same bootstrap draws); it never replaces the declared verdict.
- **Margin delta = 0.0100** (D-066): max(0.01, 2 x max(seed sd validation 0.0003, dev_x 0.0020)) for the Xception baseline at lr 3e-4; the
  floor decides. (First pass E-013 gave 0.0143, superseded.)
- **Baseline (E-013, E-013b):** Xception (ImageNet weights) fine-tuned on FF++ with the E-005 recipe; lr chosen on validation from {1e-5, 3e-5,
  1e-4, 3e-4}; 3e-4 selected (edge of the extended grid, a stated limit).

## 8. Development history (improvement tables)

Comparability flags: "NC" marks a pair of results that must not be set side by side as like for like.

### 8.1 Development ladder (exploratory unless marked)

| Step | Previous version | Limitation found | Change | Result | Interpretation | Remaining limitation | Evidence | Label |
|---|---|---|---|---|---|---|---|---|
| Sweep (I-0 encoder choice) | none | need evidence for encoder choice | 7 image and 4 speech encoders ranked under the cost rule | T-V tie set IMG-07, IMG-17, IMG-08, IMG-15, IMG-13; winner IMG-07 DINOv2 small, dev_x 0.6933 [0.6639, 0.7314] (leader IMG-13 0.7464 [0.6721, 0.8306]); best non-CNN minus best CNN +0.0994 [+0.0228, +0.1723]. T-I winner IMG-13 0.9935 [0.9803, 0.9982] (2 units). T-A tie set AU-8, AU-4, AU-6, winner AU-8 HuBERT 0.6100 [0.5434, 0.7256]; verdict INCONCLUSIVE (best lower bound 0.5338) | frozen self-supervised and vision-language features beat supervised CNNs on face video; frozen speech features did not transfer | 15 identities, 2 generator units, 12 speakers: wide intervals | E-001, E-002, E-003; data/_results/T-*/rank.md | EXPL |
| I-0 working version | sweep winners | T-V train AUC only 0.8367 (bias); validation minus dev_x +0.105 (shift); T-A validation minus dev_x +0.384, real In-the-Wild speech scored as fake; temperature did not transfer (T-V ECE 0.2048 to 0.2217, T-A 0.3293 to 0.4052) | frozen encoder + linear head, temperature, validation threshold; page on real checkpoint | T-V val 0.7985, dev_x 0.6933; T-I val 1.0000, dev_x 0.9935; T-A val 0.9942, dev_x 0.6100; parity max difference 0.0019 (tolerance 0.02) | the path works; metrics with intervals exist | no accuracy claim beyond intervals | E-004; data/_results/I-0/ | EXPL |
| I-1 fine-tuning (T-V) | I-0 frozen probe | bias (train AUC 0.8367) | full fine-tuning of 3 encoders x 3 lr (seed 0), winner by validation + cost: IMG-17 SigLIP base, lr 1e-5, 3 seeds | val 0.9899 [0.9809, 0.9967] (probe 0.8767); dev_x 0.9096 [0.8776, 0.9388] (probe 0.7396); gain +0.1700 [+0.1228, +0.2114]; training-subset AUC 0.9994; higher lrs collapsed (sensitivity 0.40 to 0.47 AUC) | SUPPORTED (C1 to C3 met); gain larger than predicted | NeuralTextures weakest on val (0.969); dev_x errors concentrate in identity id1 | E-005 | EXPL |
| BASE (re-run Xception) | none | need a published-family baseline under the same protocol | Xception fine-tuned with the E-005 recipe; grid extended to 3e-4 | 3e-4: val 0.9912 [0.9849, 0.9963], dev_x 0.8716 [0.8364, 0.9076]; IMG-17 minus BASE dev_x +0.0380 [+0.0172, +0.0592] | baseline for T3; delta 0.0100 | selected rate again at grid edge | E-013, E-013b, D-066 | EXPL |
| I-1b augmentation + mixup (T-V) | I-1 | gap val minus dev_x 0.0803 (trigger not met to the letter; run on user request) | compression, blur, resize augmentation with mixup | val 0.9705 [0.9618, 0.9795], dev_x 0.9315 [0.9055, 0.9548]; vs I-1 val -0.0194 [-0.0251, -0.0129], dev_x +0.0219 [+0.0090, +0.0382] | DOES NOT ENTER (validation loss beyond 0.0079 allowance): a trade-off, not a failure | no sealed test of I-1b | E-016, D-065 | EXPL |
| I-1c logit mean of 3 fine-tuned encoders (T-V) | I-1 | score correlation 0.89 to 0.91 | mean logit IMG-17, IMG-08, IMG-07 | val 0.9903, dev_x 0.9192 [0.8948, 0.9457]; vs I-1 dev_x +0.0096 [-0.0020, +0.0231] | DOES NOT ENTER (no gain the interval can show) | | E-017, D-065 | EXPL |
| I-2 speech preprocessing | I-0 T-A | trivial recording cues reverse sign between corpora (M-004: leading silence, RMS) | trim leading and trailing silence, peak-normalise | dev_x 0.6723 [0.6126, 0.7584]; gain +0.0623 [+0.0114, +0.1048]; val 0.9889 | SUPPORTED (manipulation check passed) | RMS still AUC 0.135 [0.054, 0.382] on In-the-Wild | E-007 | EXPL |
| I-3 nuisance augmentation | I-2 head | short, padded, saturated errors | noise, low-pass, reverberation, MP3 round trip, 3 copies per clip | dev_x 0.7455 [0.7052, 0.8121]; gain +0.0732 [+0.0122, +0.1225] | SUPPORTED | 12 speakers: gain below about +0.09 not showable | E-008 | EXPL |
| I-4 three-encoder average | I-3 AU-8 head | single encoder | mean logit of AU-8, AU-4, AU-6 | dev_x 0.8130 [0.7523, 0.8808]; gain +0.0675 [+0.0300, +0.0929]; val 0.9964 | SUPPORTED; pairwise correlation 0.534 | three encoder passes at inference | E-009 | EXPL |
| I-5 LAV-DF pre-check | frozen heads | need audio-visual evidence | score frozen heads on LAV-DF val | visual 0.3838 [0.3599, 0.4079] vs modify_video (below chance); audio 0.6476 [0.6244, 0.6701] | NOT TESTED: FF++ head does not respond to LAV-DF visual edits | | E-010 | EXPL |
| I-6 in-domain heads + fusion | I-5 | pre-check failed | heads trained on LAV-DF, logistic-regression fusion | visual 0.7491 [0.7280, 0.7697], audio 0.8604 [0.8435, 0.8769], fused 0.9935 [0.9905, 0.9962]; post hoc control: visual within produced videos 0.5070 [0.4771, 0.5361]; E-012 produced-only head 0.5169 [0.4945, 0.5392] | E-011 SUPPORTED mechanically but driven by a production-pipeline cue; E-012 NOT SUPPORTED; no audio-visual detection claim | the audio head is the part that detects manipulation | E-011, E-012, D-059 | EXPL |
| I-7 rPPG pulse check | none | rPPG as candidate cue | signal-sanity check S-1 on 432 real clips with shuffled control | 0.301 [0.260, 0.346] above threshold, required 0.6: FAILS; positive control M-005: injected 1.2 Hz sinusoid recovered at relative green amplitude 0.008 in 0.873 of clips | "rPPG signal not recoverable from these data"; nothing further run | no pulse ground truth; M-005 shows pipeline sensitivity only | E-006, M-005, D-052 | EXPL |

### 8.2 Freeze (I-8) and sealed reading

| Step | Previous version | Limitation found | Change | Result | Interpretation | Remaining limitation | Evidence | Label |
|---|---|---|---|---|---|---|---|---|
| I-8 integration and freeze; staircase; T1 to T3 | exploratory ladder | all verdicts exploratory | no model change; every tag scored once on sealed sets | T1 +0.1980 [+0.1688, +0.2242] (98.33: [+0.1627, +0.2303]) improved, robust; T2 +0.1633 [+0.1177, +0.2266] ([+0.1085, +0.2481]) improved, robust; T3 +0.0176 [+0.0025, +0.0327] ([-0.0011, +0.0355]) superior at 95 percent, not robust at Bonferroni, non-inferior at both | headline result | 344 video groups, 42 speakers; sealed video lower than dev_x (0.8218 vs 0.9096, NC: different data), sealed speech higher (0.9148 vs 0.8130); 2 of 5 predictions failed upward (speech) | E-014, D-067, D-068 | SEALED-CONF (T1 to T3), rest SEALED-EXPL |

### 8.3 After the freeze

| Step | Previous version | Limitation found | Change | Result | Interpretation | Remaining limitation | Evidence | Label |
|---|---|---|---|---|---|---|---|---|
| Analysis platform | v1 page with one label per modality | v1 wording overstated scores; no tested scope per check | modules, coordinator, schema 2 then 2.1, no global score, dashboard | 8 modules (section 4.4); safe suite 713 passed (TR 12, 13) | presentation layer, no detector change | | D-069, D-076, D-082 | ENG |
| E-015 FakeAVCeleb sealed half | frozen system | no audio-visual sealed set | score sealed identity half once | visual 0.9552 [0.9442, 0.9657]; audio 0.9263 [0.9149, 0.9366]; H3 visual RVRA vs RVFA 0.5613 [0.5428, 0.5816], audio 0.5011 [0.4801, 0.5243]; at frozen thresholds visual sensitivity 0.4804, specificity 0.996; audio sensitivity 0.9615, specificity 0.6367 | all four pre-stated criteria met; thresholds do not transfer | codec shortcut (E-021) and method-codec confound (E-023): whether 0.9552 is inflated is not estimated | E-015, e015_results.json | SEALED-EXPL (extra sealed set, not one of T1 to T3) |
| E-021 codec audit (dev half) | E-015 | RVRA vs RVFA gap 0.56 | metadata and pixel shortcut audit; canonical re-encode | dev RVFA vs RVRA 0.5517 [0.5325, 0.5729] falls to 0.5012 [0.4806, 0.5213] after re-encode; metadata-only video target 0.9985 [0.9961, 1.0]; pixel statistics 0.9585; frozen visual 0.9732 [0.9642, 0.9816] before, 0.8772 [0.8562, 0.8981] after (PARTLY RETAINED); wav2lip 0.9548 to 0.7944 | FakeAVCeleb carries a large codec shortcut (fake video mpeg4, real h264 except 26 faceswap clips) | 1,000-clip dev sample (NC with the sealed 0.9552) | E-021 | EXPL |
| E-023 codec-matched control | E-021 | residual 0.877 unexplained | give real clips an mpeg4 generation | B 0.8763 [0.8548, 0.8963]; codec-matched P1 0.8856 [0.8643, 0.9056] (+0.0092 [+0.0006, +0.0180]); S1 0.9059; "manipulation evidence remains" by the preregistered rule | codec history alone does not explain the residual | codec fully confounded with method; sealed inflation not estimated | E-023 | EXPL |
| E-018 to E-019 generated-video data audit | none | candidate corpus | metadata-only audit; canonical re-encode v1, v2, v3 | E-018 metadata AUC 1.0000 (shortcut indicated); v1 colour-tag leak (REMAINS); v2 failed rule by GBT upper bound 0.6048; v3 PASS (all 21 header fields constant by construction); pixel statistics still 0.773 [0.729, 0.817] / 0.787 [0.743, 0.827] | header shortcut removed by construction; not shortcut-free | v3 is the third protocol tried on the same data (not a blind test) | E-018, E-019 v1 to v3 | EXPL |
| E-020 generated-video heads | none | need a video generator check | C1 ReStraV-style trajectory MLP, C2 mean DINOv2-small CLS + logistic regression, C3 pixel statistics; DeepAction LOGO dev, DF26 held out once | DeepAction LOGO: C1 0.5073 [0.4608, 0.5542], C2 0.8281 [0.7908, 0.8662], C3 0.7613 [0.7182, 0.8033]; DF26 C2 all 0.4957 [0.4755, 0.5168], closed 0.4880 [0.4534, 0.5204] (H-G1 NOT SUPPORTED) | C2 installed as experimental only (D-072), gate closed (D-074) | DeepAction 100 real clips from one library | E-020 | EXPL (DeepAction), SEALED-EXPL (DF26 held-out once) |
| E-026 source-confound test | E-020 C2 | DeepAction result possibly source-confounded | score unmodified real clips of FF++ val and LAV-DF train | flagged at head threshold 0.9623: pooled 338 of 440 = 0.7682 [0.7265, 0.8052] (FF++ 0.5143, LAV-DF 0.8867) vs DeepAction real out of fold 0.19 [0.1251, 0.2778] | H-X2 SUPPORTED: head separates "this library" from everything else | driving property not identified | E-026 | EXPL |
| E-027 DF26 open-generator heads | E-020 | wrong training data | train on DF26's 6 open generators, leave-one-family-out; one counted closed look | open LOFO P8-IMG-13 (CLIP, mean of 8 frames) 0.7453 [0.7174, 0.7710]; closed 0.7548 [0.7189, 0.7885] vs 200 scenario reals, 0.7800 [0.7437, 0.8109] vs all 271 reals; temporal features add nothing (PT14 - P14 -0.0404 [-0.0650, -0.0165]); confound re-check: 0.8023 [0.7625, 0.8368] of 440 FF++ and LAV-DF real clips flagged at t_BA | H-X3 and H-X3c SUPPORTED but CONFOUND PERSISTS: not installed (D-085) | second look at closed subset with reused reals; NC: E-020 closed 0.4880 used 271 reals, compare with 0.7800 | E-027 | EXPL (open), SEALED-EXPL (closed look) |
| E-028 modern generated images | frozen T-I | ProGAN-era head untested on modern generators | score 11,550 Community Forensics-Eval images, nothing fitted | macro AUC 0.7039 [0.6855, 0.7229]; pooled AUC 0.7973 [0.7900, 0.8048]; pooled FNR 0.7395 [0.7300, 0.7500]; pooled FPR 0.0418 [0.0368, 0.0470]; GAN 0.9580, commercial 0.7351, latent diffusion 0.5808; format-only AUC 0.9301 macro; PNG-only T-I AUC 0.6123 | INADEQUATE by the preregistered line | format and size confounded with the label | E-028 | SEALED-EXPL (test-only set, one look) |
| D-083 portrait whole-image check | image_detector face path | still images with a face never reached T-I (UX-7) | experimental module generated_image runs frozen T-I on the whole image, no finding | E-028 face-routed images (3,119): T-I AUC 0.7859 [0.7717, 0.7995], FNR 0.8063; T-V face path 0.6020 [0.5836, 0.6202], FPR 0.2152; paired +0.1839 [+0.1596, +0.2074] | "better than the face path" met; "usable on faces" not met | shown as experimental only | D-083, E-028 | ENG + SEALED-EXPL |
| E-029 retrained image head | frozen T-I | ProGAN training data | same recipe on 125 recent generators (CF-Small), validated on held-out generators, one Eval look | macro AUC seed mean 0.8839 [0.8732, 0.8942] (seed 0: 0.8863 [0.8756, 0.8965]); paired +0.1800 [+0.1643, +0.1959] over T-I; pooled FPR at its validation threshold 0.5213 [0.5091, 0.5339] (LAION 0.5885, COCO 0.0328); JPEG control pixel diffusion -0.1086 | H-29 SUPPORTED; NOT ELIGIBLE, kept out (D-086) | threshold does not transfer to unseen real sources | E-029 | SEALED-EXPL |
| E-030 speech EER | E-014 AUC | literature reports EER | EER from stored sealed scores, no model run | I-4 ASVspoof 0.0694 [0.0595, 0.0782], In-the-Wild 0.1578 [0.1269, 0.1883]; I-0 0.0897 and 0.3091 | made ASVspoof rows category A | freeze report already held EER (intervals [0.0594, 0.0779], [0.1275, 0.1870]; primary record per correction) | E-030, D-084, data/_final/eval/I-4.json | POST HOC on sealed scores |
| Lip-sync E-022 | placeholder port-1 | MIN_CONFIDENCE 1.0, no speech check | 40 controlled cases; then port-2: confidence 3.0, speech heuristic, boundary rule | BEFORE: 1000 ms shift 14 and 12 of 20 potential_mismatch (H-B failed); non-speech 1 of 50 flagged; AFTER: 20 of 20 inconclusive, 50 of 50 inconclusive | changes chosen on the same clips: not a test | | E-022 | EXPL |
| Lip-sync E-024 | port-2 frozen | need held-out check | fresh clips, two sources | false alarms on real speech 9 of 60 = 15.0 percent [8.1, 26.1] (LAV-DF), 8 of 60 = 13.3 percent [6.9, 24.2] (FakeAVCeleb): FAILED; in-range shifts 176 of 180 = 97.8 percent [94.4, 99.1]; Wav2Lip flagged 85 to 93 percent (all at -3 frames) | NOT validated; +-2 tolerance sits inside natural spread | | E-024 | EXPL |
| Lip-sync E-025 band | port-2 | +-2 frames flags ~14 percent of real clips | band fitted on 230 real clips, validated on 230 disjoint | band [-6, +2]; false alarms 3 of 230 = 1.3 percent [0.4, 3.8] (old rule 27 of 230 = 11.7 percent); Wav2Lip flagged 0 of 40 in each of 4 strata; -120 ms shift flagged 14 of 28, +120 ms 2 of 28 | port-3 installed (D-073); lip sync is weak evidence, not a detector | LAV-DF pools split by file not speaker; both sources VoxCeleb2-derived | E-025 | EXPL |
| D-087 verdict wording | platform v2 findings | "Calibrated probability" text overstated scores | per-module assessment, cautious labels, is_probability false | labels in 4.3 | presentation only; no threshold change | autonomous per-module assignments need user review | D-087 | ENG |
| D-087 addendum | "Likely manipulated" | rested on FakeAVCeleb specificity alone; post hoc FPR 0.1098 (Celeb-DF family), 0.1405 (DFD) | label "Possible manipulation", uncertainty high | | user decision 2026-10-28 | | D-087 addendum; FR | POST HOC rates |
| D-088 Evaluation Lab | none | need re-analysis without test tuning | lab on dfd.eval safeguards | 57 tests passed (TR 13); I-4 re-analysis 0.8218 reproduced | engineering | cannot see unrecorded training data | D-088, TR 10 | ENG |

## 9. Headline results

### 9.1 Sealed (E-014; AUC, 95 percent group bootstrap, seeds 0 to 2)

| Set (n, groups) | I-0 working version | Final system (I-4) | Re-run Xception (BASE) | Label |
|---|---|---|---|---|
| Celeb-DF family test_x (7,650, 344) | 0.6238 [0.6015, 0.6486] | 0.8218 [0.8020, 0.8393] | 0.8042 [0.7841, 0.8223] | basis of T1, T3 |
| Celeb-DF v2 test_x (4,801, 344) | 0.6322 [0.6018, 0.6658] | 0.8999 [0.8759, 0.9210] | 0.8527 [0.8237, 0.8777] | SEALED-EXPL |
| Celeb-DF++ fakes vs v2 reals (3,593, 344) | 0.6119 [0.5821, 0.6427] | 0.7105 [0.6877, 0.7331] | 0.7351 [0.7083, 0.7599] | SEALED-EXPL |
| DFD test_x (3,431, 28) | 0.5927 [0.5629, 0.6258] | 0.8797 [0.8566, 0.9041] | 0.8847 [0.8655, 0.9052] | SEALED-EXPL |
| FF++ c23 test_in (700, 70) | 0.7946 [0.7667, 0.8243] | 0.9899 [0.9826, 0.9959] | 0.9895 [0.9823, 0.9952] | SEALED-EXPL |
| In-the-Wild test_x (24,788, 42) | 0.7516 [0.6802, 0.8035] | 0.9148 [0.8896, 0.9383] | | basis of T2 |
| ASVspoof 2019 LA test_in (71,237, 67) | 0.9684 [0.9564, 0.9777] | 0.9835 [0.9798, 0.9872] | | SEALED-EXPL |
| ProGAN test_in (8,000, 20) | 1.0000 [0.9999, 1.0000] | same head | | SEALED-EXPL |
| CNN-synthesis test_x (48,805, 16) | 0.9316 [0.8933, 0.9694] | same head | | SEALED-EXPL |
| Diffusion test_x (10,000, 10) | 0.9354 [0.8967, 0.9665] | same head | | SEALED-EXPL |

Speech staircase (In-the-Wild / ASVspoof): I-0 0.7516 / 0.9684; I-2 0.8125 [0.7361, 0.8560] / 0.9632 [0.9535, 0.9717]; I-3 0.8672 [0.8320,
0.8989] / 0.9541 [0.9452, 0.9626]; I-4 0.9148 / 0.9835. Video staircase on the family: I-0 0.6238, I-1 0.8218 (I-1 to I-4 visual identical).
LAV-DF sealed subset (test_in 2,000, 1,736 groups) [SEALED-EXPL]: I-5 visual 0.3226 [0.2953, 0.3490]; I-6 visual 0.9921 [0.9880, 0.9956],
fused 0.9945 [0.9917, 0.9969]; I-6bp within produced 0.5203 [0.4973, 0.5429]; I-6p audio within produced 0.8480 [0.8281, 0.8670].

**Confirmatory tests [SEALED-CONF]:** T1 +0.1980 [+0.1688, +0.2242], 98.33 percent [+0.1627, +0.2303]: improved, robust. T2 +0.1633 [+0.1177,
+0.2266], [+0.1085, +0.2481]: improved, robust. T3 (delta 0.0100) +0.0176 [+0.0025, +0.0327], [-0.0011, +0.0355]: "superior at the 95 percent
level, not robust to the Bonferroni level; non-inferior (above -0.0100) at both levels" (D-068 wording). Source: E-014; verified against
data/_final/eval/compare_T3.json (PB 12.1 row 7).

### 9.2 Speech EER (post hoc on sealed scores; E-030, D-084)
I-4: ASVspoof 2019 LA 0.0694 (seed SD 0.0008) [0.0595, 0.0782]; In-the-Wild 0.1578 (0.0020) [0.1269, 0.1883]. I-0: 0.0897 [0.0763, 0.1053];
0.3091 [0.2663, 0.3698]. The freeze report data/_final/eval/I-4.json (2026-10-25) already held 0.0694 [0.0594, 0.0779] and 0.1578 [0.1275,
0.1870]; either interval may be cited; the freeze report is the primary record (E-030 correction).

### 9.3 FakeAVCeleb sealed identity half (E-015) [SEALED-EXPL]
Visual AUC 0.9552 [0.9442, 0.9657] (500 real, 4,513 fake video); per method faceswap 0.9730, fsgan 0.9871, faceswap-wav2lip 0.9908,
fsgan-wav2lip 0.9895, wav2lip 0.9275 [0.9104, 0.9443]. Audio AUC 0.9263 [0.9149, 0.9366]. At frozen thresholds: **visual sensitivity 0.4804,
specificity 0.996** (TP 2,168, FN 2,345, FP 2, TN 498); audio sensitivity 0.9615, specificity 0.6367 (TP 2,200, FN 88, FP 990, TN 1,735).
EXPERIMENTS.md prints 0.480 and 0.637; result file e015_results.json gives 0.4804 and 0.6367 (use the file values). Reading: ranking high,
thresholds do not transfer, codec shortcut in the data (E-021).

### 9.4 Post hoc threshold rates on sealed scores (FR; seed means at the per-seed validation thresholds; no model run) [POST HOC]
Final system visual: Celeb-DF family FPR **0.1098**, TPR 0.5896; Celeb-DF v2 FPR 0.1098, TPR 0.7172; Celeb-DF++ with v2 reals TPR 0.4079; DFD
FPR **0.1405**, TPR 0.7323; FF++ FPR 0.0143, TPR 0.9554. Final speech: In-the-Wild FPR 0.1716, TPR 0.8560; ASVspoof FPR 0.0165, TPR 0.8092.
I-0 speech In-the-Wild FPR 0.8379. BASE: family FPR 0.0524, TPR 0.3977; DFD FPR 0.0882. T-I (I-4 file): CNN-synthesis FPR 0.0358, TPR 0.6562;
diffusion FPR 0.0347, TPR 0.6813; ProGAN FPR 0.0021. Celeb-DF v2 official-list part (399 of 518 videos in test_x, 110 groups, post hoc,
unregistered): I-4 0.8478 [0.7998, 0.8884], BASE 0.7615 [0.7096, 0.8087], I-0 0.6205 [0.5604, 0.6806] (PB 12.4). Accuracy is uninformative
on these sets (9.7 percent real in the family; 89.7 percent spoofed in ASVspoof).

## 10. Comparisons with published results (PB classes A to D; only bib-cited sources can appear in v2)

**Category A, directly comparable (difference = ours minus theirs; published values have no intervals, so no significance test):**

| Comparison | Ours | Theirs | Difference | Reading | bib key |
|---|---|---|---|---|---|
| ASVspoof 2019 LA eval pooled EER, I-4 vs AASIST | 6.94 percent [5.95, 7.82] | 1.13 percent (mean of 3 seeds) | +5.81 points | ours worse; AASIST is end-to-end for this corpus | jung2021aasist |
| same, vs frozen XLS-R 2B + logistic regression (Pascu et al.) | 6.94 percent | 0.5 percent (plus or minus 0.1) | +6.44 points | ours worse; 2B-parameter encoder; R-083 trains on LA train and dev | pascu2024generalisable |
| same, vs RawNet2 under the Pascu protocol | 6.94 percent [5.95, 7.82] | 5.9 percent (plus or minus 0.1) | +1.04 points | sign not established (our lower edge 5.95 touches their 6.0) | pascu2024generalisable |
| DF26 zero-shot pooled AUROC, E-020 C2 vs nine FF++-trained detectors | 49.57 [47.55, 51.68] | 69.7, 63.5, 61.6, 56.6, 54.7, 54.2, 52.6, 48.2, 44.0 | -20.1, -13.9, -12.0, -7.0, -5.1, -4.6, -3.0, +1.4, +5.6 | ours at chance; no ordering among values near 50; our head trained on DeepAction, 2 s at 224 px | shykula2026df26 |
| Celeb-DF family, final vs our re-run Xception (T3) | 0.8218 [0.8020, 0.8393] | 0.8042 [0.7841, 0.8223] | +0.0176 [+0.0025, +0.0327] | see T3 wording | chollet2017xception (network), our own run |
| Same pair, single sets (point estimates, no paired interval) | FF++ 0.9899; Celeb-DF v2 0.8999; Celeb-DF++ w/ v2 reals 0.7105; DFD 0.8797 | 0.9895; 0.8527; 0.7351; 0.8847 | +0.0004; +0.0472; -0.0246; -0.0050 | comparable on FF++ and DFD; higher on Celeb-DF v2; lower on Celeb-DF++ generators | own run |
| In-the-Wild AUC, I-4 vs I-0 (T2; our own implementation of the frozen-encoder recipe family of Pascu et al.) | 0.9148 | 0.7516 | +0.1633 [+0.1177, +0.2266] | improved, robust | pascu2024generalisable (recipe family) |

**Category B (same dataset, different protocol; no difference computed), only rows whose source is in refs.bib:** Muller et al. In-the-Wild
EER 33.9 percent (full set) vs ours 0.1578 on 42 of 54 speakers (muller2022inthewild); Pascu XLS-R 2B In-the-Wild 7.2 percent (full set)
(pascu2024generalisable); Pascu Table 3 frozen base encoders WavLM base plus 38.6, wav2vec 2.0 base 45.9 vs our dev_x EER 0.3930 and 0.5020
(same ordering, exploratory); UnivFD diffusion group mAP 95.00 vs our mean AP 0.9387 over the same 8 sets (computed 2026-10-27; 8,000 vs
720,000 training images) (ojha2024universal); Community Forensics card value for a ProGAN-trained CLIP probe mAP 0.583 (card value, quoted)
and paper mAP 0.987 vs our subset macro AP 0.6690 (E-028) and 0.8535 (E-029) (park2025communityforensics; no gap computed); DeepfakeBench
frame-level Xception 0.7365 on Celeb-DF v2 vs our video-level values (yan2023deepfakebench); DeepAction accuracy 97.0 percent vs our LOGO AUC
0.8281 (bohacek2025deepaction); DF26 Table 3 closed-source AUROC PwTF-DVD 48.3, DFD-FCG 34.5 vs E-027 closed 0.7548 (zero-shot vs ours trained
on open generators) (shykula2026df26); BA-TFD on full LAV-DF 0.990 vs our fused 0.9945 on a 2,000-video subset (cai2022lavdf; fused value is a
pipeline cue).
**Category C/D (context only):** FakeAVCeleb in-dataset baselines (khalid2022fakeavceleb) vs our zero-shot E-015 values; ReStraV 98.63 AUROC on
VidProM vs our C1 adaptation (interno2025restrav). **Not citable in v2 (not in refs.bib although some are READ/PARTIAL in RS):** SBI, Effort,
Forensics Adapter, LipForensics, AVAD, SyncNet (Chung and Zisserman), Wav2Lip, C2PA specification, MRDF, AVFF, Fake or JPEG. Rows that need
them (Effort 0.956, SBI 93.18 and 97.56, Forensics Adapter 0.957, LipForensics 82.4, AVAD) must be left out of v2 unless the main agent adds
the bib entries. PB 13.8: **no category A row exists for the face-video task against any published external detector.**

## 11. Why experimental detectors were not accepted

- **Generated-video heads.** E-020 C2 reached DeepAction LOGO 0.8281 [0.7908, 0.8662] but DF26 0.4957 [0.4755, 0.5168] (closed 0.4880
  [0.4534, 0.5204]); E-026 showed it flags 0.7682 [0.7265, 0.8052] of 440 unmodified FF++ and LAV-DF real clips vs 0.19 of DeepAction reals out
  of fold. E-027's CLIP head reached closed 0.7548 [0.7189, 0.7885] but still flags 0.8023 [0.7625, 0.8368] of the same 440 real clips at its
  balanced threshold (FF++ 105 of 140 = 0.7500, LAV-DF 248 of 300 = 0.8267; DF26 reals out of fold 0.2841), so CONFOUND PERSISTS; it also needs a CLIP feature path the
  module lacks (D-085). Temporal features added nothing. APPROVED_HEADS stays {} (D-074); promotion needs a preregistered independent
  validation with fresh real clips from many sources and a passing source-confound control (PB 4.5).
- **Retrained image heads (E-029).** Ranking improved (macro AUC 0.8839, +0.1800 [+0.1643, +0.1959]) but at its validation threshold the
  frozen candidate flags 0.5213 [0.5091, 0.5339] of real Eval images (LAION 0.5885, RAISE 0.4959, ImageNet 0.2488, FFHQ 0.1522, COCO 0.0328),
  and JPEG re-encoding lowers pixel-diffusion AUC by 0.1086 (line was below 0.10). Product eligibility line (AUC lower bound >= 0.85, FPR upper
  bound <= 0.10, JPEG drop < 0.10 per architecture): NOT ELIGIBLE (D-086). The threshold is not re-fitted on Eval (that would be test tuning).
- **Whole-image portrait check (D-083).** On 3,119 face-routed images T-I AUC 0.7859 [0.7717, 0.7995] but FNR 0.8063 (misses most generated
  faces); "usable on face images" needs AUC lower bound >= 0.85 and FPR upper bound <= 0.10: not met. It beats the face-swap path on the same
  images (+0.1839 [+0.1596, +0.2074]), so it is shown as an experimental card, no finding, threshold null; promotion needs a preregistered
  portrait validation (D8 open).
- **Lip sync is weak evidence.** Real speech spreads across -6 to +2 frames; Wav2Lip clips lie at -3 and -2 (inside); under the band 0 of 160
  Wav2Lip clips are flagged, and a -120 ms shift is flagged in 14 of 28 clips; offset -3 occurs in 21 of 224 (9.4 percent) real calibration
  clips (E-025). Pools split by file on the LAV-DF side (speaker overlap possible). SyncNet weights licence not stated.
- **Frozen T-I on modern generators (E-028)** remains a ProGAN-era detector: INADEQUATE (macro AUC upper bound 0.7229 < 0.75; pooled FNR lower
  bound 0.7300 > 0.50); its image_detector flag is therefore labelled "weak evidence" and a low score "Inconclusive".

## 12. Gemini-generated images

- **No Gemini experiment was run.** No Gemini data was downloaded, no model trained, nothing evaluated (GF status line; E-031 status
  "PROPOSED; NOT preregistered, NOT run"). It is an unresolved research direction, not a product capability.
- **Why considered:** user interest in modern commercial generators after E-028 and E-029; the user asked on 2026-10-27 for a data and
  provenance feasibility assessment and a preregistered proposal before any Gemini-specific model (GF). Visible logo or watermark detection is
  excluded (D-077); SynthID verification is an external Google service with no local detector and is not part of the proposal (GF).
- **Nearest measured evidence (not Gemini):** Google Imagen 3, one of the 21 generators of the Community Forensics-Eval subset (203 generated
  against 688 paired real images). Frozen T-I (E-028): AUC 0.7691 [0.7352, 0.8020], missed 0.8719 of generated images at the frozen
  threshold (FPR 0.0422). E-029 head (IMG-13, seed 0): AUC 0.8646 [0.8360, 0.8917], missed 0.0345, but flagged 0.5654 of the paired real images.
  A PNG-only format score separates this cell at AUC 0.9731 (e028_results.json, cue_only_scores). Imagen 3 is not Gemini, and Gemini image
  models are not one model.
- **Unresolved issues (GF section 3):** (1) licence and terms: whether Google's primary terms allow using Gemini outputs to train and evaluate a
  detector is not established (NR until the user reads them); uploader licences of candidate sets (bitmind/nano-banana MIT tag, 9,457 PNG
  1024 x 1024; zr-zhang set apache-2.0 plus research-only, generator per image not stated) are unverified; (2) provenance: no candidate set
  records prompts, generation dates or model build; (3) no matched real controls (format and size cue as in E-028); (4) version coverage
  (Gemini 2.5 Flash Image, Gemini 3 Pro Image, Imagen-based outputs are different generators).
- **What a future preregistered study needs (GF section 4, 4a):** one named Gemini model with recorded provenance; about 1,000 generated
  images (about 600 development, at least 400 untouched test from at least 100 distinct prompts, split by prompt); a different collection and a
  second Gemini version as separate test blocks; real images from at least three sources for the threshold and unseen sources for testing;
  both classes resized to 512 px short side and re-encoded JPEG quality 90 by one function; format-only and size-only baselines; exact and
  near-duplicate audit across splits (64-bit DCT hash, Hamming distance at most 6); models A frozen T-I, B frozen E-029 candidate, C the E-029
  recipe on Gemini data with 3 seeds; threshold fixed in advance (pooled validation FPR at most 0.02 on three real sources); lines: pooled
  test FPR upper bound at most 0.05, each unseen source at most 0.10, TPR lower bound at least 0.80, AUC drop under robustness blocks at most
  0.05; one look; no product use and no APPROVED_HEADS entry without a separate decision. Blocked until the user resolves the terms, provenance
  and real-control questions (GF section 5).

## 13. Limitations (consolidated) and future work

**Limitations.**
1. Small development slices (15 identities, 12 speakers, 2 generator units): wide intervals; every iteration verdict is exploratory.
2. Sealed video intervals rest on 344 groups (about 44 identities plus 300 YouTube videos), heavily imbalanced (744 real vs 6,906 fake) and
   mixing two sources; speech on 42 speakers.
3. T3 uses one baseline family at one input size; delta 0.0100 rests on three seeds and a floor; baseline lr at the grid edge; superiority
   holds at 95 percent only.
4. Every sealed number is one run per frozen iteration; two of five predictions (speech) failed upward; everything outside T1 to T3 is
   exploratory.
5. Thresholds do not transfer: final video TPR 0.5896 and FPR 0.1098 on the family, 0.4804 sensitivity on FakeAVCeleb; speech flags 0.1716 of
   real In-the-Wild clips; scores are not probabilities on other sources; the speech calibration is exploratory.
6. Base-size encoders only (12 GB card; CLIP ViT-L/14 full fine-tuning did not fit, M-003); published leaders use larger adapted encoders.
7. Dataset artefacts inflate AUC: codec in FakeAVCeleb, container and colour tags in DeepAction, PNG vs JPEG and size in Community
   Forensics-Eval, production cue in LAV-DF; controls reduce but do not remove them.
8. No audio-visual detection: probed FF++ and Celeb-DF videos carry no audio; LAV-DF fusion is a pipeline cue.
9. rPPG: no recoverable signal in 432 real clips; no pulse ground truth.
10. Generated video: all heads source-confounded or at chance; 2 s, 7 fps, 224 px canonical clips; v3 header pass is the third protocol tried.
11. Modern images: cost-selected 11,550-image subset (82.5 percent of reals are LAION; 8 generators share 121 RAISE reals); E-029 trained on
    about 1 percent of the published data scale with validation reals from two sources.
12. No category A comparison for face video against any published detector (test subset, 16 vs 32 frames, DFDC never used).
13. Research-only data and weights (section 14); integrity: E-019 to E-025 registrations entered git in one commit with their scripts (order
    rests on the project's own statements, RI 1); E-026 to E-029 preregistrations were hashed into result files but not committed before the
    runs (weaker than a commit); the Evaluation Lab cannot see unrecorded training data.
14. Not tested: edited images, media shared through social networks, other languages for lip sync, natural music and speech over noise for the
    speech gate, fresh clone, other browsers.

**Future work (only items stated in the docs).** Score the frozen face system on the 518-video official Celeb-DF v2 test list with 32 frames,
video and frame level, next to BASE (PB 13.9; needs a `--repeat` decision for the 399 sealed videos); parameter-efficient adaptation of a
larger CLIP encoder on the same sealed split (PB 4.1); a larger frozen speech encoder (XLS-R 300M or 1B) with the same head (PB 4.3); for lip
sync, a learned model over the full time-delay distribution trained on real videos only, with a false-alarm control (PB 4.4); generated-video
training with real clips from many sources and the source-confound control as acceptance test (PB 4.5, E-026, D-085); for the retrained image
head, a new preregistered experiment with a threshold calibrated on diverse real sources and a false-positive line on unseen sources (D-086,
VIVA_PREP); full-scale Community Forensics training (VIVA_PREP); a preregistered portrait validation before the whole-image check could decide
(D-083, D8); full In-the-Wild and DFDC evaluation (PB 13.3; DFDC needs a decision); E-031 Gemini study once its blockers are resolved (GF).

## 14. Ethics and licensing

- **Dataset terms:** all datasets are research-only. Celeb-DF, Celeb-DF++ and FakeAVCeleb forbid publishing derived data, so no frame, crop or
  per-video score of them appears in any document (DOC_FACTS 3, DL rule 4). FakeAVCeleb request-form terms are not recorded (README:
  "released under the FakeAVCeleb Request Forms"; GitHub table "Rights Cleared: No"): use limited to this academic project, no redistribution
  (D-070). LAV-DF: non-commercial research and education, VoxCeleb2 agreements apply (no standalone VoxCeleb2 agreement text found). DF26: no
  frames in documents, no redistribution, closed subset evaluation-only. Community Forensics (CC BY-NC-SA 4.0): no images in public deliverables,
  aggregate results only. In-the-Wild licence conflict unresolved. Diffusion test images and LSUN reals: licence not established.
- **Research-only weights:** OpenAI CLIP ViT-L/14 (IMG-13; no licence in its repo; accepted as research-only, D-038), used by T-I, the
  generated_image module, E-027 and E-029; WavLM base plus (AU-4; no tag, UniSpeech licence CC BY-SA 3.0; CONDITIONAL, research-only), part of
  the T-A ensemble. ImageNet-trained EfficientNet-B4 and Xception treated as research-only. Permissive: SigLIP base (Apache-2.0 tag), DINOv2
  (Apache-2.0), HuBERT base (Apache-2.0 tag), Whisper small (card Apache-2.0, upstream MIT), YuNet (MIT), c2pa-python (MIT or Apache-2.0).
- **E-029 heads** are derived from CC BY-NC-SA 4.0 data: research artefacts, not installed, not redistributed; sharing rights open for the user.
- **SyncNet v2 weights** (`syncnet_v2.model`, SHA-256 961e8696...5442): code MIT, **weights licence not stated by the source**; no permission is
  inferred; used for an academic prototype at the user's instruction (DL G4).
- **What this restricts:** checkpoints and derived features cannot be published or used commercially; dataset media cannot appear in the paper,
  report, deck or website (the dashboard figure fig11 shows no dataset media; screenshots with FF++ and LAV-DF previews stay local, TR 12);
  results are aggregate only.
- **Product ethics:** no module says media is authentic; no overall verdict; experimental signals never decide; C2PA validity concerns origin
  relative to an operator-configured trust anchor, not truth of content.

## 15. Software verification facts [ENG, not accuracy]

Latest (TR sections 12 and 13, 2026-10-28): full safe suite with GPU hidden (`CUDA_VISIBLE_DEVICES=-1`), 13 data-dependent tests deselected:
**713 passed, 2 skipped** (the two opt-in tests), exit 0. Focused: verdict, explain and pipeline tests 112 passed; generated_image (D-083
constraints) 7 passed; Evaluation Lab 57 passed. `uv lock --check` exit 0. Documentation check `check_docs.py --numbers` 71 pass, 0 fail
(numbers traced: paper 1,072 of 1,072, report 1,998 of 1,998, slides 949 of 949). Builds (TR 14): paper 31 pages, 0 undefined references;
report 119 pages, layout 26 of 26; slides 58, checker 955 pass, 0 fail.
Opt-in GPU tests (TR 9, 2026-10-27): 5 passed; wrapper parity on 11 non-sealed files, largest absolute difference 0.001919, tolerance 0.02.
Live browser demo (TR 10 and 12, Edge, real detectors, 8 demonstration inputs from non-sealed sources): **8 of 8 as specified** (6 media files
completed with valid schema 2.1 reports, the unsupported .txt and the corrupt .mp4 failed with UNSUPPORTED_MEDIA_TYPE and UNREADABLE_FILE), 0
page errors, no horizontal overflow at 390 px; face-video card reads "Possible manipulation" (TR 12).
Defects found and fixed: a failed live analysis left the MOCK report on screen (regression test; TR 4); the report overview hidden under the
sticky header (TR 10); the lab confusion-matrix caption (TR 10). Documented, not changed: the speech module scores any sound (a 440 Hz tone
gets a finding, score 0.0); the generated-video score is 1.0 on a real LAV-DF clip and on a synthetic test pattern (TR 4).
History (label as earlier records): 497 passed (2026-10-26), 568 (2026-10-27), 710 (2026-10-28 recovery; the current paper still quotes 710);
live E2E 60 checks, 0 failed; C2PA 5 of 5; generated-video reproducibility 16 of 16. Not run: the 13 data-dependent tests, a fresh clone,
browsers other than Edge, screen readers.

## 16. Figures available

| File | Shows | Status |
|---|---|---|
| assets/figures/fig01_pipeline (.png, .pdf; copy paper/figures/fig01_pipeline.pdf) | three scoring paths, frozen encoders, heads | current: regenerated 2026-10-28 20:00 so it no longer says scores are probabilities (TR 14) |
| fig02_splits | group-disjoint splits, dev slice vs sealed tests | current (schematic) |
| fig03_ladder | table of the measured iteration ladder | current for I-0 to I-7 (2026-10-25); no post-freeze work |
| fig04_staircase | schematic of the staircase of sealed reads | current (schematic) |
| fig05_rppg | processing chain of the pulse check and its control | current |
| fig06_timeline | the seven-day plan 24 to 30 Oct | historical plan only: the freeze actually ran on 25 Oct (D-067), not Thu 29 Oct |
| fig07_web | v1 page against the development stand-in (MOCK) | stand-in only; may appear only labelled as such (DOC_FACTS 10) |
| fig07_web_real | frozen v1 page with the real I-0 checkpoint on three training items | history: shows the v1 page (its labels "Likely fake/Likely real/Undetermined" are history only) |
| fig08_ladder_results | dev_x AUC with intervals along the ladder | current (exploratory values) |
| fig08_shortlist | encoder shortlist with parameters and feature sizes | current |
| fig09_lavdf_cells | LAV-DF per-cell AUC and the produced-video control | current |
| paper/figures/fig10_platform (.pdf, .png, 2026-10-26) | platform structure | partly stale: shows 7 modules (no generated_image module, added 2026-10-27), lip_sync coloured experimental |
| paper/figures/fig11_dashboard.png (2026-10-27 00:13) | top of a live report for demo input 3 (LAV-DF train clip): groups, "No indication...", "Score: 0.32 (decision line 0.61)", 7 checks | stale layout: before the per-module verdict labels (D-087) and before generated_image; no dataset media visible; caption must say so (TR 13). Not used in the current paper (report and slides only) |
| paper/figures/somaiya_vidyavihar_logo.jpg | university logo | identity asset |

Generators: scripts/make_figures.py (fig01 to fig09), paper/figures/make_platform_figure.py, make_dashboard_figure.py. Do not edit them
(protected); v2 workstreams render their own figures in their folders.

## 17. References (paper/refs.bib; all 43 keys are cited in the current paper)

Status from docs/RESEARCH.md. "Model name only" = PARTIAL, bibliographic record only, content not read: cite only to name the model.
E = essential for a 12 to 15 page paper.

| Key | R-id, status | Supports | E |
|---|---|---|---|
| rossler2019faceforensics | R-001 PARTIAL (abstract, dataset, evaluation, benchmark, results) | FF++ dataset, c23 compression | E |
| li2020celebdf | R-002 PARTIAL (abstract, dataset, settings, results, compression table) | Celeb-DF v2 | E |
| li2025celebdfpp | R-053 READ | Celeb-DF++ with 21 newer methods | E |
| dolhansky2020dfdc | R-003 PARTIAL | DFDC exists, not used | |
| li2020facexray | R-004 PARTIAL | generalisation motivation | |
| dong2023implicit | R-044 READ | FF++-trained detectors learn identity cues (motivates identity-disjoint splits) | E |
| yan2023deepfakebench | R-017 PARTIAL (tables 3, 4) | Xception 0.9450 within vs 0.7718 cross-domain average frame AUC (paper intro); frame-level comparison (class B) | E |
| ojha2024universal | R-011 PARTIAL | frozen CLIP linear probe generalises across generators; T-I recipe | E |
| wang2020cnnspot | R-047 PARTIAL | ProGAN training set and CNN-synthesis test (ForenSynths) | E |
| cozzolino2024raising | R-045 PARTIAL | CLIP-based generated-image detection | |
| kornblith2019better | R-018 PARTIAL | transfer-learning caveats | |
| oquab2024dinov2 | R-068 model name only | DINOv2 encoders | E |
| radford2021clip | R-069 model name only | CLIP ViT-L/14 | E |
| zhai2023siglip | R-070 model name only | SigLIP base | E |
| tan2019efficientnet | R-071 model name only | EfficientNet-B4 candidate | |
| chollet2017xception | R-072 model name only | Xception candidate and baseline network | E |
| hu2022lora | R-036 PARTIAL | low-rank adapters (considered, not used) | |
| wu2023yunet | R-077 model name only | YuNet face detector | E |
| ciftci2020fakecatcher | R-005 PARTIAL | rPPG detection claims | |
| ciftci2020hearts | R-006 PARTIAL | rPPG source attribution | |
| hernandez2020deepfakeson | R-007 PARTIAL | heart-rate deepfake detector | |
| seibold2025heart | R-008 PARTIAL | high-quality deepfakes keep a pulse | |
| harraq2026physio | R-009 PARTIAL | reproduction of DeepFakesON-Phys | |
| lin2023ppg | R-010 PARTIAL (title, abstract, introduction only; results not read) | PPG manipulation attack (cite only for its stated aim) | |
| wang2017pos | R-057 READ | POS pulse extraction used in S-1 | E if rPPG is described |
| wang2020asvspoof | R-013 PARTIAL | ASVspoof 2019 LA | E |
| jung2021aasist | R-012 PARTIAL (tables 1 to 3) | AASIST EER 1.13 percent (category A) | E |
| tak2022wav2vecspoof | R-029 READ | wav2vec 2.0 front end for spoofing | |
| muller2022inthewild | R-033 READ | In-the-Wild dataset; 33.9 percent EER (class B) | E |
| baevski2020wav2vec2 | R-073 model name only | wav2vec 2.0 base candidate | |
| hsu2021hubert | R-074 model name only | HuBERT base | E |
| chen2022wavlm | R-075 model name only | WavLM base plus | E |
| radford2023whisper | R-076 model name only | Whisper small encoder | E |
| khalid2022fakeavceleb | R-014 PARTIAL | FakeAVCeleb dataset | E |
| chugh2021notmade | R-015 PARTIAL | audio-visual dissonance detection | |
| cai2022lavdf | R-024 READ | LAV-DF dataset; BA-TFD 0.990 (class B) | E |
| guo2017calibration | R-016 PARTIAL | temperature scaling, ECE definitions | E |
| interno2025restrav | R-078 PARTIAL | ReStraV (our C1 is an adaptation; installed head is not ReStraV) | E |
| shykula2026df26 | R-081 PARTIAL | DF26 benchmark; Table 5 zero-shot AUROC (category A); Table 3 | E |
| bohacek2025deepaction | R-082 PARTIAL | DeepAction dataset | E |
| pascu2024generalisable | R-083 PARTIAL (1, 3, 4.1 to 4.5, Tables 2 to 4) | frozen SSL encoder recipe; EER 0.5, 5.9, 7.2 percent | E |
| park2025communityforensics | R-038 PARTIAL | Community Forensics data and its mAP | E |

Gaps: no bib entry for SyncNet, Wav2Lip, C2PA, VoxCeleb2, DeepAction's Pexels source or Effort/SBI; the lip-sync and C2PA modules must be
described without a citation unless the main agent adds verified entries (PB section 8 holds BibTeX-ready fields for R-084, R-085, R-086).

## 18. "Do not say" list

1. "Calibrated probability that the content/file is fake" as the meaning of any current score; "the probability that the file is fake"; a score
   shown as a percentage of certainty. (DOC_FACTS section 1 still lists "calibrated probability that the content is fake" as wording to use;
   that line is superseded by D-087 and SPEC.md.)
2. "Likely fake", "Likely real" or "Likely manipulated" as a current label (the v1 page's labels and the retired D-087 label are history only);
   "Likely AI-generated" (no rule can reach it).
3. Any overall, fused or combined verdict or score for a file; "the system says this video is fake/real".
4. "Authentic", "genuine", "verified real", "safe" for media; "no manipulation" for a no-flag result (say "no strong evidence ... within the
   tested scope").
5. "Real-time", "state of the art", "best", "outperforms", "better than X" (only category A differences, stated as differences); "beats
   Xception" (T3 is not robust at the Bonferroni level).
6. The subtitle "A Hypothesis-Driven Study Under a Group-Disjoint Protocol".
7. "Audio-visual deepfake detection" or "multimodal fusion" as a capability; "fused AUC 0.99 shows detection" (pipeline cue).
8. "Detects AI-generated video", "detects Sora/Veo/Kling"; calling the installed head "ReStraV"; "validated" or "approved" generated-video or
   generated-image heads; "shortcut-free" for canonical-v3 clips.
9. "Detects modern AI images" or "Gemini detection"; any Gemini result (none exists); "logo or watermark detection" (excluded, D-077).
10. "Lip-sync deepfake detector"; "lip sync detects Wav2Lip" (0 of 160 under the band).
11. "Pulse/rPPG detection works" (S-1 failed).
12. "The FakeAVCeleb AUC shows face-manipulation detection" without the codec shortcut; "the sealed 0.9552 is inflated by X" (not estimated).
13. Calling E-030, the official-list AUCs, the AP values or the FPR rates of 9.4 sealed confirmatory; calling any dev_x or LAV-DF number
    confirmatory.
14. Explanations as "saliency", "attribution" or "why the network decided".
15. "Seven modules" for the current platform (it has 8); "710 tests passed" as the latest count (713).
16. Effort, SBI, Forensics Adapter, LipForensics or AVAD numbers (not in refs.bib); comparing E-028/E-029 macro AUC with Community Forensics mAP
    0.987 as a gap.
17. Dates or schedule claims from fig06 as what happened; statements about which AI tools helped write anything.

---

## Appendix: discrepancies found between sources (for the main agent)

1. Author order: IDENTITY.md and paper (Vaibhav first) vs SPEC.md (Shivansh first); SPEC third name is a placeholder; IDENTITY spelling
   "Adithya Jithesh".
2. Test count: paper/main.tex software paragraph says 710 passed (TR 11); latest TR 12 and 13 say 713 passed. Authoritative: TR 13.
3. Module count: paper, DOC_FACTS 16 and TR 4 say seven modules (pre-D-083); code has 8 (pipeline.default_modules); fig10_platform shows 7.
4. E-015: EXPERIMENTS prints visual sensitivity 0.480 and audio specificity 0.637; the result file gives 0.4804 and 0.6367 (consistent, more
   decimals; use the file).
5. E-030 EER intervals [0.0595, 0.0782] and [0.1269, 0.1883] vs freeze report [0.0594, 0.0779] and [0.1275, 0.1870] (different bootstrap seeds;
   freeze report primary). E-030's premise "E-014 reported AUC only" is true only of the EXPERIMENTS table.
6. Checkpoint name I-5-freeze-candidate (metadata iteration "I-5") holds the system E-014 calls tag I-4; E-014 tag I-5 is the LAV-DF pre-check.
7. DOC_FACTS section 1 wording "calibrated probability that the content is fake" contradicts D-087 and SPEC.md (superseded).
8. CNN-synthesis test_x unit count: PB section 2 says "5 CNN-synthesis units pooled", PB 12.4 and 13.1 say 6 generator folders; E-014 records 16
   groups and 48,805 images. Not resolved here; quote only n and groups.
9. GEMINI_FEASIBILITY and E-031 give Imagen 3 FNR 0.8719; E-028 per-generator line prints 0.872 (same value rounded).
10. E-021 frozen visual AUC on original files 0.9732 (1,000-clip sample, real video = RVRA plus RVFA) vs E-023 0.9768 (RVRA vs 500 fakes):
    different samples, not comparable.
11. NUMBERS_LIT.json lacks the RawNet2 5.9 percent and the Pascu 0.5 percent ASVspoof entry is present; if the v2 checker traces literature
    numbers, 5.9 and the nine DF26 Table 5 values other than 69.7 and 44.0 may not be in NUMBERS_LIT.json.

## Phase B addendum (2026-10-28, evening; main agent)

Added after the hostile review (deliverables_v2/audit/) and the approved Phase B decisions. These supersede the corresponding statements above.

- E-032 (docs/EXPERIMENTS.md): the post hoc false-positive rates of the frozen video threshold, Celeb-DF family 0.1098 (744 real videos) and DFD 0.1405 (363 real videos), re-analysis of sealed scores already in the ledger, not confirmatory. Script scripts/eval/benchmark_audit_2026-10-27/frozen_rates.py; replay output docs/test_evidence/2026-10-28_phase_b/frozen_rates_I4_replay.txt (SHA-256 46e012d6e207b8ce869fc8b1c7bc483e8e4d7d1e899b1d5ebb68120b505d2e11). Quote only in results and limitations, as "about 0.11" and "about 0.14".
- D-090: the submission set is deliverables_v2/; the original paper, report and deck are an unchanged archive.
- D-091: T3 wording: "non-inferior at both levels (margin 0.0100); the declared 95 percent superiority rule was met but does not survive the Bonferroni correction (98.33 percent interval [-0.0011, +0.0355]); the baseline ran at 224 px". "Superior" never appears without this qualifier. Supersedes the wording of D-068.
- Three confirmatory tests (T1, T2, T3) plus the 98.33 percent robustness line were fixed by D-062 (b), committed 2026-10-25 18:44, before the first sealed read (20:16) and the sealed run (ledger committed 22:16). docs/PROTOCOL.md and docs/DOC_FACTS.md section 7 are corrected (record correction of 2026-10-28).
- Software tests, measured 2026-10-28 evening: 713 passed, 2 skipped (opt-in GPU tests), 13 deselected (data-dependent tests).
- Speech T2 shortcut: M-004 (leading silence AUC 0.375 on ASVspoof dev against 0.719 on In-the-Wild dev_x; RMS 0.594 against 0.248; 12 speakers), the sealed staircase step I-0 0.7516 to I-2 0.8125 (+0.061 of the 0.163), E-007 (RMS AUC 0.135 [0.054, 0.382] on In-the-Wild dev_x after the lever).
- Discrepancy 7 above (old wording in DOC_FACTS section 1) stays superseded by D-087.
- Published comparisons: no published number is directly comparable except the re-run Xception baseline; AASIST (1.13 percent), RawNet2 and the DF26 detectors are context only (class C for DF26).
