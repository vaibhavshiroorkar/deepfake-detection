# Threat model

This project produces research evidence about talking-head deepfake detection.
It does not prove that media is authentic. Its score must not decide identity,
law-enforcement, employment, or moderation outcomes by itself.

## Protected assets

- Integrity of the research result.
- Separation of training, validation, and test information.
- Raw media and derived biometric data.
- Model, threshold, and evaluation provenance.
- Users' understanding of uncertainty and failure.

## In-scope inputs

- Talking-head video with one stable primary speaker.
- Face swap, reenactment, synthetic or replaced speech, and combined attacks.
- Lip and audio mismatch that exists during the sampled speech segment.
- Common compression, resizing, frame-rate changes, and background noise.
- Missing or unusable modality evidence, reported through abstention.

## Out-of-scope inputs

- Image-only and audio-only authenticity claims.
- Real-time streaming guarantees.
- White-box adversarial attacks against the trained model.
- Videos without a visible, stable speaking face.
- Identity recognition or attribution to a generator.
- Claims about manipulation types absent from all evaluation sets.
- Deliberate corruption of the local machine or stored model files.

## Attacker profiles

### Opportunistic attacker

The attacker uses a public generator and common post-processing. They do not
know the detector internals.

### Adaptive black-box attacker

The attacker knows common detector cues. They may recompress, add noise, alter
frame rate, replace silence, or choose clips that hide the mouth.

### Dataset attacker

The attacker is not a person. Dataset construction leaks identity, codec,
duration, silence, or generator shortcuts that inflate measured performance.

White-box gradient attacks remain future work. The final report must state that
exclusion beside any robustness claim.

## Main threats and controls

The third column was headed "Current **or** required control", which never
distinguished a control that exists from one the project intends. That ambiguity
hid a real gap for months: "required audio masks" sat here while the audio branch
pooled over padding unmasked in every trained checkpoint. The status column is
now explicit, and a row marked `required` is a commitment, not a safeguard.

| Threat | Failure | Control | Status |
|---|---|---|---|
| Source identity leakage | The model recognizes people instead of fakes | Source-disjoint splits and overlap audits | **in place**, tested |
| Generator shortcut | Scores collapse on an unseen method | Method-family holdouts and external zero-shot evaluation | **partial**: zero-shot done, leave-one-method-family-out never run |
| Codec or resolution shortcut | The model learns dataset encoding | Corruption tests and distribution reports | **partial**: corruption tests done (F24, F25); no distribution report exists |
| Silence shortcut | Audio labels leak through leading silence | Silence ablation and explicit audio quality fields | **partial**: quality fields exist and `cache build --keep-leading-silence` exists, but the cache-level ablation has never been run. F25 measures a half-second shift at inference and finds no effect |
| Padding shortcut | The sync or audio branch learns sequence length | Real-context offsets and required audio masks | **partial**: masking implemented and tested, but no shipped checkpoint was trained with it and the caches those used record no sample count |
| Face detector failure | Wrong crops create false evidence | Quality gates, abstention, and detector benchmark | **partial**: gates and abstention in place; the benchmark's human review is unrun, so Phase 2's gate is open |
| Identity switch | The mouth and audio belong to different people | Track stability checks and identity-switch measurement | **partial**: stability checks in place; **no identity-switch measurement exists anywhere in the codebase** |
| Missing modality | Fusion treats absence as evidence | Explicit coverage gate and blank fusion result | **in place**: coverage reported beside every metric, absent streams masked to exactly zero including bias |
| Class imbalance | Accuracy hides failure on real media | PR-AUC, balanced metrics, and natural test frequencies | **partial**: PR-AUC and balanced accuracy reported; no natural-frequency evaluation exists |
| Validation overfit | Repeated choices consume the validation set | Predeclared comparisons and limited search budgets | **partial**: comparisons are predeclared in the experiment matrix; no search budget is enforced or counted |
| Calibration shift | A probability looks more certain than it is | Brier score, calibration error, and external reporting | **in place**, and it fired: ECE 0.0022 in-domain against 0.6366 on DFDC |
| Unsafe model loading | A joblib file executes local code | Load only artifacts created by this project | **convention only**: `joblib.load` is called in `cli.py` and `inference/loading.py` with no provenance check on the file |
| Demographic harm | Error rates differ across groups | Subgroup intervals, coverage reports, and restrained claims | **in place but uninformative**: F20 finds no gap at a ceiling where none could appear, and neither cross-corpus set carries demographic annotation |

Three rows have **no implementation at all** behind part of their control and are
the honest gaps: the identity-switch measurement, the distribution report, and
the natural-frequency evaluation. Two more are convention rather than control:
joblib provenance and the search budget.

## Trust boundaries

The code trusts local datasets, FFmpeg, pretrained model downloads, and project
checkpoints. Hashes detect accidental changes. They do not make an untrusted
artifact safe.

Raw media and derived face crops are sensitive biometric data. Keep them
outside Git. Follow dataset licenses and university retention rules.

## Safe result language

Report a probability as model evidence under the evaluated protocol. Do not
call it proof. Always show coverage, abstention, threshold, and known domain
shift. Preserve negative findings and failed subgroups in the final report.

## Review triggers

Review this threat model when:

- A new dataset or manipulation family enters the project.
- The system begins real-time or remote processing.
- A new modality changes the evidence contract.
- The dashboard exposes results to users outside the research team.
- Adversarial robustness becomes a project claim.
