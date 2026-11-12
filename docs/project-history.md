# Project history

This file tells the story of the project from the first commit on 2026-07-04 to
now. It says what we planned, what we measured, what turned out to be wrong, and
what we changed because of it. Every number here is already recorded in a
finding, an experiment entry or a commit, and each one is cited.

The detail lives elsewhere. Read this first, then follow the links.

## The question

Can a detector that combines several cues (face artifacts, voice spoofing and
lip-audio agreement) catch deepfakes it was not trained on better than a face
detector alone?

The short answer so far: the combined system is very accurate on the data it
trained on, and that accuracy mostly comes from shortcuts. These include the
video codec, the recording conditions, and whether a clip has an audio track at
all. The cue each part was built to read is often not the cue it uses. Most of
the work below is about finding that out and proving it.

## Timeline

| Dates | Phase | Who |
|---|---|---|
| 2026-07-04 to 07-23 | First plan and prototype | Vaibhav, Adithya |
| 2026-08-01 to 08-05 | First trained visual streams | Adithya, Vaibhav |
| 2026-08-24 to 09-04 | Rebuild as a research platform | Vaibhav |
| 2026-09-09 to 09-15 | Full pipeline, first real results | Vaibhav |
| 2026-10-13 to 10-22 | Seeds, challengers and the paper draft | Vaibhav |
| 2026-10-24 to 10-29 | Team project with a sealed test | Shivansh, Vaibhav, Adithya |
| 2026-10-27 to 10-29 | Audit of this project, claims withdrawn | Vaibhav |
| 2026-11 | Both projects in one repository | Vaibhav |

## 1. The first plan (July)

The first design had five streams: three visual models (Xception, EfficientNet
and DINOv2) and two cross-modal streams (lip-sync and emotion). The idea was
that a lip-sync fake changes only the mouth, so the frames look real. The fake
lives in the mismatch between sound and face. The primary dataset was
FakeAVCeleb, with identity-disjoint splits from day one.

On 2026-07-09 the design changed in three ways (commit `d06fd6b`):

- **Feature-level fusion instead of score averaging.** The streams pass
  embeddings, not scores, to one joint model.
- **Embedding cross-attention for lip-sync.** We dropped the Bohacek and Farid
  method, which lip-reads, transcribes and compares words. Our stream compares
  audio and mouth embeddings directly, with no text in between.
- **No Swin.** DINOv2 became the third visual stream, because a self-supervised
  backbone was expected to transfer better than two artifact CNNs.

Adithya built the Stage 1 data pipeline and the first EfficientNet stream
(`d30cc0a`). On 2026-07-23 the code was snapshotted and taken apart so it could
be rebuilt one step at a time (`926624a`). That snapshot records five places
where the plan and the code disagreed. The full plan of that time is still in
history: `git show 926624a^:PROJECT_OVERVIEW.md`.

In early August Adithya added the trainer, resume support, and trained Xception
and DINOv2 streams (`bf09253` to `81c04db`). DINOv2 was then swapped for DINOv3
(`60b0f0f`).

## 2. Rebuild as a research platform (late August)

The prototype could train models but could not prove anything. So on
2026-08-24 the project restarted from a new root commit with evaluation as the
first concern:

- source-disjoint and identity-strict splits, checked in code;
- a preprocessing hash and a split hash stored with every checkpoint;
- local MLflow tracking, versioned configs and a CPU smoke run in CI;
- a handbook that teaches each part from first principles.

The first baseline, EfficientNet-B0 plus a GRU, scored 0.9992 ROC-AUC on
FakeAVCeleb validation. Four more runs at the same settings showed that the seed
alone moves validation loss by 2.8 times ([status](handoff.md), "Run
comparison"). That was the first sign that one run is one draw, not a result.

A planned human-reviewed face-detector benchmark was dropped. Nobody was going
to label the 625 frames it needed, so the decision is recorded as a decision,
not a measurement ([roadmap](../ROADMAP.md), Phase 2).

## 3. Full pipeline and the first surprises (September)

All three branches (visual, audio, sync) and the fusion head were trained at
full scale. Three results changed how the project worked:

- **The selection rule was the result.** Two streams picked their checkpoint by
  validation loss. One kept an epoch-1 checkpoint that scored 0.4668, below
  chance. Both results were withdrawn and rerun with ROC-AUC selection (F3,
  `b63cf87`).
- **In-domain numbers were shortcuts.** Three times a near-perfect in-domain
  score collapsed elsewhere: 0.9990 against 0.4920 cross-corpus, a 0.9969
  matched-pair stream reading 0.5059 under new capture conditions, and 0.9997 on
  Veo 3 that was entirely a content confound. After that, leave-one-generator-out
  became the only headline metric (`1395e2d`).
- **False alarms on real video range from 0 to 91 percent**, depending on
  capture conditions (F16). A density model over genuine footage cut false
  accusations by about two thirds (F17).

Other results from this phase: training on FaceForensics++ did not improve
transfer (F10). Xception was removed on an argued redundancy with EfficientNet,
and the commit says it was argued, not measured (`3284514`). DF26's audio was
audited before use and found unusable for audio-visual work (F14).

## 4. Seeds, challengers and the central finding (October)

Every branch and the fusion head were retrained at seeds 17, 29 and 43.

- **Seed beats every effect.** Cross-corpus on DFDC, the fused score moves 0.1078
  with the seed. The 0.7500 headline was the best of three; the mean is 0.6873
  (F27). At three seeds, fusion never beats the visual branch on DFDC (F19).
- **The named mechanism is absent.** The cross-attention streams logged diagonal
  attention mass every epoch, never as part of the loss. It sat at chance in
  every run while the emotion stream reached 0.9991 ROC-AUC (F1). The model is
  accurate, but its attention does not line up sound with face.
- **Two of three branches do not transfer.** The sync branch is at or below
  chance at all three seeds (F26). The audio branch reads 0.5054 on DFDC against
  0.7739 in-domain (F30).
- **An unseen manipulation family costs 0.12.** On FaceForensics++, macro
  ROC-AUC falls from 0.8743 to 0.7511, and FaceSwap alone falls to chance (F34).
- **The visual model reads the codec.** The served models respond to the video
  encoding, not only the face (F35). Training that shortcut away cost more on
  DFDC than it gained (F37).

A second track tested a frozen encoder with a linear probe on generated video
(DF26). In DF26, whether a clip has an audio track nearly predicts its label.
A user web app, an in-browser demo and a LaTeX paper draft followed.

## 5. The team project (late October)

From 2026-10-24 Shivansh Gupta, Vaibhav Shiroorkar and Adithya Jithesh built a
second system for the final-year submission, supervised by Dr. Tejashree Phatak.
It lives in [`team/`](../team/README.md). Shivansh wrote most of its commits; it
ran on Vaibhav's GPU and reused datasets from this project.

It was built to a stricter protocol than this project had used:

- each hypothesis and its pass line written down before the run;
- a test split sealed and read once, after the system was frozen, with a ledger
  of every read;
- three confirmatory tests declared in advance, with a Bonferroni correction.

Its main results ([team experiments](../team/docs/EXPERIMENTS.md), E-014):

- video, Celeb-DF family: 0.6238 at the MVP to 0.8218 for the final system;
- speech, In-the-Wild: 0.7516 to 0.9148;
- the advantage over an Xception baseline holds at the 95 percent level but not
  at the corrected 98.33 percent level, and the paper says so.

It found the same two problems this project found, with different code, data
splits and encoders:

| Problem | This project | Team project |
|---|---|---|
| The audio-visual signal is not audio-visual | attention at chance while accuracy is high (F1) | visual head at chance within produced LAV-DF videos, 0.5032 (E-014) |
| File format beats content | visual models read the codec (F35) | codec metadata alone separates real from fake FakeAVCeleb video at 0.9985 (E-021) |

Two independent pipelines reaching the same result is a replication, not one
pipeline's bug. That agreement is the strongest result either project has.

The team submitted on 2026-10-29 (tag `team/submission-final`).

## 6. Audit and withdrawn claims (late October)

Reading this project against the team protocol exposed three faults
(`5d4285d`, [status](handoff.md)):

- **Fold leakage.** Fusion fold checkpoints chose their epoch using labels from
  the holdout they were later scored on.
- **Shifted LAV-DF windows.** A default shifted windows that started at zero, so
  some "genuine" windows overlapped manipulated spans. The 0.9969 lip-sync score
  must be rerun before it is read.
- **Wrong input normalization** for the Perception Encoder and CLIP features
  (F33). The old 0.7527 open-world score does not stand.

The code was fixed and every affected row is marked `requires rerun` in the
[result registry](research/result-traceability.md). Three of the four stated
targets now wait on reruns. The withdrawn numbers stay in the record with the
reason.

## What we changed because of mistakes

Each fault became a rule, so it cannot recur silently:

| Mistake | Rule now |
|---|---|
| Checkpoint chosen on validation loss kept a below-chance model | Select on the metric that is reported |
| One seed reported as the result | Three seeds, mean and spread, best-of-three named as such |
| In-domain near-perfect scores read as success | Leave-one-generator-out and cross-corpus are the headline |
| Fold checkpoints saw holdout labels | Code rejects any OOF export that saw its own labels |
| Test data reachable during tuning | Sealed split, read once, ledger entry per read (team protocol) |
| A number in a doc with no file behind it | Every number resolves to a registry row and a file hash |

## Still open

- Rerun the cross-fit seeds, rebuild the LAV-DF cache, and re-extract the PE and
  CLIP features. Commands are in [status](handoff.md).
- The MNW visual scores sit near 0 ROC-AUC on 85 clips. A ranking that is almost
  perfectly inverted usually means a flipped label. Quick look, not a
  measurement: check label polarity before reading it as a result.
- Test the audio-visual claim causally: shift or swap the audio at test time and
  see whether the score moves. Attention at chance shows the model is not
  aligned; it does not yet show the model ignores timing.
- The team's preregistered E-035 (zero-shot AV-HuBERT on LAV-DF windows) and
  E-036 (trained heads) are the next audio-visual runs.

## Where to read more

- [Findings](research/findings.md): every measured claim, F1 to F37.
- [Contributions](research/contributions.md): what is new and what is prior work.
- [Status](handoff.md): what is measured and what runs next.
- [Roadmap](../ROADMAP.md): phases and exit gates.
- [Changelog](../CHANGELOG.md): software and protocol changes.
- [Team decisions](../team/docs/DECISIONS.md) and
  [team experiments](../team/docs/EXPERIMENTS.md): the team project's record.
