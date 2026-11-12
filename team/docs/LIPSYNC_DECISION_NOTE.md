# Lip-sync and audio-video synchronisation: decision note (WS4)

Written 2026-10-26. Status: PROPOSED, nothing downloaded, installed or run for this note. Sources read today: the GitHub API and README of `joonson/syncnet_python`, the README and Hugging Face file list of `bytedance/LatentSync` and `ByteDance/LatentSync-1.6`. The SyncNet paper (Chung and Zisserman, ACCV workshop 2016) is not in docs/RESEARCH.md, so it is cited here as NR (not retrieved) and nothing below is attributed to its contents.

## What the candidates measure

| | SyncNet (`syncnet_python`) | LatentSync StableSyncNet |
|---|---|---|
| What it is | The reference audio-to-video synchronisation network with a demo pipeline (face detection and tracking, offset estimation) | A SyncNet-style network released to supervise a lip-sync generator |
| Output (from the README) | An audio-video offset in frames, a minimum distance and a confidence value (the demo prints `AV offset`, `Min dist`, `Confidence`) | A pretrained SyncNet "with 94% accuracy on both VoxCeleb2 and HDTF" per the LatentSync README; the accuracy definition and the inference interface were not reviewed |
| Detects | Whether audio and the visible mouth movement of one face track are temporally aligned, and by how much. Also which face is speaking, when several are present (README) | The same kind of audio-visual agreement; used inside LatentSync as a training loss |
| Does not detect | Manipulation as such. A synchronised fake passes; an unmanipulated dubbed, badly muxed or off-sync recording fails. Some lip-sync generators are trained with a sync network as a loss (LatentSync does, per its README), so their output may be well synchronised by construction | Same |

Consequence for the product: a low sync confidence or a large offset is reported only as `potential_mismatch` in the cross-modal section. It is never an authenticity finding and never combined into a fake score. A good sync is `consistent`, which says nothing about manipulation.

## Applicability rules the module needs

- No audio stream: `not_applicable` (reason: no audio).
- No face found or no usable mouth region: `not_applicable`.
- Several faces: report per-track results or `inconclusive` with the warning that speaker attribution is not established; do not pick the largest face silently.
- Audio without speech (music, silence): `inconclusive`.
- Very short clips (under about 1 s, estimate, to be fixed in the preregistration): `inconclusive`.
- Frame-rate and sample-rate conversions are needed (the demo pipeline works on its own re-encoded tracks); the exact requirements were not verified and must be read from the code before integration.

## Legal and access

| Item | Finding (2026-10-26) |
|---|---|
| `syncnet_python` code | MIT (GitHub API `spdx_id: MIT`; LICENSE.md read) |
| SyncNet weights `syncnet_v2.model` | Fetched by `download_model.sh` from `robots.ox.ac.uk/~vgg/software/lipsync/data/`. The README states no weight licence. The same file is re-hosted at 55 MB in the `ByteDance/LatentSync-1.6` repository. **Weight licence: not stated, so not cleared.** |
| Face detector used by the SyncNet pipeline | `sfd_face.pth` (S3FD), 90 MB, same server. Licence not stated. Project already has YuNet (see below), which may replace it, but that changes the pipeline and would need its own check |
| LatentSync code | Apache-2.0 (GitHub API) |
| LatentSync-1.6 model repository | Hugging Face card licence: `openrail++` (use restrictions apply; not yet read in full). Contains `stable_syncnet.pt` (about 1.6 GB), `syncnet_v2.model`, `latentsync_unet.pt` (about 5.1 GB, the generator, not needed) |
| AV-HuBERT based detectors (for example SpeechForensics) | Excluded for now: the AV-HuBERT licence was found non-commercial and the toolkit needs fairseq, which was judged incompatible with Python 3.13 in the feasibility document |

## Integration effort and hardware

- The reference pipeline is distributed with conda environment files (`environment.yml`); this project uses uv and Python 3.13. Dependencies were not reviewed. Expected work: port the model definition and weight loading to a plain PyTorch module (the model is small, 55 MB), reuse `dfd.prep` audio extraction and the YuNet face detector for mouth crops, and write the offset search. Estimate: 2 to 4 days including tests (estimate, not a measurement).
- Hardware: the SyncNet network is small; inference should fit easily beside the other models on the 12 GB card (to be measured, not asserted).
- LatentSync generation itself needs 8 GB (v1.5) or 18 GB (v1.6) VRAM per its README; we do not need the generator.

## Recommended next step

1. Approval needed: download of `syncnet_v2.model` (55 MB) from the official Oxford URL, and a decision on the unstated weight licence (ask the user to accept research use of weights from a published academic release, or contact the authors). A download does not authorise inference or evaluation.
2. Read the SyncNet paper in full and add it to RESEARCH.md before citing it.
3. Port the network (engineering, no data).
4. Preregister the evaluation before any run.

## Measurable evaluation design (to be preregistered as a new E-entry before running)

- Question 1 (does the measure respond to temporal misalignment?): take LAV-DF videos, which have word-level timestamps and independently edited audio or video (`modify_video`, `modify_audio`, `fake_periods`); compare the sync confidence inside labelled fake periods with unmodified periods of the same videos. Success criterion to be fixed beforehand, for example a within-video paired AUC lower bound above 0.6. Risk: LAV-DF's content-driven edits produce production artefacts (the I-6 visual cue was 0.9921 across videos but 0.5032 within videos, E-011 and E-012), so the within-video comparison is the one that counts.
- Question 2 (false-positive behaviour): the rate of `potential_mismatch` on unmanipulated real videos of other sources, including dubbed and low-frame-rate content. Report it as a rate with an interval; it bounds how often the module will warn on innocent media.
- Question 3 (FakeAVCeleb): sync response on Wav2Lip fakes against real video with cloned audio. Only after the FakeAVCeleb decision of docs/FAKEAVCELEB_SPLIT_PLAN.md is executed and only on its development partition.
- All splits are by speaker identity; thresholds are chosen on development data only and frozen before any sealed read.
