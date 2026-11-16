# P-002 design: rebuilding for maximum performance

A proposal for the study after the final-year submission, written on 2026-11-16 from everything P-001 showed. It is not preregistered yet. P-002's preregistration will fix its tests and bars before any run, and the open items it needs are in [PENDING.md](../PENDING.md), section 6.

It assumes we start over from what the project has taught us, and it says plainly what to keep and what to replace.

## What the evidence says limits performance today

In order of how much each one costs:

1. **Training data is the biggest limit.**
   - Every model learns from one small, old corpus: FakeAVCeleb, from 2021, with about 500 real videos and crude fakes.
   - The sync path saw about 350 real speakers.
   - The models never see a modern generator, so off-corpus AUC falls to 0.55 to 0.66.
2. **Clip-level scoring hides short edits.**
   - We score a fixed 3 s window and average it.
   - LAV-DF edits one word; 46% of its fake clips had no edit in the window (A9).
   - Averaging over windows also dilutes a short local edit.
3. **Weak per-path backbones.**
   - EfficientNet-B0 for faces and Wav2Vec2 Base for audio are small.
   - The literature shows large frozen foundation features generalize across generators much better than small fine-tuned CNNs: CLIP ViT-L in Ojha et al., CVPR 2023, and SSL front ends in speech anti-spoofing (XLS-R or WavLM with AASIST).
4. **Lip-sync alone cannot see joint generators.** Sora 2 and Veo 3.1 generate sound and picture together, so the lips match: the sync AUC is about 0.40 to 0.50. A mismatch detector needs partners that look at other evidence.
5. **Preprocessing faults were found late.**
   - Face finding missed small faces (A10).
   - Some clips had less video than audio (mistake 67).
   - The window missed LAV-DF edits (A9).
   - Each was found after a result had been read.
6. **Compute.** Preprocessing is bound to the CPU (6 cores) and stores pixels. It caps how much data we can use.

## Keep (it is good, and rare)

- The protocol and its machinery:
  - preregistration with dated amendments;
  - identity- and corpus-disjoint splits, with locked targets;
  - receipts;
  - `numbers.json`, which feeds every deliverable;
  - the shortcut audit and the neutralizer;
  - real-only conformal and umbrella calibration (H4).

  This is the project's real asset. Any rebuild sits on it.
- The real-only sync idea, as one path among several, not the main detector.

## Replace or add

1. **Data: multi-source, large, and modern.**
   - **Train on several corpora at once:**
     - AV-Deepfake1M and 1M++, with more than a million clips, many generators and segment labels;
     - LAV-DF, which has segment labels;
     - FakeAVCeleb;
     - the full DFDC;
     - KoDF and DeepSpeak v2, if granted.
   - **Real-only data for the one-class parts:** VoxCeleb2 and LRS3, checking speaker overlap with AV-HuBERT's pretraining.
   - **Main protocol: hold out by corpus and by generator,** never by random clip. Leave one corpus out, and leave one generator out.
   - **One external, unseen set of real-world clips:**
     - an in-the-wild benchmark such as Deepfake-Eval-2024 or MNW;
     - phone clips recorded with consent, to measure the false-alarm rate on real video.
2. **Ingestion with data-contract tests, before any model runs.** Each check runs on every corpus and fails the build:
   - face found in at least 90% of windows;
   - video at least as long as the audio;
   - the labelled segment inside the window that is scored;
   - shortcut ceiling at most 0.60 after neutralization.

   These turn mistakes 67 to 69 and A9 and A10 into automatic gates.
3. **Make the edges of the data unreliable as cues, not just uniform.** Keep the neutralizer, and during training also randomize codec, bitrate (the leftover H1 cue), resolution, loudness and the window offset. The model then cannot lean on any of them, and it becomes robust to platform recompression.
4. **Temporal localization instead of one score per clip.**
   - Score many short windows across the whole clip, with segment labels where the corpus has them.
   - Train a temporal head on top.
   - A clip's score is its strongest window (max or top-k), not the mean, so a one-word edit counts.
5. **Stronger paths, mostly frozen foundation features.**
   - **Visual:** CLIP ViT-L/14 with LayerNorm-only tuning, then a light temporal head. Chosen in [P-002 backbones](P-002-backbones.md).
   - **Audio:** a frozen XLS-R front end with a weighted sum of layers and an AASIST-style head. Chosen in the same record.
   - **Lip-sync:** AV-HuBERT Large, real-only, at the window level.
   - **A fourth path for joint generators:** an audio-visual model fine-tuned on segment labels, in the AVFF style, so the system has a path that does not depend on mismatch.
6. **Fusion and attribution.**
   - Late fusion trained out-of-fold on held-out corpora, with masks for a missing modality.
   - Per-modality heads that say which stream was faked (H3).
   - Train the fusion on corpus-held-out predictions, so it learns which path to trust when the domain shifts.
7. **Calibration for deployment.** Keep the real-only conformal threshold, and add a Mondrian version (one threshold per condition, such as resolution or orientation) for platforms whose real clips mix conditions.
8. **Compute redesign.**
   - Decode and find faces on the GPU (NVDEC, plus a GPU face detector).
   - Cache embeddings, not pixels.
   - Shard by corpus.
   - Train the large-data stages on a rented GPU node if local time is too long; budget it before starting.
   - AV-Deepfake1M++ is 1.46 TB and this PC has about 519 GB free: plan a stratified subset or an external disk before requesting the download.
9. **Baselines on the same protocol:** AVH-Align, HAVIC, AVFF if code is available, and a CLIP linear probe. A claim of better performance is valid only against these, on the same splits.

## Order and size

1. **Preregister P-002:**
   - the corpus-held-out protocol;
   - the contract tests;
   - the paths;
   - the success bars, set against baselines on the same splits.
2. **Data access and ingestion with the contract tests, and the shortcut audit on every corpus.** About 1 to 2 weeks, depending on access approvals.
3. **Baselines on the new protocol.** About 1 week.
4. **New paths:**
   - frozen feature caches;
   - temporal heads;
   - the joint audio-visual path.

   About 2 to 3 weeks of GPU time.
5. **Fusion, attribution and calibration, then the full evaluation.** About 1 week.
6. **Paper.**

Roughly 6 to 8 weeks, with access to AV-Deepfake1M as the main dependency.

## What to expect, without bias

- **Likely:** clear gains on corpora and generators that resemble the training mix. Better localization of short edits. Lower false alarms on real video under calibration.
- **Uncertain:** generalization to the newest joint generators, such as Sora 2 and Veo 3.1. Nobody has shown this to be solved. Frozen foundation features and a joint audio-visual path are the best current bets, not a guarantee.
- **Not promised:** "better than every existing detector". The defensible claim is a measured gain over reproduced baselines on a shortcut-free, corpus-held-out protocol, with every result reported. That is what a strong venue paper rests on.
