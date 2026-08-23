# Data card

## Intended use

This project studies talking-head deepfake detection. It is a research system, not an identity, moderation, or law-enforcement system.

The primary development dataset is [FakeAVCeleb](https://arxiv.org/abs/2108.05080). Deepfake-Eval is the external evaluation target. Consult each dataset's current access and license terms before downloading or sharing files.

## Record contract

Each clip record contains:

- Dataset and clip identity.
- Media path.
- Source identity and optional target identities.
- Manipulation method and manipulation type.
- Global clip label.
- Independent video and audio labels.
- Race and gender metadata when supplied by the dataset.
- Leading-silence duration when known.

The branch labels have different meanings. Visual models use `video_fake`. Audio models use `audio_fake`. Fusion uses the global clip label. The synchronization branch learns alignment from authentic media and generated correspondence tasks.

## Split policy

The primary protocol separates source identities. It preserves every evaluation row. The training loader may rebalance cue labels through sampling.

Target identities still cross the primary split. FakeAVCeleb's identity graph prevents a useful full split that isolates both source and target roles. The project generates an identity-strict filtered subset and reports its reduced coverage.

The existing prototype test split is not a final blind test. The new split, seed, and hash must be frozen before final model selection.

## Quality and exclusions

The cache records face coverage, face-track stability, audio presence, clipping, duration mismatch, and preprocessing identity. A per-clip fingerprint includes media content and timing metadata. A separate global hash identifies the code and view configuration.

The system abstains when it lacks a stable primary face, enough face coverage, audio, or aligned audio-video duration. It does not replace failed face detection with the full frame.

Multi-person videos without a stable primary face are outside the core protocol. Report their abstention rate rather than deleting them from the denominator.

## Known risks

- Dataset-specific compression, silence, identity, and generation artifacts can become shortcuts.
- Demographic labels may be incomplete or coarse.
- FakeAVCeleb contains far more fake clips than real clips.
- Some manipulation methods have limited identity-strict coverage.
- A high in-domain AUC does not prove real-world reliability.
- Detector errors can cause harm when users treat scores as facts.

## Handling rules

- Keep raw videos outside Git.
- Do not publish derived face crops without checking dataset terms.
- Do not infer identity or protected traits beyond supplied audit metadata.
- Do not tune models on external test labels.
- Keep failed preprocessing rows in coverage reports.
- Remove local media and model artifacts according to university retention rules.
