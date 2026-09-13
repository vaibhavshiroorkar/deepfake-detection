# Data card

## Intended use

This project studies talking-head deepfake detection. It is a research system, not an identity, moderation, or law-enforcement system.

The primary development dataset is
[FakeAVCeleb](https://arxiv.org/abs/2108.05080), used for training, validation
and the held-out test partition.

Celeb-DF-v2 is the cross-dataset generalization target. Its role is now frozen:
no Celeb-DF clip enters training or model selection, and the cross-dataset
numbers use the official 518-clip `List_of_testing_videos.txt` protocol so they
are comparable with published work. Note that its list writes `1` for real,
which is the opposite of this project's label; `data/celebdf.py` inverts it once
so no call site has to remember.

FaceForensics++ c23 is present, and its provenance is weaker than the others.
The corpus is officially distributed under a EULA with TUM, where you complete
their form and they send a download script. This copy came from a third-party
Hugging Face mirror that states no licence, pulled by `scripts/fetch_ffpp.py`.
The data is research-available either way, but "third-party mirror" is a weaker
claim than "signed EULA" and the difference belongs on the record rather than in
someone's memory. Prefer the official route when there is time for it.

What is actually in the tree: 7,000 clips, 1,000 originals and 1,000 from each
of six manipulation families, of which 6,992 cached successfully. The corpus
carries no audio track at all, which `ffprobe` confirms and the cache audit
recorded as `missing_audio` on every clip, so it can train the visual stream and
nothing else. Its role here is the cross-dataset training corpus: a model
trained on FF++ and scored zero-shot on Celeb-DF-v2 and DFDC is the one result
in this project directly comparable with published tables.

The [Microsoft-Northwestern-WITNESS benchmark](https://github.com/microsoft/MNW)
is the locked external evaluation target. MNW is evaluation-only. It cannot be
used for training, validation, model selection, or threshold selection. MNW
also prohibits commercial use. This is no longer only a convention: the check
in `src/deepfake_detection/data/guards.py` refuses an MNW manifest in split
building, branch training, fusion training, and any feature export whose
partition role is not `external`. Consult every dataset's current terms before
downloading, deriving, or sharing files.

### What MNW actually contains

Verified against the checkout, because the benchmark's headline figure of
50,000 artifacts covers images and audio as well, and the video half is far
smaller. Pinned at commit `df66c459dd8b043cc7a8aeab30de8f8126710c7f`, fetched by
`scripts/fetch_mnw.ps1`.

| Part | Clips | Classes | Audio |
|---|---:|---|---|
| `Deepfake_Video/` | 120 (12 generators x 10) | fake only | none |
| `AI_media_in_the_wild/Video/` | 14 | 8 manipulated, 3 authentic, 3 inconsistent | present |

Four consequences follow, and every one of them constrains how MNW may be read:

- **No audio at all in the lab half.** Every filename contains `_no_audio_`, and
  `ffprobe` confirms a video stream only. The audio and sync branches cannot be
  scored on it. The cache marks each clip `missing_audio`.
- **No real videos in the lab half.** With no negatives, ROC-AUC and every other
  ranking metric is undefined. The reported statistic is the per-generator
  detection rate at the frozen threshold. A paired AUC set may be constructed by
  combining MNW forgeries with held-out real video, but it must be labelled a
  constructed protocol, never an MNW-native benchmark result.
- **Nine of the twelve generators are absent from FakeAVCeleb** (Diff2lip,
  SadTalker, VASA-1, MuseTalk, Video-retalking, HeyGen, EchoMimic, RaskAI, and
  the two Wav2lip enhancement variants). That is what makes this an unseen-
  generator test rather than a second in-domain one.
- **n is small.** 120 lab clips and 11 usable in-the-wild clips support wide
  confidence intervals and no subgroup analysis. The in-the-wild half is a case
  study, not a benchmark. `inconsistent` clips are excluded because that verdict
  means the experts could not decide, which is not a ground truth.

A schema caveat worth stating plainly: `ClipRecord.manipulation_type` is a
closed four-value enum with no way to record "audio absent". MNW forgeries are
therefore stored as `FakeVideo-RealAudio`, which literally asserts genuine audio
where there is none. The truth is carried by `QualityReport.audio_present`,
which is False for every MNW clip. Do not read the manipulation type of an MNW
row as a claim about its audio.

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
- Keep MNW out of every training and validation manifest.
- Do not publish derived face crops without checking dataset terms.
- Do not infer identity or protected traits beyond supplied audit metadata.
- Do not tune models on external test labels.
- Keep failed preprocessing rows in coverage reports.
- Remove local media and model artifacts according to university retention rules.
