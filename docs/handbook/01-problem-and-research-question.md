# Problem and research question

## Learning goals

After this chapter, you should be able to classify the attack types, state RQ1
through RQ5, and explain what a positive or negative fusion result means.

## Required background

You need only the idea that a classifier maps an input to a score. Read the
[research design](../research-design.md) and [data card](../data-card.md) beside
this chapter.

## Problem definition

Real talking-head media has authentic video and authentic audio. A face swap
replaces a person's face while the speech can remain real. Reenactment changes
facial motion or expression. Synthetic speech replaces or generates the audio.
A combined attack changes both video and audio.

The manifest represents these cases as four cue combinations:

| Manipulation type | Video cue | Audio cue | Clip label |
|---|---:|---:|---:|
| `RealVideo-RealAudio` | real | real | real |
| `FakeVideo-RealAudio` | fake | real | fake |
| `RealVideo-FakeAudio` | real | fake | fake |
| `FakeVideo-FakeAudio` | fake | fake | fake |

The platform asks whether visual artifacts, audio spoofing, and mouth-audio
alignment provide useful evidence across new sources and attacks. It does not
ask whether one high score can prove that a person created or shared a fake.

Let `Y` be the real or fake clip label and `S(X)` be a detector score for media
`X`. The implemented classifiers estimate evidence related to `P(Y = 1 | X)`
under their training data. This conditional score is not a causal claim and is
not guaranteed to transfer to a new domain.

## Research questions

- RQ1, fusion generalization: Does calibrated cue-specific fusion generalize
  better than a strong visual baseline under source-disjoint and
  shortcut-controlled evaluation?
- RQ2, branch contribution: Which visual, audio, and synchronization cues
  remain useful on unseen manipulation families and an external dataset?
- RQ3, view integrity: How do face detection, tracking, and landmark alignment
  affect coverage, synchronization quality, and downstream performance?
- RQ4, reliability: How do calibration and quality-aware abstention affect
  confidence and coverage under missing or degraded evidence?
- RQ5, cost: Which component choices provide the best validation evidence
  within the local compute budget?

For RQ1, let `D` be the paired performance difference between fusion and the
visual baseline on the frozen evaluation units:

```text
D = metric(fusion) - metric(visual)
H0: D <= 0
H1: D > 0
```

The null hypothesis says fusion does not improve the chosen metric. The
alternative says it does. A paired source bootstrap is planned to estimate the
uncertainty of `D`. The locked test must not be used to select the metric,
threshold, branches, or direction of the hypothesis.

## Contribution

The contribution is controlled evidence, not a new fusion architecture.
Audio-video fusion, speech encoders, visual backbones, and synchronization
learning already exist. This project combines known components with
cue-specific labels, source-disjoint splits, method holdout, calibration,
coverage reporting, and paired evaluation.

Benchmark accuracy can look high for the wrong reason. The same identity may
occur in training and test. A codec may identify a generation pipeline.
Leading silence may identify synthetic audio. A manipulation method may occur
in every partition. In each case, the model can recognize a shortcut instead
of the intended cue.

### Worked example

Suppose a test set has ten clips from Person A in training and ten more clips
from Person A in test. A model can learn Person A's face or recording setup.
Nine correct test predictions then give 90 percent accuracy, but the result
does not answer how the model handles an unseen source. Grouping all clips from
Person A into one partition removes this route.

### Project code path

The claim starts in [research-design.md](../research-design.md). Labels enter
through [`ClipRecord`](../../src/deepfake_detection/data/manifest.py). Source
allocation and method holdout live in
[`protocols.py`](../../src/deepfake_detection/data/protocols.py). Later stages
export cue scores, fit late fusion, and evaluate locked predictions.

### Design trade-offs

Source-disjoint evaluation protects the main identity role while retaining
useful data. It does not isolate every target identity. The smaller
identity-strict subset tests that harder condition but can lose method
coverage. Late fusion is easier to audit than a large end-to-end fusion model,
but it may miss interactions between raw modalities.

### Failure cases

- A detector can exploit identity, codec, silence, background, or resolution.
- Missing faces or audio reduce coverage and can make the scored subset easier.
- A new generator can create artifacts absent from the development data.
- Coarse demographic labels can hide subgroup differences.
- Repeated tuning on the test set turns it into training data.

### Supporting tests

[`test_protocols.py`](../../tests/test_protocols.py) checks source separation,
identity-strict filtering, stable split hashes, and method holdout. Run:

```powershell
uv run pytest tests\test_protocols.py -v
```

## Limits

The intended use is research on talking-head deepfake detection. Non-goals
include identity recognition, authorship attribution, automated moderation,
law-enforcement decisions, and a guarantee of real-world truth.

A negative fusion finding is valid evidence. It means the planned comparison
did not show a reliable gain under the stated data, metric, and uncertainty.
It does not prove that multimodal fusion can never help. The result should be
reported with its confidence interval, coverage, and protocol limits.

### Exercises

1. Classify one example of each manipulation type and name the correct branch
   label.
2. Rewrite RQ1 as variables, controls, and a measurable outcome.
3. List two shortcuts that source-disjoint splitting does not remove.
4. Explain why deleting abstained clips can inflate reported performance.

## Viva questions

1. Why is benchmark accuracy alone weak evidence of generalization?
2. Why does the project use cue-specific branches?
3. What result would support `H1`, and what uncertainty is needed?
4. Why can a negative result still be a contribution?
5. What claim is outside the intended use?

## Sources

- [FakeAVCeleb dataset paper](https://arxiv.org/abs/2108.05080)
- [FaceForensics++ paper and benchmark](https://openaccess.thecvf.com/content_ICCV_2019/html/Rossler_FaceForensics_Learning_to_Detect_Manipulated_Facial_Images_ICCV_2019_paper.html)
- [Wav2Lip paper](https://arxiv.org/abs/2008.10010)
- [AVFF audio-visual fusion paper](https://openaccess.thecvf.com/content/CVPR2024/html/Oorloff_AVFF_Audio-Visual_Feature_Fusion_for_Video_Deepfake_Detection_CVPR_2024_paper.html)
- [Deepfake-Eval dataset paper](https://arxiv.org/abs/2503.02857)
