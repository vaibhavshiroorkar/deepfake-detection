# Corpora for the generative-video problem

The project trains on face manipulations of real recordings and is close to
blind on fully generated video: 0 of 10 on `vasa_1` and 0 of 4 on `raskai` in
the [paper source book](paper-source.md) section 5.7. Closing that gap needs
data of the right class. This is the survey of what exists and what it costs,
so the choice is made once and recorded rather than re-searched every time.

Two different needs, and they want different corpora. **Person-centric
audio-visual fakes** are the class this pipeline is shaped for: a person
talking, with a soundtrack, where lip-sync and emotion streams have something to
read. **Fully generated video** is the class it has no mechanism for: no real
source, no seam, and the evidence lives in the whole frame and in the frequency
domain.

## Person-centric, closest to what this project already does

### DeepSpeak v2

Real people talking to their own webcams, which is the nearest public thing to
the phone video a deployed system would meet. 52 hours, 134 GB, and the fakes
come from six current engines across lip-sync, face-swap and avatar generation:
FaceFusion, Diff2Lip, HelloMeme, LatentSync, LivePortrait, Memo.

Free to qualifying academic institutions through a request form, then gated on
Hugging Face. `data/DeepSpeak_v2/` already exists here and is empty, so the
download was started and never finished.

<https://huggingface.co/datasets/faridlab/deepspeak_v2>

This is the first one to get. It is audio-visual, it is person-centric, it is
recent, and it is licensed for training rather than evaluation only.

### AV-Deepfake1M and AV-Deepfake1M++

The literal answer to "LLM-generated fakes": the manipulation is driven by an
LLM that rewrites the transcript, then TTS and lip-sync render the edit, so the
forgery is a small semantic change inside an otherwise genuine clip. 1M clips in
the original, 2M in the ++ release, with audio-visual perturbations added to
simulate real-world degradation. Research-only licence.

<https://github.com/ControlNet/AV-Deepfake1M>

Relevant beyond its size: the manipulations are localized in time, which is the
temporal-localization task this project's sync branch was designed for and has
never been evaluated on.

### Deepfake-Eval-2024

In-the-wild deepfakes actually circulated in 2024, collected from 88 websites in
52 languages through a detection platform used by journalists: 44 hours of
video, 56.5 hours of audio, 1,975 images.

<https://huggingface.co/datasets/nuriachandra/Deepfake-Eval-2024>

Use as evaluation only, for the same reason MNW is evaluation-only here. If it
is ever downloaded, add it to `EVALUATION_ONLY_DATASETS` in `data/guards.py`
before anything else touches it.

## Fully generated video

### CoCoVideo-26K

The closest match to the tools people actually use: 13 commercial generators
including Sora, Veo 3 and Kling, with semantically aligned real and fake pairs,
which is what lets a detector be tested on content rather than on style. Its
stated motive is that open-source generators are visibly worse than commercial
ones, so detectors trained on the former do not transfer to the latter.

<https://github.com/DonoToT/CoCoVideo>

### GenVidBench

6.78 million videos from 11 generators, built so that the training and test
generators differ, which is the cross-generator protocol this project's
research design already requires and has never run.

<https://github.com/genvidbench/GenVidBench>

### GenVideo, with the DeMamba baseline

Million-scale, and it ships two evaluation protocols worth copying: a
cross-generator task and a degraded-video task that scores robustness to the
quality loss of being shared and re-encoded.

<https://github.com/chenhaoxing/DeMamba>

## Two traps to avoid before downloading anything

**Watermarks are a shortcut, not a feature.** Commercial generators embed
provenance marks, and Sora output carries a visible one. A detector trained on
such clips can learn the watermark and report a number that collapses the moment
someone crops or re-encodes. There is now a de-watermarked Sora benchmark
specifically because of this
(<https://arxiv.org/abs/2512.10248>). Any generated-video corpus used here must
be checked for watermarks first, and that check belongs in the data card. This
is the same class of problem as the shortcuts the project's evaluation protocol
was built to control.

**Licences differ per corpus and some forbid training.** MNW is evaluation-only
and `data/guards.py` enforces it in code rather than in a document. Anything
downloaded under a research-only or evaluation-only licence goes in that set
before it is used, not after.

## What to do with them

Ordered by value for this project:

1. DeepSpeak v2 for training and for the practical question, since it is
   person-centric webcam footage with modern engines.
2. Generate a small set with the tools you have access to, a hundred or two per
   tool, as a held-out test set nobody has trained on. A corpus you generated
   yourself is the only one you can be certain no published detector has seen.
3. CoCoVideo for the commercial generative class.
4. AV-Deepfake1M++ if the temporal-localization task is taken up.
5. Deepfake-Eval-2024 as a final evaluation, never for training.
