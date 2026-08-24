# Audio-video foundations

## Learning goals

After this chapter, you should be able to convert time to frame and sample
positions, explain presentation timestamps, and identify media shortcuts that
can fool a detector.

## Required background

You need arithmetic with seconds and rates. Chapter 02 introduces the tensor
shapes used below.

## Digital video

A digital video stream contains decoded pictures called frames. Frame rate is
measured in frames per second (FPS). At a constant 25 FPS, the nominal period
between frames is:

```text
frame_period = 1 / 25 = 0.04 seconds
```

Containers can store variable frame rate media, so frame number alone is not a
safe clock. A decoder maps compressed packets to frames. Each decoded frame has
a presentation timestamp that says when it belongs on the playback timeline.

The visual view samples 16 visual frames per clip. It does not assume the
source contains exactly 16 frames. The current normalized visual tensor is
`[B, 16, 3, 224, 224]`.

## Digital audio

Audio sampling rate is the number of waveform measurements per second. The
current rate is 16,000 samples per second, or 16 kHz. For duration `t` seconds:

```text
sample_count = round(t * sample_rate)
sample_index = round(timestamp * sample_rate)
```

The four-second audio view contains 64,000 samples because
`4 * 16000 = 64000`. Its tensor is
`[B, 64000]`. The two-second sync waveform contains 32,000 samples.

Resampling converts source audio to the required rate. It needs an anti-alias
filter when lowering the rate. Treating 48 kHz values as 16 kHz without
resampling changes pitch and duration.

Leading silence is the quiet interval before speech. It can reflect editing or
generation pipelines rather than speech authenticity. The current view config
removes known leading silence by default. Evaluation must compare this choice
because silence can become a shortcut.

## Timestamps and synchronization

Audio and video must share one time origin. There are 50 mouth frames in a
two-second sync view at 25 FPS because `2 * 25 = 50`. The code samples the center
of each frame period. A view starting at one second begins at 1.02 seconds and
ends at 2.98 seconds.

The maximum training offset is 0.32 seconds, equal to 5,120 samples at 16 kHz:

```text
offset_samples = round(0.32 * 16000) = 5120
```

Positive offset means one modality is shifted relative to the other under the
objective's stated convention. A wider audio context allows a shifted slice
without padding. Padding would reveal the offset class through artificial
zeros.

For evaluation, `eval_overlap = 0.5`. A four-second window therefore advances
by `4 * (1 - 0.5) = 2` seconds. `sliding_window_starts()` also appends the last
complete interval when the stride does not land on it.

## Codecs and shortcuts

A container such as MP4 holds streams and timing metadata. A codec compresses
and reconstructs a stream. H.264, AAC, and other codecs can leave quantization,
blocking, ringing, bandwidth, or delay patterns.

If real and fake clips use different encoders, a model may identify the codec
instead of manipulation evidence. The same risk applies to frame rate,
resolution, aspect ratio, leading silence, loudness, and file duration.
Re-encoding can reduce one shortcut but also create a new common artifact.

Presentation timestamps matter because decode order can differ from display
order. Seeking by an assumed frame index can return the wrong moment. Separate
audio and video decoders can also round boundaries differently. One shared
timeline makes every view refer to the same seconds before converting to frame
timestamps or sample positions.

## Worked timeline

Consider a ten-second clip with no removable leading silence.

- Visual view: sample 16 frame-center timestamps uniformly over the usable
  interval. Its output shape is `[1, 16, 3, 224, 224]`.
- Audio view: take four seconds. At 16 kHz its output shape is `[1, 64000]`.
- Sync view: choose start `s = 3.0` seconds. Its 50 timestamps run from 3.02
  through 4.98 seconds. The audio starts at sample `3.0 * 16000 = 48000` and
  contains 32,000 samples.
- Shift context: add 0.32 seconds on both sides. The full context lasts
  `2 + 2 * 0.32 = 2.64` seconds and contains 42,240 samples.
- Maximum shift: move the two-second slice by 5,120 samples while keeping its
  length unchanged.

### Project code path

[`views/timeline.py`](../../src/deepfake_detection/views/timeline.py) defines
`ViewConfig`, uniform timestamps, sliding windows, and sync windows.
[`views/media.py`](../../src/deepfake_detection/views/media.py) reads media
metadata, seeks frames, and decodes mono audio. [`views/preprocessor.py`](../../src/deepfake_detection/views/preprocessor.py)
uses one config to build the visual, audio, sync, and wider context views.

### Design trade-offs

- Uniform 16-frame sampling limits compute, but can miss short artifacts.
- A four-second audio view gives context, but may include unrelated speech.
- A 25 FPS sync grid is simple, but source frames may have other rates.
- Overlap improves temporal coverage, but repeats evidence and costs compute.
- Removing leading silence limits a shortcut, but can remove real context.

### Failure cases

- Bad or missing timestamps can misalign modalities.
- Variable frame rate media breaks frame-index timing assumptions.
- Incorrect resampling changes waveform time and frequency content.
- Codec delay or decoder rounding can shift audio boundaries.
- Short clips need bounded timestamps and padding-free context rules.
- Synthetic silence can reveal a label without any speech analysis.

### Supporting tests

[`test_views.py`](../../tests/test_views.py) checks window counts and exact sync
alignment. [`test_preprocessor.py`](../../tests/test_preprocessor.py) checks
bounded timestamps and prepared shapes. [`test_media_decoder.py`](../../tests/test_media_decoder.py)
uses generated media to test the decoder when FFmpeg is available. Run:

```powershell
uv run pytest tests\test_views.py tests\test_preprocessor.py `
  tests\test_media_decoder.py -v
```

## Exercises

1. Calculate samples in 3.5 seconds at 16 kHz.
2. Calculate frame centers for a 0.2-second window at 10 FPS.
3. List the four-second window starts for a nine-second clip at 50 percent
   overlap.
4. Explain how zero padding could reveal whether audio was shifted left.
5. Change a fixture's frame rate and verify that timestamp requests remain in
   seconds.

## Viva questions

1. Why is a presentation timestamp safer than a frame number?
   Expected answer: timestamps state display time even when frame rate varies
   or decode order differs from presentation order.
2. Why does resampling need filtering?
   Expected answer: lowering the sample rate without an anti-alias filter folds
   frequencies above the new limit into false lower frequencies.
3. Why does the sync objective decode wider audio context?
   Expected answer: it can crop every shifted two-second view from real decoded
   audio. Offset-specific zero padding would reveal the target class.
4. How can a codec become a shortcut?
   Expected answer: if codec artifacts correlate with labels or methods, the
   model can identify the encoding pipeline instead of manipulation evidence.
5. Why must every branch share one timeline?
   Expected answer: visual frames and audio samples must refer to the same
   seconds. Independent origins or rounding rules can create false misalignment.

## Sources

- [FFmpeg command documentation](https://ffmpeg.org/ffmpeg.html)
- [FFmpeg codec documentation](https://ffmpeg.org/ffmpeg-codecs.html)
- [PyAV time documentation](https://pyav.org/docs/stable/api/time.html)
- [PyTorch audio resampling tutorial](https://docs.pytorch.org/audio/stable/tutorials/audio_resampling_tutorial.html)
- [Wav2Lip paper](https://arxiv.org/abs/2008.10010)
