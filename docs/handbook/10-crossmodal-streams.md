# 10: Cross-modal streams

## Why this exists

The visual branch reached ROC-AUC 1.0000 on FakeAVCeleb validation and on its
held-out test partition, with zero errors on 1,195 clips. It then scored 0.648
on Celeb-DF-v2 and a 19 percent detection rate on MNW.

The shape of that failure is what matters. On Celeb-DF it caught 303 of 307
forgeries but called **130 of 155 genuine videos fake**, with a mean probability
of 0.835 on real video. And wav2lip, a method that is in the training set,
scored 0 of 7 on MNW.

That is not a model too weak to see forgeries. It learned what FakeAVCeleb looks
like: its compression, its resolution, its crop framing. Any unimodal appearance
classifier is exposed to this, because "what does authentic video look like" is
a question about the training corpus rather than about forgery.

A cross-modal stream asks a different question. Whether this audio matches this
mouth is answered *inside* one clip, so there is no dataset-level appearance
prior available to memorise instead.

## The mechanism

Audio is the Query. Video is the Key and the Value.

```
mouth  [B, 50, 3, 112, 112] -> frame encoder -> [B, 50, Dv] -> [B, 50, 256]
audio  [B, 32000]           -> wav2vec2      -> [B, Ta, Da] -> [B, Ta, 256]
both resampled to a common step count
cross-attention  softmax(QK^T / sqrt(d_k)) V   -> weights  [B, 50, 50]
                                               -> attended [B, 50, 256]
mismatch = audio - attended                     the residual
mean over time, Linear, LayerNorm               -> embedding [B, 256]
```

The direction is the claim: the sound is what we have, and we ask how well the
mouth accounts for it.

`dashboard/lib/cross_modal.py` is the written specification, and
`tests/test_cross_attention.py` asserts the trainable module reproduces its
numpy result to 1e-5, so the two cannot drift apart.

### The residual, not the attended vector

The stream's product is `audio - attended`, not `attended`. Two reasons, and the
second is the decisive one.

`stream_spec.py` calls this a "synchronisation-mismatch vector", and the
mismatch is what the mouth failed to explain about the sound. The residual is
the literal reading.

It is also the only version that trains. Attention is near-uniform at
initialisation, so every attended step collapses onto the mean video token and
averaging over time erases what little structure survives. Measured on an
untrained stream, shifting the audio by 320 ms moved the embedding by **2e-7**,
which is float noise, not a gradient. The residual moved it by **0.072** against
an embedding scale of 1.0.

## `diagonal_mass`, the independent check

The fraction of attention mass within one step of the diagonal, scaled for
differing token rates between the two modalities.

In a genuine recording, sound arrives at a near-fixed offset from the
articulation that produced it, so a stream that learned the real correspondence
concentrates its attention near the diagonal. A stream that found a shortcut
does not.

This makes the stream checkable **independently of whether it classifies well**,
which is a stronger position than accuracy alone. It is recorded every epoch and
**never optimised**: a model rewarded for producing diagonal attention would
learn to produce it without learning synchronisation, destroying the only
honest signal available.

Chance for a 50x50 map with a band of 1 is 3/50 = 0.060.

### It has already earned its place twice

On a synthetic fixture separating classes by a constant offset, training loss
fell sevenfold from 0.77 to 0.11 while diagonal mass stayed flat at 0.343. There
was no temporal correspondence to learn, the model took the offset shortcut, and
the metric said so.

On FakeAVCeleb it sat at 0.0592 for every epoch of a real run whose best
validation loss was 0.645. Without the metric that number would have read as
modest progress. It was not: the attention never left uniform.

The cause is structural. **FakeAVCeleb manipulates whole clips.** Every frame of
a wav2lip clip is forged, so there is no question of *when* the audio stopped
matching the mouth, and recognising the generator's artifacts is easier than
learning alignment.

## Why LAV-DF

LAV-DF hides a content-driven forgery inside an otherwise genuine clip: a word
is chosen for the largest swing it causes in perceived sentiment, text-to-speech
generates replacement audio, and facial reenactment follows it. The manipulated
span is 0.66 seconds at the median inside a 7.3 second clip, and the metadata
says exactly where.

| | FakeAVCeleb pilot | LAV-DF |
|---|---|---|
| Label prior | 87.5% fake | 42% fake, near-balanced across all four types |
| Audio-only fakes | 500 | 33,170 |
| Forgery extent | whole clip | localized, 0.66 s median |

It is also the only corpus here that populates all four manipulation types
honestly. Celeb-DF-v2 and MNW have to be forced into `FakeVideo-RealAudio`
because they carry no audio manipulation at all.

### Matched pairs

Every forgery yields **two** windows from the same file:

- one centred on the manipulated span, labelled fake
- the widest window that misses every manipulated span, labelled real

Same speaker, same lighting, same microphone, same codec, same re-encoding pass.
They differ in one thing: whether that particular two seconds was manipulated.
A model cannot separate them by recognising the generator's compression
signature, because both windows carry it.

99.7 percent of forgeries leave room for both.

The 0.3 percent that do not contribute their fake window alone. Pairing them
against overlapping audio would put a manipulated span under a real label, which
is worse than a smaller dataset.

## Protocol constraints

**Window placement.** The cache pins its synchronisation window near the start
of a clip. Measured on LAV-DF, **56,582 of 99,873 forgeries (57 percent) fall
entirely outside it**, so a fixed window would train on contradictions.
`ClipRecord.sync_start_sec` lets a dataset place the window where the
manipulation is.

**Cross-corpus evaluation needs DFDC.** FakeAVCeleb and LAV-DF are both built on
VoxCeleb2, so scoring one on the other changes the generator and not the corpus.
Celeb-DF-v2 and MNW carry no audio. DFDC was filmed with paid actors and is the
only audio-bearing corpus here independent of everything the streams train on.

**The emotion stream has an alignment gap.** `visual_view` samples 16 frames
across the whole clip while `audio_view` is a fixed 4.0 second window, and 70
percent of clips run longer than that. Comparing a face at t=10 s against vocal
affect that ended at t=4 s measures nothing. That stream must trim itself to the
overlapping prefix and report the coverage. The lip-sync stream is unaffected:
its two views are both exactly 2.0 seconds from the same start.

## Encoder choices, and why they are not the specified ones

`stream_spec.py` names AV-HuBERT and Whisper. Neither is used yet, deliberately.

**AV-HuBERT** needs fairseq, which pins PyTorch to 1.08, 1.13 or 2.0. This
environment runs torch 2.12.1+cu130 and installing fairseq would take the
working CUDA stack down with it.

**Whisper's encoder does not accept a waveform.** It takes `[B, 80, 3000]`
log-mel and rejects anything else outright:

```
ValueError: Whisper expects the mel input features to be of length 3000,
but found 32000.
```

3000 frames is 30 seconds. The lip-sync window is 2 seconds, so Whisper would
mean 28 seconds of padding and 100 useful tokens out of 1500.

Wav2Vec2 takes a raw waveform of any length and is already used by
`branches/audio.py`. The contribution is the mechanism, not the encoder, and
swapping one in later is a clean ablation rather than a rewrite.

## Training

`ddf train stream --stream lipsync`. The encoders are frozen for the first
epochs so the randomly initialised attention head settles before anything
pretrained moves.

**Use `--encoder-learning-rate`.** A single rate across a new head and 100M
pretrained encoder parameters drove training loss to 0.039 while validation
climbed to 2.56. The default is 5e-6 against the head's 1e-4.

## What to read from a run

Loss falling with `diagonal_mass` rising is evidence the stream learned the real
correspondence.

Loss falling with `diagonal_mass` flat at chance means it found a shortcut, and
the accuracy should not be quoted as evidence of synchronisation.

An untrained stream is also **order-blind**, because uniform attention makes the
attended vector a mean over frames and a mean cannot see order. Becoming
order-sensitive is what training must achieve, so that check should invert on a
trained checkpoint.
