# What this project contributes

A single place to answer "what is new here". Nothing in this file is a new
measurement: every claim points at a result ID in
[result traceability](result-traceability.md) and a finding in
[findings](findings.md), both of which resolve to a file hash.

## How to read this file

Claims here come in three strengths and they are kept apart on purpose, because
"we measured X" and "nobody has measured X" are different kinds of statement and
only the first is ours to make.

**Measured.** An artifact in this repository contains the number. Anyone with
the repository can recompute it. This is the strongest claim and most of the
file is this.

**Not found in prior work.** A targeted literature search in September 2026 did
not surface the result. That is weaker than it sounds: the search was targeted,
not systematic, it covered a handful of queries rather than a field, and absence
from a search is not absence from a literature. Treat every one of these as
"needs a citation check before the viva", not as priority.

**Prior work exists.** Named here so the project does not claim something that
is already published. Our version may still be a useful replication or a
sharper measurement, and where that is the case it says so.

**One limit applies to everything below.** Of 29 registry rows, 1 is `accepted`,
27 are `provisional (1 seed)` and 1 is `pending`. The protocol asks for three
training seeds and only the visual branch has them. F21 measures what that
costs, and the answer is that seed alone moves the cross-corpus number by 0.0733,
which is wider than several effects reported here. Nothing in this file should
be presented as settled until the seeds are done.

## 1. A detector that is accurate and provably not using the cue it is named for

**Measured.** `B-training-histories`, `B-lavdf-lipsync`, `B-stream-scores`, F1.

Two cross-attention streams were built to measure audio-visual correspondence.
Diagonal attention mass, the fraction of attention near the diagonal, was
recorded every epoch and never included in the loss. Across every epoch of three
separate training runs it sat at its chance value and stayed there, while the
emotion stream's ROC-AUC rose from 0.9847 to 0.9991 and became the most accurate
model in the project.

The strongest version is on LAV-DF, where the corpus itself forbids the
alternatives: matched windows from the same file, one on a 0.66-second
manipulated span and one missing it, holding speaker, codec and re-encoding
constant. The stream reads 0.9969 ROC-AUC there with diagonal mass still at
chance. So the accuracy is real, the mechanism is absent, and the corpus design
rules out the usual explanations.

**Prior work exists** on cross-attention lip-sync detection: DF-TransFusion uses
lip-audio cross-attention, and LIPINC-V2 uses cross-modal transformers with
inconsistency losses. We do not claim the architecture.

**Not found in prior work:** recording a mechanism metric every epoch as a
never-optimised falsification probe, and reporting that it sits at chance while
the model is accurate. The searched lip-sync literature reports detection
performance; it did not surface a case of a paper publishing that its own named
mechanism was absent. This is the result we would put first.

## 2. A benchmark defect that would have produced a near-perfect audio branch

**Measured.** `C-df26-audio-audit`, F14.

DF26 is a 2026 benchmark of 2,691 single-person public-speaking clips, 271 real
and 2,420 from seven modern generators. Before training anything on it we ran
ffprobe over 30 clips per group and found that **audio presence predicts the
label almost perfectly**: every commercial generator has a track on every
sampled clip, every open-source generator has none, and only 3 of 30 sampled
real clips carry one.

An audio branch trained on DF26 would have scored near 1.0 by detecting a
container property, and it would have looked exactly like the audiovisual
contribution the benchmark's own limitations section invites.

**Not found in prior work.** The DF26 paper reports human and detector
performance near chance and names audio, speech quality and lip-sync as future
work; the audio-presence confound did not appear in it or in the search. This is
concrete, cheap to verify, and useful to anyone about to use that benchmark.

## 3. Four shortcuts caught before they produced a result, not after

**Measured.** F14, F15, `C-ffpp-logo`, and the handoff's shortcut list.

Each was caught by one cheap check run before training rather than by a
post-hoc explanation of a suspicious number:

- Veo 3 lab clips scored against talking-head reals, which would have produced a
  perfect-looking content classifier.
- Real clips on both sides of a generator split, and in FaceForensics++ a fake
  can be matched to its own source video, so the split must be on source as well
  as generator.
- Generation modes of one model treated as separate generators, which leaks a
  model through its sibling. Grouping modes under one model is what gives DF26
  seven generators rather than ten.
- The DF26 audio confound above.

**Not found in prior work** as a practice reported in this form. Individual
leakage types are well known. What we did not find is a paper that enumerates
the shortcuts it caught before reporting, with the check that caught each one.

## 4. A project that publishes the claims it withdrew

**Measured.** The superseded table in [findings](findings.md), seven rows.

Claims this project made and later retracted with the measurement that killed
them, including: that emotion's diagonal mass was six times chance (it was
compared against a different query length, and at 8 steps 0.3438 *is* chance);
that freezing BatchNorm was a clean win (it is a trade, F2); that training on a
more varied corpus would improve transfer (FF++ made DFDC transfer worse, 0.7611
to 0.4920, F10); and that camera motion causes false alarms (measured with the
opposite sign, F6).

**Not found in prior work** at this granularity. Papers report limitations;
a retraction log with the superseding measurement for each withdrawn claim is
not something the search surfaced.

## 5. The cue-label trap that makes a fusion gain look fifteen times smaller

**Measured.** `A-branch-ablation`, F19.

FakeAVCeleb contains `RealVideo-FakeAudio` clips. For a visual branch those are
genuine, because the video really is unmanipulated, and the branch reads 0.9988
in-domain on that task. Fusion answers the clip-level question, where the same
clip is fake, and the identical visual logits read 0.9742.

Both are correct and they are not comparable. Quoting fusion's 0.9990 against
the branch's 0.9988 makes fusion look worth +0.0002. Measured on one task,
paired and clustered on speaker, the gain is **+0.0159 [+0.0129, +0.0190]**, and
all of it is the 75 clips a visual branch cannot detect even in principle.

**Not found in prior work** as a stated evaluation pitfall, though it is
specific to corpora with cue-disjoint categories. It is a trap this project fell
into and then caught, and the error-analysis script now excludes those clips
automatically rather than trusting anyone to remember.

## 6. A measured ceiling on what any fusion head could achieve

**Measured.** `A-error-analysis`, F23.

On DFDC the fused score and its own visual branch are both wrong on **1,585 of
2,351** compared clips. No fusion head over these branches can recover those:
the information is not in the inputs. Fusion repairs 314 of the visual branch's
errors and breaks 5, a strongly asymmetric trade that ROC-AUC does not show
because it changes threshold decisions without improving the ranking.

So 67 percent of that corpus is beyond head design, and head work is work on the
remaining third.

**Not found in prior work** as a reported quantity. Ablations typically report
what fusion gains; the both-wrong count bounds what it could ever gain and is
one line of analysis on predictions people already have.

## 7. Seed variance larger than the effects being reported

**Measured.** `A-visual-seeds`, F21.

Three seeds, everything else fixed. In-domain: 0.9988, 0.9991, 0.9993. On DFDC:
**0.7611, 0.6963, 0.6878**, a spread of 0.0733 against an in-domain spread of
0.0005.

That spread is four and a half times this project's in-domain fusion gain and
nine times its cross-corpus fusion difference. And the seed every result in this
repository reports, seed 17, is the **best of the three**, against a mean of
0.7151.

The fused system is worse, not better. Scoring all three seeds through the same
head gives DFDC 0.7500, 0.6814 and 0.6442, a spread of **0.1057** against 0.0001
in-domain, and again the reported seed is the best of the three. Combining three
branches whose seeds all move did not average the noise away; it compounded it.
That spread is twelve times the fusion-against-visual difference the project
uses it to discuss.

**Prior work exists** on seed sensitivity in deep learning generally. **Not
found** for deepfake cross-corpus transfer specifically, and rarely reported at
all, because reporting it weakens your own headline. Ours weakens ours.

## 8. Corruption that destroys the decision while leaving the ranking intact

**Measured.** `A-corruption-visual`, `A-corruption-audio`, F24, F25.

JPEG at quality 75, lighter than a messaging app applies, costs 0.0054 of
ROC-AUC and takes false alarms from 0 to 76 of 150 genuine clips. At quality 50
**every** genuine clip is called fake while ROC-AUC still reads 0.9187.

The audio branch fails in the opposite direction: noise at 20 dB SNR costs
0.0227 of ROC-AUC and misses 857 of 859 fakes. Visual degradation calls
everything fake; audio degradation calls everything genuine. A fused system
inherits both and they do not cancel.

A clean negative: a half-second silence shift changes nothing, so the silence
shortcut the threat model names is absent from this checkpoint.

**Prior work exists**, and this is the item most at risk of overclaiming.
"When AUC Misleads: Polarization-Aware Evaluation of Deepfake Detectors under
Domain Shift" makes the general argument, compression-induced false positives
are documented, and TPR at low FPR is already the recommended operating metric
in moderation contexts. We claim the specific paired measurement, not the idea.

**What may still be ours:** the same pattern measured independently three times
in one system, by three different mechanisms, all showing ranking surviving
where the decision does not: the `rtvc` head holdout (AUC falls 0.0437 while 73
of 75 clips go missed, F20), the confidence inversion (F22), and this. That
convergence is the argument, not any one table.

## 9. Confidence that inverts cross-corpus, closing the abstention route

**Measured.** `A-error-analysis`, F22.

On both cross-corpus sets the model is **more confident when wrong than when
right**: 0.8742 against 0.8299 on DFDC, with 951 of 1,590 fusion errors past 0.9
confidence, and 1,482 of 1,942 for the visual branch.

This matters because it closes a specific escape route. F9 already found that
abstaining on partial coverage buys no accuracy; this shows abstaining on low
confidence cannot work either, because confidence does not separate right from
wrong once the corpus changes. It is also why the density model in F17 works:
it scores the input rather than trusting the detector's certainty, and it
intercepts 79.4 percent of unfamiliar genuine clips at 8.7 percent abstention on
familiar ones, taking false accusations on unfamiliar real video from about 91
percent to roughly 19.

**Prior work exists** on miscalibration under distribution shift, which is a
well-established result. Our contribution is the operational consequence
measured on this pipeline, and the density model as the response.

## 10. A false-alarm rate that ranges from 0 to 91 percent with capture conditions

**Measured.** `A-false-alarms`, F16.

Genuine clips called fake: 0.00 percent in-domain, 4.14 on DFDC, **90.97 on
Celeb-DF**, where the visual branch calls 141 of 155 authentic videos
manipulated. Forcing zero false alarms on Celeb-DF needs a 0.9998 threshold, at
which nothing is caught at all.

The thresholds this system ships with are chosen by Youden's J, which weights
sensitivity and specificity equally. That is the wrong rule for any deployment
that cannot accuse a genuine video, and the finding says so.

**Prior work exists** on cross-dataset degradation. The framing we would defend
is the 0-to-91 range within one frozen model as a function of capture conditions
alone, reported as the deployment-blocking number rather than as a caveat.

## What we explicitly do not claim

- **Not a new architecture.** EfficientNet-B0 plus GRU, Wav2Vec2, cross-attention
  over temporal tokens. All standard.
- **Not state-of-the-art accuracy.** The research question was answered
  negatively: cue-specific fusion does not generalize better than a strong
  visual baseline on this data, cross-corpus -0.0085 [-0.0196, +0.0027].
- **Not a deployable system.** The [model card](../model-card.md) gives five
  measured reasons it should not be deployed.
- **Not priority over anything.** Every "not found" above is one targeted search
  away from being wrong, and each needs a citation check before it goes in a
  thesis.
- **Not a frozen-probe contribution.** "What Do Deepfake Benchmarks Measure? An
  Audit Using Frozen Self-Supervised Representations" (arXiv 2606.26384) covers
  ground close to our `C-ffpp-logo` region-of-interest result, and should be
  read before that finding is written up.

## The through-line

If the project has one argument, it is that **a number recorded beside the
training objective and never optimised against it is worth more than the
objective.** Diagonal attention mass caught a mechanism that was absent.
Per-slice error counts caught three threshold collapses that ROC-AUC hid. An
ffprobe pass over 30 clips caught a benchmark confound before it became a
result. None of those cost more than an afternoon, and each one changed what
could honestly be claimed.

That is the transferable part, and it is why this file leads with the failures.
