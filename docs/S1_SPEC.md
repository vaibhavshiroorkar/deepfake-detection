# S-1 design spec (fixed before any measurement)

Status: written 2026-10-25 before the full run. Smoke test on 3 clips per dataset (not a result) only checked that the
pipeline runs; no parameter was changed after seeing it. Code: `src/dfd/rppg/pulse.py` (parameters in `S1Params`),
driver `scripts/rppg/run_s1.py`. "IC" marks an implementer's choice, not taken from the source. "PD" marks a value
pre-declared in docs/ITERATIONS.md section I-7. "SRC" marks a value from R-057 (Wang, den Brinker, Stuijk, de Haan,
IEEE TBME 64(7), 2017, author's pre-review version).

## 1. Question and pass rule (PD)

Can a pulse signal be recovered from our real videos at all? Real clips only. S-1 passes if at least 60 percent of the
sampled real clips have a peak SNR above the 95th percentile of the POOLED shuffled-control SNR values (pooled over all
usable clips and all shuffles). The ground-truth heart-rate clause of I-7 does not apply (no dataset with pulse
ground truth is held, UBFC-rPPG is DEFER in D-021).

## 2. Data and sampling

| Item | Rule |
|---|---|
| Datasets and splits | FF++ c23 original (real) sequences in split train and val (860 videos); Celeb-DF v2 real videos in dev_x (146 videos); LAV-DF real videos in the official train split (`split == "train"`, `n_fakes == 0`, `modify_video` and `modify_audio` false, `original` null: 21,254 videos). Never test_in, test_x or the LAV-DF dev and test splits |
| Record selection | FF++ and Celeb-DF: `dfd.data.loaders.select_records(manifest, split_csv, split, modality="video")`, keep label 0. LAV-DF has no manifest and no split CSV in this repository; the records are read from `data/lavdf/LAV-DF/metadata.min.json` (its `split` field is the official split) (IC) |
| Candidate order | sorted by sample_id (string order) |
| Eligibility (IC) | the video has at least n frames, n = round(10 s x fps). FF++ and Celeb-DF: frame count and fps read with OpenCV before sampling (metadata, not an outcome). LAV-DF: `video_frames >= 250` from the metadata (the files are 25 fps, verified on the sampled files), then re-checked in the worker. A sampled clip that turns out too short is kept in the denominator as unusable (reason `short_file`) |
| Sample | seed 0: `numpy.random.default_rng(0).permutation(n_eligible)[:150]`, taken separately for each dataset, then processed in sorted order. Fewer than 150 eligible: all eligible clips. Eligible counts (from the metadata scan on 2026-10-25): FF++ 839 of 860, Celeb-DF 132 of 146, LAV-DF 3,731 of 21,254 |
| Clip identity | one clip per video, no per-group cap. Group counts per dataset are reported (Celeb-DF dev_x has only 15 identity groups among its 146 real videos, so 132 eligible clips are not independent persons; the Wilson interval treats clips as independent, which is optimistic for Celeb-DF) (IC) |

## 3. Segment

Contiguous frames at the native frame rate from the ORIGINAL video, not the 16-frame crops of D-034.

| Item | Rule |
|---|---|
| Length | n = round(10 x fps) frames (PD: about 10 s). 250 frames at 25 fps, 300 at 30 fps |
| Start (IC) | frame round(0.5 x fps), i.e. skip the first 0.5 s (encoder start-up, possible fade-in); if the video is too short for that, the largest start that still gives n frames (down to frame 0) |
| Decoding | OpenCV sequential read (grab and discard to reach the start, no seeking), BGR to RGB (IC) |
| Shorter videos | not sampled (not eligible); reported as availability. No padding, no looping |
| Minimum usable length (IC) | at least round(9 x fps) frames actually decoded, else unusable (reason `short_read`) |

## 4. Face and skin region per frame (all IC)

1. Detector: the project's YuNet (`make_detector("yunet", DEFAULT_MODEL_DIR)`, defaults score 0.6, NMS 0.3, max side 640), CPU, run on every frame of the segment, OpenCV threads set to 1 per worker.
2. Face: the largest box per frame. A box with min(w, h) below 48 original pixels counts as no face.
3. Box smoothing (causal, no look-ahead): smoothed = 0.7 x smoothed + 0.3 x new box; if the IoU of the new box with the smoothed box is below 0.3, the smoothed box is reset to the new box. Reason: per-frame box jitter changes the pixel set and adds colour noise unrelated to the pulse.
4. ROI: the central part of the smoothed box, [0.2, 0.8] of its width and of its height, clipped to the frame.
5. Skin mask inside the ROI: OpenCV YCrCb with 133 <= Cr <= 173 and 77 <= Cb <= 127 (a common fixed skin rule, not tuned). If fewer than max(30 pixels, 20 percent of the ROI) are skin, the frame counts as no face.
6. Per-frame feature: mean R, G, B over the skin pixels (8-bit gamma-encoded values, no linearisation).

## 5. Frames without a face

The RGB row of such a frame is NaN. The trace is linearly interpolated per channel (constant extrapolation at the ends). Clip usable only if the share of frames with a face is at least 0.8 (reason `face_share` otherwise) and the longest run of frames without a face is at most 1 s (reason `long_gap` otherwise). Unusable clips are not measured and have no shuffles; they stay in the denominator of the primary pass rule and count as "not above the threshold" (conservative). The share among usable clips is reported as a second figure.

## 6. POS (SRC, Algorithm 1 and equations (32) to (34), page 6 and 7; implemented without a band-pass as in the source)

For every window of l frames (window slides one frame): temporal normalisation C_n = C / mean over the window, per channel; S1 = G_n - B_n, S2 = -2 R_n + G_n + B_n (projection matrix (33)); h = S1 + (std(S1)/std(S2)) S2 (34); h made zero-mean; overlap-add into H. Window l = round(1.6 x fps) frames (SRC: l = 32 at 20 fps, 1.6 s): 40 frames at 25 fps, 48 at 30 fps (rounding is IC). Standard deviation with ddof 0 (the ratio is independent of ddof). If std(S2) is 0, alpha is set to 0 (IC, guard).

## 7. Spectrum and SNR

| Item | Rule |
|---|---|
| Detrend and window (IC) | linear detrend of H, then a Hann window over the whole segment |
| Zero padding (IC) | FFT length the next power of two of at least 100 x fps samples, i.e. bin spacing at most 0.01 Hz (about 0.0073 Hz at 30 fps, 0.0061 Hz at 25 fps). The natural resolution is 1/T = 0.1 Hz; padding only interpolates |
| Peak (PD) | the maximum of the periodogram in 0.7 to 3.0 Hz (42 to 180 bpm); bpm = 60 f |
| SNR (SRC: ratio of energy around the first two harmonics to the rest of the spectrum, as used in the source benchmark; the source places the harmonics at the reference pulse frequency, here the blind peak is used (IC)) | signal = sum of power within +-delta of the peak f0 and within +-delta of 2 f0; noise = all remaining power inside 0.7 to 3.0 Hz; both bands are cut to 0.7 to 3.0 Hz; SNR in dB = 10 log10(signal / noise) |
| delta (IC) | 0.2 Hz, equal to 2/T for T = 10 s, the half-width of the main lobe of a Hann window of that length, so the leakage of a true sinusoid falls into the signal band. The value of delta in the source's own SNR is not in the text read (it refers to de Haan and Jeanne 2013, NR) |
| Note | the peak is chosen blind (maximum of the spectrum), so the SNR is biased upwards even for noise; this is why the shuffled control uses exactly the same code |

## 8. Negative control (PD, with IC counts)

For every usable clip, 20 independent random permutations of the rows of the filled per-frame RGB trace (time shuffle; at least 10 is the pre-declared minimum) pass through the same POS, spectrum and SNR code. Each clip has its own random generator, `default_rng([0, 7, dataset_index, clip_index])`, so results do not depend on worker scheduling. Threshold = the 95th percentile (numpy linear interpolation) of all pooled shuffled SNR values (usable clips x 20, about 8,000 values). A real clip is "above" if its SNR is strictly greater than the threshold.

## 9. Reporting

Per dataset and pooled: eligible and candidate counts, sampled, usable and unusable (with reasons); the fraction of sampled clips above the threshold with the Wilson 95 percent interval (primary), the fraction among usable clips with its interval (secondary), quantiles of the real and of the shuffled SNR, number of identity or source groups in the sample, median face share, time per clip. Pass rule applied to the POOLED primary fraction (sum above over sum sampled) against 0.6. The per-dataset fractions are reported but do not enter the rule (IC: the pre-declared rule says "real clips", pooled). Outputs: `data/_results/S-1/{clips.csv, clips.npz, traces.npz, summary.json, result_block.md}`.

## 10. Known limits written before the run

- The shuffle null keeps the marginal distribution of the RGB values and destroys all temporal structure, including the slow, coloured (1/f-like) drift and motion of real traces. It is therefore a lenient null: real clips can exceed it through motion or lighting periodicity or compression artefacts without any pulse. A pass of S-1 shows "more narrow-band structure than a time-shuffled trace", not "a heart rate was recovered". No pulse ground truth is available to check it.
- The source benchmark used uncompressed 20 fps video; our videos are H.264 compressed (FF++ c23 is CRF 23), which the source's authors note can remove or pollute the pulse information (footnote 8, a statement about MAHNOB-HCI).
- Clips are not independent persons (see section 2).
