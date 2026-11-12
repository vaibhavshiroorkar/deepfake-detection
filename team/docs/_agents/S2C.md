# S2-C: data-preparation pipeline and its speed measurement

Written 2026-10-24 by stream S2-C. Code in `src/dfd/prep/` and `tests/prep/`; raw numbers in `data/_prep_bench/`; crops in `data/_crops/` (all git-ignored).

**Caveat that applies to every number below.** Every run was made while other processes used the machine: FF++ downloads, ASVspoof and Celeb-DF extraction and normal desktop programs. Windows reported about 13 to 17 percent total CPU busy before each of my runs (the "system CPU load before run" line of each command). Numbers are therefore indicative, not clean, and they include warm file-cache reads (the sample is about 220 MB, so it fits in cache after the first pass). Cold-disk reads were not measured separately. Machine: Ryzen 9 7900X (12 cores, 24 threads), 31 GB RAM, Windows 10, CPU only, no GPU used.

## 1. What was built

| File | What it is |
|---|---|
| `src/dfd/prep/video.py` | Uniform frame sampler `sample_indices` (centres of N equal segments) and three decode backends (`cv_seq`, `cv_seek`, `ffmpeg`) as lazy generators; `probe`, `is_stable` (no `.part`, not modified for 20 s) |
| `src/dfd/prep/face.py` | Three detectors behind one interface (`yunet`, `ultraface`, `scrfd`, all through OpenCV, no torch), largest-face rule, square crop with enlarge factor, black padding outside the frame, resize |
| `src/dfd/prep/audio.py` | `extract_audio` (ffmpeg to 16 kHz mono float32, empty array if no audio track), `extract_audio_to_wav`, `load_audio` (FLAC/WAV via soundfile, resample via ffmpeg if the rate differs) |
| `src/dfd/prep/pipeline.py` | CLI and `run()`: manifest to crops plus index, process pool, resumable, one frame in memory at a time |
| `src/dfd/prep/bench.py` | The measurement commands used for this report (`sample`, `decode`, `decodepool`, `detect`, `stress`, `jpeg`, `audio`) |
| `tests/prep/` | 26 tests, about 1.5 to 3 s in total (synthetic ffmpeg videos, fake detector for pipeline tests) |

### Packages added (pyproject.toml and uv.lock)

- `opencv-python-headless` 5.0.0.93 (OpenCV 5.0.0, FFmpeg backend present, `cv2.FaceDetectorYN` present). Note: OpenCV 5 no longer has `cv2.CascadeClassifier` or `cv2.dnn.readNetFromCaffe`, so the Haar cascade and the res10 SSD (Caffe) detectors cannot be used; `cv2.dnn.readNet` on ONNX works.
- `soundfile` 0.14.0 (pulls `cffi` 2.1.1 and `pycparser` 3.12).
- `psutil` 7.2.2 (system CPU percent in the benchmarks and the pipeline summary; optional at run time).

Not installed: MediaPipe (a `uv pip install --dry-run` showed it would pull `opencv-contrib-python`, which clashes with the headless package, and matplotlib). No torch, no onnxruntime.

### Model files (downloaded once, kept in `data/_prep_bench/models/`, NOT in git; re-download if that folder is wiped)

| File | Source | SHA-256 | Licence |
|---|---|---|---|
| `yunet_2023mar.onnx` (232,589 B) | `https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx` | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` | not verified by me (NR); check the opencv_zoo folder before logging in DATA_LOG |
| `ultraface_rfb320.onnx` (1,270,727 B) | `https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/raw/master/models/onnx/version-RFB-320.onnx` | `34cd7e60aeff28744c657de7a3dc64e872d506741de66987f3426f2b79f88017` | not verified by me (NR) |
| `det_500m.onnx` (2,524,817 B, from `buffalo_sc.zip`) | `https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip` | `5e4447f50245bbd7966bd6c0fa52938c61474a04ec7def48753668a9d8b4ea3a` | InsightFace model packs are, as far as I know, research-only; not verified here (NR). Label as research-only until checked (CLAUDE.md rule 7) |

I did not edit DATA_LOG.md (not my file); the main agent should record these three before use.

## 2. How to run

```
# crops for a manifest (the data manifests store paths relative to data/, so --root defaults to data)
PYTHONPATH=src .venv/Scripts/python.exe -m dfd.prep.pipeline --manifest data/_manifests/ffpp.manifest.csv --out data/_crops/ffpp_c23_yunet_n16 --workers 16
# options: --detector yunet|ultraface|scrfd  --backend cv_seq|cv_seek|ffmpeg  --n-frames 16  --enlarge 1.3  --size 224  --quality 95  --limit K  --min-age 20
# tests
.venv/Scripts/python.exe -m pytest tests/prep -q
# measurements (all outputs go to data/_prep_bench/)
PYTHONPATH=src .venv/Scripts/python.exe -m dfd.prep.bench sample --per-method 25
PYTHONPATH=src .venv/Scripts/python.exe -m dfd.prep.bench decode --every 3
PYTHONPATH=src .venv/Scripts/python.exe -m dfd.prep.bench detect
```

The package is not installed in the venv, hence `PYTHONPATH=src` (pytest already has `pythonpath = ["src"]`).

Output of the pipeline under `--out`: `<folder>/<frame_index>.jpg`, per-video `index.csv` and `_done.json`, a merged `index.csv`, `videos.csv` (status and seconds per stage per video), `summary.json`, `params.json`.

- Folder of a sample is its `sample_id` with every character outside `[A-Za-z0-9._-]` replaced by `_` (Windows does not allow `:`; manifest ids look like `ffpp:Deepfakes:123_456`). The index keeps the true `sample_id`; `crop_path` is relative to `--out`.
- Index columns: `sample_id, frame_index, source_frame_index, detected, box, crop_path, score, n_faces, crop_box, note`. `frame_index` is the position among the sampled frames (0 to N-1), `source_frame_index` the frame number in the video, `box` the detected face `x,y,w,h` in source pixels, `crop_box` the enlarged square `x0,y0,side`. Frames without a detection have `detected=0` and empty `box` and `crop_path`; an undecodable frame has `note=not_decoded`. Nothing is dropped.
- Resume: a video is finished when `_done.json` exists with the same settings; a rerun skips it, retries errors and unfinished videos, and refuses to mix different settings in one `--out`. Files that are missing, end in `.part` or were modified in the last 20 s are skipped and listed as `skipped_unstable` (`--min-age`).
- Memory: frames are decoded and processed one at a time. Measured peak working set of one process on the largest sample video (1920x1080, 927 frames, 21 MB file; decoding it fully would need 5.77 GB): 133 MB (53 MB at start of the process). Command: in-process `process_video` with `psutil.Process().memory_info().peak_wset` (an ad-hoc snippet, not a bench command).
- The manifest of stream S2-B (`data/_manifests/ffpp.manifest.csv`, columns `sample_id, dataset, family, path, modality, ...`) works as is (tested with `--limit 12`; rows with `modality` other than `video` are ignored).

## 3. Measurements

Sample for all video measurements: `data/_prep_bench/sample_c23.csv`, made by `bench sample --per-method 25` (seed 0): 100 originals (youtube c23) and 25 videos of each of Deepfakes, Face2Face, FaceSwap, NeuralTextures (100 manipulated), total 200. Only files that were stable (20 s) and that ffprobe opened. All were h264 yuv444p. Frame heights in the 3,200 sampled frames: 480 p 53 percent, 720 p 36 percent, 1080 p 11 percent. Of 50 of them checked with `ffprobe -count_frames`, 0 had a frame count different from the OpenCV header value; frame rates 25 (25 videos), 30 (24), 24 (1); `r_frame_rate` equalled `avg_frame_rate` in all 50 (no sign of variable frame rate in this sample).

### 3.1 Decode backends (16 uniformly spaced frames per video, 1 thread, no detection)

Command: `bench decode --every 3` (67 videos, 1,072 frames; backends rotated per video so cold reads are spread). System CPU 7.9 percent before the run.

| Backend | mean s/video | median | p90 | ms/frame (mean) |
|---|---|---|---|---|
| cv_seq (OpenCV, read from start, `grab` and `retrieve` only the wanted frames) | 0.739 | 0.518 | 1.408 | 46.2 |
| cv_seek (OpenCV, `CAP_PROP_POS_FRAMES` per frame) | 2.876 | 2.471 | 7.562 | 179.8 |
| ffmpeg (subprocess, `select` filter, raw pipe) | 0.742 | 0.513 | 1.348 | 46.4 |

All three returned pixel-identical frames at identical indices on all 67 videos (printed check, 0 differences); the unit tests check the same on a lossless synthetic video. A repeat of the same command gave 0.748, 2.908 and 0.744 s/video.

- Threads: with 4 decoder threads in one process (`--threads 4 --backends cv_seq ffmpeg`, same 67 videos): cv_seq 0.318 s/video, ffmpeg 0.350. In a process pool one thread per worker is used (pool scaling is in 3.3).
- cv_seq by resolution (from the raw rows `data/_prep_bench/decode_threads1.csv`): 640x480 0.254 s/video (14 videos), 1280x720 0.704 (30), 1920x1080 2.311 (9).
- Dependence on N (20 videos, `bench decode --every 10 --n N`): cv_seq 0.522 s/video at N=4, 0.590 at 16, 0.777 at 64; ffmpeg 0.607, 0.598, 0.642; cv_seek 0.603, 2.431, 9.929. Sequential decoding costs almost the same for any N (the whole video up to the last sample is decoded anyway); seeking costs about 150 ms per frame in these files, so it is only competitive for very few frames.
- Per-video fixed cost: opening the header (`probe`) 4.6 ms (warm).
- Process start of an ffmpeg subprocess is about 28 ms (see 3.5), visible only on tiny videos.

### 3.2 Face detectors (CPU, 1 thread, in-process, detector only, same frames for every detector)

Commands: `bench detect` (3,200 frames: the 200 sample videos x 16 frames; yunet_full is YuNet without the 640 px resize) and `bench detect --n 64 --every 3 --tag _dense64 --detectors yunet ultraface scrfd` (4,288 frames: 67 videos x 64 frames). Machine busy as above (12.8 to 17.8 percent before). Raw rows: `data/_prep_bench/detect_frames*.csv`.

| Detector | model size | frames with at least one face (set 1 / set 2) | frames with more than one face (set 1 / set 2) | ms/frame mean (set 1 / set 2) | ms/frame at 480p, 720p, 1080p (set 1) |
|---|---|---|---|---|---|
| YuNet 2023mar (`cv2.FaceDetectorYN`, input long side resized to at most 640) | 0.23 MB | 100.00 / 100.00 percent | 11.2 / 14.8 percent | 12.3 / 12.1 | 13.2, 10.9, 12.6 |
| UltraFace RFB-320 (`cv2.dnn`, input 320x240) | 1.27 MB | 100.00 / 100.00 | 8.8 / 12.1 | 4.9 / 4.9 | 4.9, 5.0, 5.1 |
| SCRFD 500M (`cv2.dnn`, input 640x640) | 2.5 MB | 100.00 / 100.00 | 9.9 / 13.7 | 30.1 / 29.7 | 30.1, 29.7, 31.4 |
| YuNet at full resolution (set 1 only) | 0.23 MB | 100.00 | 16.4 | 33.0 | 13.7, 41.9, 97.0 |

Detection rate was 100 percent for all detectors, for real and for manipulated videos, and for each of the four methods, in both sets. FF++ sampled frames are close to frontal talking heads (mean largest-face box side about 230 px), so this rate does not separate the detectors; there is no ground truth, so it is a rate of "at least one box", not accuracy. Where the largest boxes of two detectors were both present, they overlapped with IoU of at least 0.5 in 99.3 to 99.8 percent of frames (set 1) and 99.9 percent (set 2). Resizing YuNet to 640 px did not change the largest box in practice (99.5 percent IoU agreement with full resolution) and was 2.7 times faster on average (7.7 times at 1080 p).

Failure behaviour, `bench stress` (200 frames, one mid-video frame per video, transformed; percent with at least one detection):

| Transform | YuNet | UltraFace | SCRFD |
|---|---|---|---|
| none | 100.0 | 100.0 | 100.0 |
| rotated 90 degrees | 69.0 | 72.5 | 94.0 |
| rotated 180 degrees | 91.5 | 92.0 | 82.0 |
| Gaussian blur sigma 4 | 100.0 | 100.0 | 100.0 |
| JPEG quality 10 | 100.0 | 100.0 | 100.0 |
| brightness x 0.15 | 100.0 | 96.5 | 100.0 |
| image scaled to 1/4 in a black canvas (face about 25 to 60 px) | 99.5 | 87.5 | 100.0 |
| image scaled to 1/8 | 69.0 | 32.0 | 72.5 |
| left half of the frame only (faces cut by the border) | 67.5 | 68.5 | 71.0 |

On 500 images that contain no face as a subject (random ImageNet-style objects and animals from `data/forensynths/cnn_synth_subset`, `0_real` folders of biggan, cyclegan horse, zebra, apple, orange; some contain people), images with at least one detection: YuNet 30 (6.0 percent), UltraFace 101 (20.2 percent), SCRFD 23 (4.6 percent). Quick look, not a measurement, at 24 of those images for UltraFace and YuNet: many hits are real human faces in the picture or animal faces (dog, bear, zebra, horse); UltraFace also fires on objects (fridge, jug, tractor, pomegranate), which I did not see in the YuNet sheet. A black frame and a noise frame gave 0 faces for all three. Every detector returns an empty list when nothing is found; there were no exceptions.

Largest-face rule (quick look at the frames where any pair of the three detectors had IoU below 0.5, 8 of 4,288 frames, from 2 videos): in a news-studio video the largest box was sometimes a face in the background screen or a bystander instead of the anchor, and in another video YuNet drew a large box on a window area. Multi-face frames are common in the pipeline output (359 of 3,200 YuNet frames, 11.2 percent; 47 of 200 videos have at least one; 19 videos have 8 or more of 16), so "largest face" is not always "the speaker". I did not measure how often the wrong face is chosen.

### 3.3 Pipeline throughput (YuNet, cv_seq, 16 frames per video, 224 px, JPEG quality 95, enlarge 1.3)

Command, for each K: `PYTHONPATH=src .venv/Scripts/python.exe -m dfd.prep.pipeline --manifest data/_prep_bench/sample_c23.csv --out data/_crops/bench_wK --workers K` (200 videos, 3,200 frames, 3,200 crops, 0 errors, 0 frames without detection; one OpenCV thread per worker). Raw: `data/_prep_bench/pipeline_wK.json`. "System CPU" is Windows total CPU percent averaged over the run, including the other processes (about 13 to 17 percent before each run).

| Workers | wall s | videos/s | frames/s | s/video (wall) | system CPU mean percent | in-worker s/video (decode, detect, write) |
|---|---|---|---|---|---|---|
| 1 | 159.7 | 1.25 | 20.0 | 0.798 | 14.7 | 0.797 (0.579, 0.192, 0.018) |
| 4 | 48.5 | 4.13 | 66.0 | 0.242 | 30.6 | 0.948 (0.705, 0.212, 0.021) |
| 8 | 33.9 | 5.91 | 94.5 | 0.169 | 48.3 | 1.301 (0.994, 0.270, 0.025) |
| 16 | 31.5 | 6.35 | 101.6 | 0.158 | 81.8 | 2.333 (1.771, 0.505, 0.038) |
| 24 (extra) | 32.1 | 6.22 | 99.5 | 0.161 | 94.7 | 3.431 (2.589, 0.739, 0.069) |

Repeat runs of the same command on an earlier build of the same code: 8 workers 5.78 and 5.92 videos/s, 16 workers 6.27 and 6.28, 24 workers 6.34. So run-to-run spread is about 2 percent at 16 workers and up to 2.5 percent at 8.

Reading: one worker is one core, 4 workers give 3.3 times, 8 give 4.7 times, 16 give 5.1 times the throughput of one; beyond 8 workers throughput nearly stops growing although CPU use keeps rising, and the in-worker time per video grows (0.8 s to 2.3 s). I did not isolate the cause (12 physical cores with SMT, memory bandwidth for 4:4:4 decoding, and the other processes are all candidates). Time split with one worker: decode 73 percent, detection 24 percent, crop and write 2 percent. Decoding the video is the bottleneck, not the face detector.

Same run with `--backend ffmpeg` at 16 workers: 5.98 videos/s (95.7 frames/s), against 6.35 for cv_seq (single run each).

Decode only, no detection, `bench decodepool --workers K` over the same 200 videos (warm-up excluded): 8 workers 8.07 videos/s, 16 workers 8.72, 24 workers 9.19. The upper limit of this pipeline on this sample with this machine state is therefore about 9 videos/s; detection and JPEG writing bring it to about 6.3.

### 3.4 Projections (stated as projections, from the numbers above)

- Full FF++ c23, 5,000 videos, 16 frames each (80,000 crops), the same kind of video as the sample (the sample was drawn randomly from the files present, the resolution mix of the full set may differ): at 16 workers 6.35 videos/s gives 787 s, about 13 minutes; at 8 workers 846 s, about 14 minutes; at 1 worker 3,992 s, about 67 minutes. Same caveat: measured with downloads running; also the file cache was warm. DFD (3,431 videos) was not measured and is not projected.
- LAV-DF scale, 136,304 videos, decoding only, from a different dataset: this is a projection, not a measurement, and LAV-DF is not on this machine. As a stand-in I re-encoded 40 of the FF++ sample videos to 224x224, yuv420p, H.264 (`ffmpeg -vf scale=224:224 -c:v libx264 -preset veryfast -crf 23`, mean 504 frames per clip, range 300 to 1249) in `data/_prep_bench/proxy224/` and measured: one thread, cv_seq, 16 frames: 0.031 s per video (1.94 ms per sampled frame; ffmpeg 0.063, cv_seek 0.124) (`bench decode --manifest data/_prep_bench/proxy224.csv --every 1`); pool, same files repeated 25 times (1,000 decodes, all from file cache): 315 videos/s at 16 workers and 384 videos/s at 24 workers (`bench decodepool --manifest data/_prep_bench/proxy224.csv --workers K --repeat 25`). If LAV-DF clips were about 500 frames each, 136,304 videos would take about 433 s (16 workers) or 355 s (24 workers), 6 to 7 minutes. The cost scales about linearly with frames per clip (sequential decoding), and LAV-DF's real clip length, codec settings, GOP and the cost of opening 136,304 files from disk (cold, not in my test) are unknown here; treat the figure as an order of magnitude, a lower bound for the disk-bound case.
- Audio: see 3.5. ASVspoof 2019 LA flac (122,299 files in `data/asvspoof2019_la/LA`): 4.76 ms per file measured, so about 582 s single-threaded for all files, projected from 200 random files.

### 3.5 Audio (`bench audio`, 1 thread, CPU 12.4 percent before)

- ASVspoof 2019 LA flac, 200 random files out of 122,299 (all three folders train, dev, eval): `load_audio` 4.76 ms per file, 3.36 s of audio per file on average, about 705 times real time; all sampled files are 16 kHz, so no resampling was needed.
- Videos with audio: no video of the datasets present has an audio stream. FF++ c23 originals: 0 of 20 files had audio (`extract_audio` returns an empty array, 28.2 ms per file, almost all of it ffmpeg process start). Celeb-DF v2: 0 of 50 random complete files gave audio through `extract_audio`, and `ffprobe` found no audio stream in 0 of 250 files (100 Celeb-real, 50 YouTube-real, 100 Celeb-synthesis, shuf-selected, extraction was still running). So `extract_audio` on real audio-bearing video has only been measured on a synthetic file: a 10 s test tone muxed into an H.264/AAC mp4 at 44.1 kHz made with ffmpeg (`data/_prep_bench/tone10s.mp4`): 30.9 ms per file including process start, 160,000 samples out (10.00 s at 16 kHz). This is a synthetic measurement only; no speed on a real audio-bearing deepfake video (for example FakeAVCeleb) exists yet.

### 3.6 Disk size of the crops

From the 200-video run (`du` and file sizes of `data/_crops/bench_w1`): 13,924 bytes per crop (224x224, quality 95), 222,786 bytes per video at N=16 (range by group 216,956 to 225,821). On disk, with NTFS 4 KB clusters, `du` reports 51 MB for the 200 videos (about 255 KB per video).

Projection for 5,000 videos (80,000 crops at N=16): about 1.11 GB of file data, about 1.28 GB on disk with cluster rounding. It scales linearly with N (N=32 doubles it).

JPEG quality (`bench jpeg`, 320 crops from 20 videos; PSNR against the uncompressed crop, mean):

| Quality | bytes per crop | PSNR dB | 5,000 videos x 16 frames (file data) |
|---|---|---|---|
| 100 | 28,838 | 48.66 | 2.31 GB |
| 95 | 13,740 | 45.49 | 1.10 GB |
| 90 | 9,792 | 43.82 | 0.78 GB |
| 85 | 7,960 | 42.62 | 0.64 GB |
| 75 | 6,169 | 40.79 | 0.49 GB |

Disk is not a constraint at any of these (404 GB free on D: per PLAN.md). Whether JPEG compression artefacts affect a deepfake detector is not measured here; the FF++ c23 source is already compressed, and the crop re-encodes it.

## 4. Recommendation to the main agent (not a decision; D-023, PROTOCOL section 9)

Evidence-based suggestion for the detector: YuNet 2023mar with the input resized to at most 640 px (the default in the code), N and enlarge left to you.

- Detection rate was 100 percent for all three on 7,488 FF++ frames, so rate does not discriminate them; the choice rests on speed, behaviour off the easy case and dependency cost.
- YuNet: 12 ms per frame (2.5 times faster than SCRFD, 2.5 times slower than UltraFace), 0.23 MB model, runs through `cv2.FaceDetectorYN` alone, 6.0 percent hits on face-free images against 20.2 percent for UltraFace, 99.5 percent hits for small faces (1/4 scale) against 87.5 percent for UltraFace.
- SCRFD found the most faces in rotated and tiny cases (94 percent at 90 degrees, 72.5 at 1/8 scale) and had the fewest hits on face-free images (4.6 percent), but costs 30 ms per frame and its weights come from InsightFace (research licence to be checked).
- UltraFace is the fastest but the weakest off the easy case; its saving is 0.12 s per video of about 0.8 s, because decoding dominates.
- Because decoding is 73 percent of the time per video, choosing a faster detector changes the total little: replacing YuNet by SCRFD would add roughly 0.28 s to the 0.8 s per video (30.1 - 12.3 ms times 16 frames), by my arithmetic from the per-frame numbers, not a pipeline run.

For decoding: `cv_seq` (OpenCV, sequential `grab`) and the ffmpeg pipe are equal in speed on a single process (0.739 against 0.742 s per video) and `cv_seq` was slightly faster in the 16-worker pipeline (6.35 against 5.98 videos/s, one run each); seeking is 4 times slower on FF++ files (long GOP) and should not be used. Use a pool of 8 to 16 workers with one thread each: 8 workers already give 93 percent of the throughput of 16.

Other facts that bear on choices:

- N (frames per video) barely changes decode time for `cv_seq`, but detection and writing cost scale with N, and disk scales linearly.
- The largest-face rule can pick a bystander or a false box in multi-person scenes (11 percent of frames have more than one box). If that matters for training, a rule such as keeping the face nearest the previous frame's face, or the highest score, would need its own check; not built.
- Nothing in the 200-video sample had zero detections, so the "recorded, not dropped" path is exercised only by the tests, not by real data; a real run should still count `detected=0` rows and see whether they cluster in particular videos.

## 5. Limits of these measurements

- Concurrent load (downloads, extraction, other programs; 13 to 17 percent of the CPU busy before each run), warm file cache, one machine, one sample of 200 videos drawn from the FF++ files that were complete at 2026-10-24 about 19:00 to 20:00 (not all 5,000). Sample drawn with seed 0, so it is reproducible only for the same set of files.
- No ground truth for faces: detection rate is not accuracy, agreement is only a proxy, and the visual checks are quick looks.
- Quality of the crops for the later detector (does 1.3 versus another factor, or JPEG 95, change accuracy) is not measured; that needs a model and belongs to the experiments.
- Only FF++ c23 was run through the pipeline. Celeb-DF v2 (extraction unfinished), DFD, LAV-DF (absent) and FakeAVCeleb were not measured; the LAV-DF number is a projection from re-encoded FF++ videos.
- Real audio-in-video extraction speed is unmeasured (no local video has audio); only a synthetic tone file was timed.
- The CPU utilisation is the system-wide figure; I did not separate my processes from the others.
- Pool scaling beyond 8 workers was not explained.

## 6. Surprising or notable

- FF++ c23 and Celeb-DF v2 videos carry no audio stream at all (0 of 20 and 0 of 250 checked). The audio branch of D-011 for video must therefore use other data (FakeAVCeleb and similar); FF++ gives only the visual path.
- FF++ c23 is H.264 High 4:4:4 (yuv444p) in all 50 videos probed, heavier to decode than the usual 4:2:0. Resolutions are mixed (480 p, 720 p, 1080 p); 1080 p videos take 9 times longer to decode than 480 p ones (2.31 against 0.25 s with cv_seq).
- OpenCV 5.0.0 removed `CascadeClassifier` and the Caffe loader, which is why the usual Haar and res10 candidates are absent.
- The first quick test gave 0.30 s for a 640x480 video; the mean over the sample is 0.74 s because of the 720 p and 1080 p videos. Quote the sample means, not that single video.
- Frame counts in the container header matched `ffprobe -count_frames` for all 50 videos checked; no variable frame rate and no undecodable file was found among the 200 sample videos or in the 67 and 20 video subsets (0 decode failures).
- An earlier version of `bench decode` wrote its raw rows to the same file name for different manifests; the file `decode_threads1.csv` is the FF++ one (regenerated), the proxy run is `decode_threads1_proxy224.csv`.
- One test-writing slip: `bench audio` selects Celeb-DF files with `rglob`, so it takes mostly Celeb-synthesis files; the `ffprobe` check above covers all three folders.

## 7. Questions for the main agent

1. Do you want me (or the main agent) to run the full FF++ c23 crop job now? Command: `--manifest data/_manifests/ffpp.manifest.csv --out data/_crops/ffpp_c23_yunet_n16 --workers 16`, about 13 minutes projected, resumable, files that are still downloading are skipped (rerun later).
2. Frames per video N, crop enlarge factor (1.3 is the FF++ paper value, R-001) and JPEG quality are parameters with defaults only; who decides them, and does an iteration test them?
3. Frames with no detection: record only (as now), or fall back to the previous detection or a centre crop? Not needed on the 200-video sample.
4. Which face to take when several are present (largest now)?
5. Please log the three model files (sources and hashes above) in DATA_LOG.md and verify their licences before any is used in the paper.

---

# Addendum 2 (2026-10-24, evening): YuNet against UltraFace and SCRFD, licences and a 300 + 300 frame comparison

D-034 (YuNet, 640 px, defaults) is unchanged; no pipeline default was touched and nothing was re-cropped. New code: `src/dfd/prep/detcmp.py`. Raw output under `data/_prep_bench/` (git-ignored).

Caveat for every number here: downloads and extraction were still running; Windows reported 14 to 20 percent CPU busy before each run (the "system CPU load before run" line in the result files). Speed is single thread, in process, on exactly the listed frames, mean of 3 timings per frame, the three detectors interleaved per frame.

## A. Licences (read from the repositories' own files on 2026-10-24; copies in `data/_prep_bench/licences/`)

The only network use was downloading these small text files (LICENSE and README files) from raw.githubusercontent.com.

| Detector | What I read | Result |
|---|---|---|
| YuNet 2023mar | opencv_zoo `models/face_detection_yunet/README.md`, section License: "All files in this directory are licensed under MIT License" (this folder holds the ONNX file); its `LICENSE` begins "MIT License, Copyright (c) 2020 Shiqi Yu". The repository root `LICENSE` of opencv_zoo is Apache 2.0; the folder licence is the more specific one. The README says the model source is ShiqiYu/libfacedetection.train (that repository's licence and the licence of its training data, WIDER Face, not read: NR) | MIT, commercial use permitted. Re-check confirmed |
| UltraFace RFB-320 | Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB `LICENSE`: "MIT License, Copyright (c) 2019 linzai", no clause about the weights; the README has no licence statement and says the models are trained on WIDER Face with the cleaned labels of RetinaFace (licence of those training data: NR, not read) | MIT, commercial use permitted for the repository files including the ONNX; any restriction inherited from the training data is NR |
| SCRFD 500M (`det_500m.onnx` in `buffalo_sc.zip`) | InsightFace `README.md`, License section: "The training data containing the annotation (and the models trained with these data) are available for non-commercial research purposes only." and "Both manual-downloading models from our github repo and auto-downloading models with our python-library follow the above license policy(which is for non-commercial research purposes only)."; `model_zoo/README.md`: "ALL models are available for non-commercial research purposes only."; `python-package/README.md`: "The library code is released under the MIT License ... The pretrained models provided with this library are for non-commercial research only" | Research-only (non-commercial) for the weights; MIT only for the library code. I did not find a licence line specific to the `buffalo_sc` pack, so the general lines above apply |

Nothing marked NR was guessed. A research-only detector used only for cropping would still have to be declared as such in the paper if it were chosen.

## B. Frame lists (seeded, repeatable)

Seed 20261024 (the seed value of the split files). Frames only: labels and method names were used only to balance the draw, never in any result.

- FF++: `data/_splits/ffpp-037e2976fade.csv`, rows with `split == val` (700 videos: 140 real, 140 per method). Draw: all 140 real videos (the split has fewer than 150) plus 40 videos of each of Deepfakes, Face2Face, FaceSwap, NeuralTextures, 300 distinct videos, one uniformly random frame each. List: `data/_prep_bench/detcmp_frames_ffpp.csv`. All files were complete (0 skipped).
- Celeb-DF v2: `data/_splits/celebdf_v2-82a3a70594e9.csv`, rows with `split == dev_x` (1,728 videos, 15 identity groups, 146 real and 1,582 fake; this is the file the coordinator confirmed as final). The earlier `celebdf_v2-b2069a4a29aa.csv` has only test_x and was not used. Draw: all 146 real videos plus 154 fake videos (the manifest has one fake method label, `celebdf_synthesis`), 300 distinct videos, one random frame each. List: `data/_prep_bench/detcmp_frames_celebdf.csv`. 0 skipped.

Commands:

```
PYTHONPATH=src .venv/Scripts/python.exe -m dfd.prep.detcmp frames --manifest data/_manifests/ffpp.manifest.csv --split-file data/_splits/ffpp-037e2976fade.csv --split val --n 300 --out data/_prep_bench/detcmp_frames_ffpp.csv
PYTHONPATH=src .venv/Scripts/python.exe -m dfd.prep.detcmp frames --manifest data/_manifests/celebdf_v2.manifest.csv --split-file data/_splits/celebdf_v2-82a3a70594e9.csv --split dev_x --n 300 --out data/_prep_bench/detcmp_frames_celebdf.csv
PYTHONPATH=src .venv/Scripts/python.exe -m dfd.prep.detcmp run --frames data/_prep_bench/detcmp_frames_ffpp.csv --tag ffpp
PYTHONPATH=src .venv/Scripts/python.exe -m dfd.prep.detcmp run --frames data/_prep_bench/detcmp_frames_celebdf.csv --tag celebdf
```

Outputs: `detcmp_ffpp_result.txt`, `detcmp_celebdf_result.txt` (full printed results), `detcmp_raw_<tag>.csv` (largest box and time per frame and detector), `detcmp_disagree_<tag>.jpg` (images). Detector settings are the defaults of `face.py` (YuNet input resized to at most 640 px, score threshold 0.6; UltraFace 320x240, 0.7; SCRFD 640x640, 0.5).

## C. Results (300 frames per dataset)

Per detector:

| Detector | Dataset | frames with at least one face | ms per frame (mean, median, p90) | frames with more than one box |
|---|---|---|---|---|
| YuNet | FF++ val | 300 of 300 (100.0%) | 12.1, 12.3, 13.8 | 38 (12.7%) |
| UltraFace | FF++ val | 300 of 300 (100.0%) | 4.8, 4.7, 5.2 | 24 (8.0%) |
| SCRFD | FF++ val | 300 of 300 (100.0%) | 29.7, 29.5, 31.6 | 30 (10.0%) |
| YuNet | Celeb-DF dev_x | 300 of 300 (100.0%) | 12.4, 12.1, 14.5 | 6 (2.0%) |
| UltraFace | Celeb-DF dev_x | 300 of 300 (100.0%) | 4.8, 4.7, 5.2 | 4 (1.3%) |
| SCRFD | Celeb-DF dev_x | 300 of 300 (100.0%) | 30.9, 30.8, 32.2 | 6 (2.0%) |

Pairwise on the largest box of each detector. Every frame had a box from every detector, so "both detected" is 300 for every pair, the disagreement on whether a face exists is 0 of 300 for every pair and dataset, and no frame had no box from any detector. "A < half of B" is the share of frames where the largest box of A has less than half the area of the largest box of B.

| Pair (A vs B) | Dataset | mean IoU | IoU at least 0.5 | disagree on existence | A < half of B | B < half of A | median area B / A | median centre shift (fraction of A box side) |
|---|---|---|---|---|---|---|---|---|
| YuNet vs UltraFace | FF++ val | 0.897 | 300 of 300 | 0 | 0 | 0 | 0.974 | 0.022 |
| YuNet vs SCRFD | FF++ val | 0.923 | 300 of 300 | 0 | 0 | 0 | 1.018 | 0.017 |
| UltraFace vs SCRFD | FF++ val | 0.908 | 300 of 300 | 0 | 0 | 0 | 1.048 | 0.019 |
| YuNet vs UltraFace | Celeb-DF dev_x | 0.887 | 300 of 300 | 0 | 0 | 0 | 0.993 | 0.025 |
| YuNet vs SCRFD | Celeb-DF dev_x | 0.921 | 300 of 300 | 0 | 0 | 0 | 1.030 | 0.017 |
| UltraFace vs SCRFD | Celeb-DF dev_x | 0.902 | 300 of 300 | 0 | 0 | 0 | 1.035 | 0.020 |

Area ratio spread (p10 to p90) across pairs and datasets: 0.89 to 1.13; centre shift p90 at most 0.043 of the box side.

Images: there were no frames with an existence disagreement or an IoU below 0.5 in either set, so there are no true disagreement frames. `detcmp_disagree_ffpp.jpg` and `detcmp_disagree_celebdf.jpg` show instead the 8 frames per dataset with the lowest agreement (lowest pairwise IoU 0.75 to 0.81 on FF++, 0.70 to 0.79 on Celeb-DF). Green box YuNet, blue UltraFace, red SCRFD, thick line the largest box of each detector. What I see: all three detectors box the same face in all 16 tiles; the boxes differ only in extent by a few percent, most visibly in vertical extent (the UltraFace and SCRFD boxes sit a little higher or lower than YuNet's, taking in more or less forehead or chin). In the two-person tile (a two-anchor studio in FF++) all three took the same, larger face as the largest. Quick look, not a measurement.

## D. Plain statement for the main agent

On these 600 frames the three detectors are not distinguishable by what they find: every frame had a face for all three, they never disagreed on existence, and the largest boxes overlap with a mean IoU of 0.89 to 0.92 (every frame at least 0.5), with median box area within 5 percent and centres within about 2 percent of the box side. A 1.3 times enlarged square around these boxes would differ between detectors by a small shift and scale of a few percent. The measured difference that is large is speed (UltraFace 4.8 ms, YuNet 12 ms, SCRFD 30 ms per frame), which matters little for the pipeline because decoding is about three quarters of the time per video (section 3.3 above). The earlier stress test (section 3.2 above) is where the detectors differed (rotated, tiny and face-free images); these frames contain no such cases, so this comparison cannot show a difference there, and with 300 frames per dataset a 100 percent rate only says the true miss rate on such frames is probably below about 1 percent (rule of three).

The licence is the clearest difference: YuNet and UltraFace are MIT; the SCRFD weights are research-only. I did not measure whether a few percent of box shift changes the accuracy of any model, so I cannot say it does not matter downstream. On this evidence I see no geometric reason to re-crop or to change D-034; if the user is to be asked anything, it is only whether to keep YuNet, which is the lowest-risk option on licence and has no measured disadvantage here.

## E. Note on JPEG quality

JPEG quality 95 for the stored crops (13,740 bytes per crop, PSNR 45.5 dB against the uncompressed crop, section 3.6 above) is a candidate preprocessing iteration for later (for example quality 100 or lossless storage against 95 or 90); it is not changed here.
