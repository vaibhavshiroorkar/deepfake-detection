# HiFi-AVDF, raw files

The shortcut audit on all 3,810 raw HiFi-AVDF clips (dataset revision c5a6aff6): 1,905 real videos, each paired with a generated copy from Kling 2.5, PixVerse V5, Seedance 1.0, Sora 2, Veo 3.1 or Wan 2.5. Folds are grouped by real-generated pair.

The ceiling is 0.9985 (clip label). Each generator writes its files differently:

- **Frame size.** Real clips are 1280x720. Seedance writes 1248x704, Sora 2 1280x704, Kling 1920x1072, and PixVerse and Wan 1920x1080.
- **Clip length.** Sora 2 clips are exactly 10.0 s and Veo 3.1 clips exactly 8.0 s. Real clips have a median of 5.7 s.
- **Bitrate.** Seedance runs at 6.4 Mb/s; real clips at 1.5 Mb/s.
- **Leading silence.** Kling clips start with about 68 ms of it, Seedance clips about 25 ms. Real clips have none.

Under P-001 amendments A5 and A7, this dataset is neutralized with one frame size (1280x720) before any model scores it.
