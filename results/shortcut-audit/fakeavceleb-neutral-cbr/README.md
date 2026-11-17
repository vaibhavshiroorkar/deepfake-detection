# P-001 amendment A18: H1 again with a fixed video bitrate

**Exploratory.** Fixed in [amendment A18](../../preregistration/P-001-shortcut-neutral-av-study.md) before this result existed. H1's preregistered verdict stays "not met".

Neutralizer v4 with one change: video encoded at a constant 160 kb/s instead of constant quality (CRF 23). All of FakeAVCeleb (`runs/p001-a18.sh`), audited with the unchanged `scripts/shortcut_audit.py`. The same 1,260 clips were refused as too short, as under v4.

| Label | v4 (constant quality) | A18 (160 kb/s constant) | Strongest single cues under A18 |
|---|---|---|---|
| Clip | 0.632 | 0.624 | video bitrate 0.582, trailing silence 0.569, loudness 0.568 |
| Video | 0.617 | 0.555 | video bitrate 0.580, overall bitrate 0.527 |
| Audio | 0.793 | 0.830 | video bitrate 0.648, trailing silence 0.640, loudness 0.603 |

20,284 clips, none failed. The 600-clip check sample (`../fakeavceleb-neutral-cbr-check/`) gave 0.594 against 0.598 for v4.

## Reading, by the rule fixed in A18

- **A fixed bitrate does not bring the clip-label ceiling to 0.60** (0.624). Something besides the bitrate setting remains.
- **The bitrate cue shrinks but does not vanish.** The encoder overshoots its 160 kb/s target by a few percent on busier content, and that overshoot still follows the label. It now tracks fake *audio* most (0.648), so the audio-label ceiling rises while the video-label ceiling falls.
- **The other residuals are the same as under v4:** trailing silence and loudness after AAC encoding.
