# H1 attempt 2: neutralizer v4 on all of FakeAVCeleb

Preregistered test H1 (P-001): the shortcut ceiling on the clip label must be at most 0.60.

**Result: not met.** The ceiling is 0.632 (clip), 0.617 (video) and 0.793 (audio), on 20,284 clips (19,784 fake, 500 real).

- The strongest remaining cue is video bitrate (0.573 clip, 0.636 video). Every clip is encoded at the same constant quality (CRF 23), so bitrate follows image content: smoother face swaps compress into fewer bits. That makes it a property of the manipulated pixels, not of the generator's file pipeline. But H1 was defined on all cues, so it fails as written.
- The audio label keeps a ceiling of 0.793, mainly from millisecond differences at clip ends (trailing silence 0.640) and loudness after AAC (0.603).
- 1,260 clips were refused as too short: 1,210 FakeVideo-FakeAudio, 50 RealVideo-FakeAudio, and no real clip (`neutralize-summary.json`). This selection effect is reported with every result on these copies.

Files: `summary.json` and `clips.csv` (the audit), `neutralize-summary.json` and `cuts.csv` (the neutralization). Attempt 1 is in `../fakeavceleb-neutral-v1/`.
