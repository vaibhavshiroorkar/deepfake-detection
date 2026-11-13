# Neutralizer v4 check

The 600-clip FakeAVCeleb sample (150 per category, `sample-manifest.csv`, pandas `random_state=5`) used to check neutralizer v4 before the full rerun. These clips are a check set only. They are not used to choose anything else.

- `neutralize-summary.json`: 29 clips refused as too short (16 FakeVideo-FakeAudio, 13 RealVideo-FakeAudio, 0 real).
- `cuts.csv`: every cut.
- `summary.json` and `clips.csv`: the audit. Ceilings: 0.598 (clip), 0.489 (video), 0.702 (audio).

Cited by P-001 amendment A4.
