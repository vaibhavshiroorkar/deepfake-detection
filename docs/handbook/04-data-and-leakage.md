# Data and leakage

## Learning goals

After this chapter, you should be able to validate a manifest row, select the
right label for each branch, audit identity overlap, and explain method
holdout.

## Required background

You should know CSV rows, Python data classes, sets, and train, validation, and
test partitions. Read the [data card](../data-card.md) first.

## Manifest contract

`ClipRecord` is the normalized unit of data. Every field has one role:

| Field | Type | Meaning |
|---|---|---|
| `clip_id` | `str` | Nonblank clip identifier. |
| `dataset` | `str` | Dataset name attached during loading. |
| `video_path` | `Path` | Media location. Raw media stays outside Git. |
| `manipulation_type` | `str` | One of the four cue combinations below. |
| `method` | `str` | Real or generation method family. |
| `source` | `str` | Identity whose source performance defines the main split. |
| `targets` | `tuple[str, ...]` | Optional target identities from `target1` and `target2`. |
| `clip_fake` | `bool` | True when either video or audio is fake. |
| `video_fake` | `bool` | True when the visual stream is fake. |
| `audio_fake` | `bool` | True when the audio stream is fake. |
| `race` | `str` | Supplied audit group, default `unknown`. |
| `gender` | `str` | Supplied audit group, default `unknown`. |
| `leading_silence_sec` | `float` | Known nonnegative leading silence duration. |

The four valid `manipulation_type` values are `RealVideo-RealAudio`,
`FakeVideo-RealAudio`, `RealVideo-FakeAudio`, and `FakeVideo-FakeAudio`.
`ClipRecord` derives and validates the three Boolean labels from this value.

`load_manifest()` reads UTF-8 CSV, groups rows by media path, and checks clip
identity consistency. It quarantines a path when one clip ID maps to several
paths or duplicate rows disagree on manipulation type or method. It returns
usable records and the quarantined paths separately.

## Cue-specific labels

The visual branch uses `video_fake`. An authentic face paired with synthetic
speech should not teach the visual branch that authentic pixels are fake. The
audio branch uses `audio_fake` for the same reason.

Fusion uses `clip_fake` because its task is the whole clip decision. The sync
branch learns correspondence from authentic clips and generated alignment
tasks. A globally fake clip can still have good mouth-audio alignment, so the
global label is not a clean sync target.

For a batch of two audio records, labels have shape `[2]`. Cached audio values
have shape `[2, 64000]` under the current four-second view. A binary branch
returns logits shaped `[2]`.

## Source-disjoint splits

`build_source_split()` collects unique source identities. It checks that each
source has consistent race and gender metadata. It then allocates sources, not
rows, to train, validation, and test with ratios 70/15/15. At least three
sources are required so each partition can contain one.

### Three-person worked example

Assume each person has one real clip and one fake clip:

| Source | Rows | Valid assignment |
|---|---|---|
| Person A | `A-real`, `A-fake` | train |
| Person B | `B-real`, `B-fake` | validation |
| Person C | `C-real`, `C-fake` | test |

The smallest valid split has one source in each partition. The exact person
assigned to each partition depends on the fixed seed. Both rows for a person
move together.

A row-level random split could put `A-real` in train and `A-fake` in test. The
model could recognize Person A, the room, camera, or compression history. Test
performance would then mix deepfake detection with identity recall. More rows
do not fix this dependence.

`audit_split()` reports source overlap, overlap across source plus target
identities, and method counts. `identity_strict_subset()` keeps a row only when
all known targets belong to the same partition as its source. This subset tests
a stricter identity condition but can reduce data and method coverage.

`split_hash()` sorts partition name, clip ID, source, and targets. It hashes the
JSON bytes with SHA-256. Reordering rows preserves the hash. Moving a record to
another partition changes it.

## Leakage and shortcuts

Leakage occurs when training or model selection receives information that
belongs only to evaluation. Shortcuts are features correlated with the label
but unrelated to the intended evidence.

- Identity leakage: one source occurs in train and test.
- Target leakage: a target identity crosses partitions even if sources do not.
- Method leakage: the claimed unseen generator occurs during training or
  validation.
- Codec leakage: encoding settings identify the label or method.
- Silence leakage: leading silence or padding identifies synthetic audio.
- Duplicate leakage: the same media appears under several rows or paths.
- Selection leakage: test metrics guide architecture, thresholds, or exclusions.

Source separation solves the first item. The audit and stress protocols expose
some others. No single split removes every shortcut.

## Method holdout

`build_method_holdout_protocol()` receives an existing train, validation, and
test split plus one or more held-out methods. It removes those methods from
train and validation. Test keeps all real clips and fake clips made by the
held-out methods.

This design permits threshold and model selection without seeing the claimed
unseen fake method. It also keeps real test examples for a valid binary
comparison. Holding out a method only from training is insufficient because
validation tuning can still adapt to it.

## Project code path

1. [`load_manifest()`](../../src/deepfake_detection/data/manifest.py) parses,
   validates, deduplicates, and quarantines.
2. [`build_source_split()`](../../src/deepfake_detection/data/protocols.py)
   groups source identities into partitions.
3. [`audit_split()`](../../src/deepfake_detection/data/protocols.py) reports
   overlaps and method counts.
4. [`identity_strict_subset()`](../../src/deepfake_detection/data/protocols.py)
   filters cross-partition target identities.
5. [`split_hash()`](../../src/deepfake_detection/data/protocols.py) fingerprints
   the assignment.
6. [`build_method_holdout_protocol()`](../../src/deepfake_detection/data/protocols.py)
   creates an unseen-method evaluation from the frozen split.

### Theory and equations

Let `g(i)` be the assigned partition for source identity `i`. Source
disjointness requires:

```text
for every source i: g(i) is exactly one of {train, val, test}
```

For partition source sets `S_train`, `S_val`, and `S_test`:

```text
S_train intersect S_val = empty
S_train intersect S_test = empty
S_val intersect S_test = empty
```

These equations apply to source identities. The full identity audit also adds
all targets and may find overlap. That is why the strict subset is reported
separately.

### Design trade-offs

- Source grouping reduces leakage, but exact row ratios can differ from 70/15/15.
- Demographic strata guide allocation, but small strata cannot fill every split.
- Quarantine preserves uncertainty, but reduces usable coverage.
- Identity-strict filtering strengthens separation, but can remove methods.
- Method holdout tests novelty, but one method may not represent future attacks.

## Failure cases

- Blank identifiers, paths, methods, or sources fail validation.
- Negative or nonfinite leading silence fails validation.
- Cue labels that conflict with the manipulation type fail validation.
- Fewer than three source identities cannot form the required split.
- Conflicting demographics for one source stop split construction.
- A missing split partition or empty held-out set stops method holdout.
- Row-level random splitting silently creates dependent evaluation examples.

### Supporting tests

[`test_manifest.py`](../../tests/test_manifest.py) covers parsing, cue labels,
quarantine, and invalid fields. [`test_protocols.py`](../../tests/test_protocols.py)
covers source grouping, audit, strict filtering, hashes, and method holdout.
Run:

```powershell
uv run pytest tests\test_manifest.py tests\test_protocols.py -v
```

## Exercises

1. Build four fixture rows, one for each manipulation type. State every label.
2. Add two clips for one source and verify they stay in one partition.
3. Add a cross-partition target and inspect the identity-strict subset.
4. Reorder split rows and confirm the hash stays fixed. Move one row and confirm
   it changes.
5. Explain why balancing test rows would change the reported deployment mix.

## Viva questions

1. Why does the visual branch not use `clip_fake`?
2. What does source-disjoint mean, and what does it leave unresolved?
3. Why are target identities treated in a separate stress subset?
4. Why must validation exclude a held-out method?
5. What does the split hash prove, and what does it not prove?

## Sources

- [FakeAVCeleb dataset paper](https://arxiv.org/abs/2108.05080)
- [FakeAVCeleb source repository](https://github.com/DASH-Lab/FakeAVCeleb)
- [FaceForensics++ paper and benchmark](https://openaccess.thecvf.com/content_ICCV_2019/html/Rossler_FaceForensics_Learning_to_Detect_Manipulated_Facial_Images_ICCV_2019_paper.html)
- [Shortcut learning paper](https://www.nature.com/articles/s42256-020-00257-z)
- [scikit-learn grouped cross-validation documentation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data)
