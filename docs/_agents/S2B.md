# S2-B report: Stage 1 data tooling (manifests, group-aware splits, leakage check)

Written 2026-10-24. Nothing was committed or pushed. pyproject.toml and uv.lock untouched. Only the standard library and numpy-free code was needed at run time (no numpy used).

## 1. Files built

Code (all under src/dfd/data/):
- manifest.py: schema (12 columns), Record, writer, reader, validation (label/method rule, relative forward-slash paths, namespaced ids, unique sample_id).
- builders/__init__.py: LayoutNotVerified exception.
- builders/ffpp.py: FF++ c23 builder, union-find components, official split loader (fetches train/val/test.json once), partial-data tolerant, report found vs documented.
- builders/asvspoof2019_la.py: CM-protocol builder (group = speaker, official train/dev/eval, extras counted and excluded, speaker-disjointness report).
- builders/forensynths.py: ProGAN val and test, CNN synthesis subset, diffusion test (group rules documented in the module docstring).
- builders/celebdf.py: Celeb-DF v2 builder (real, run on the real files); Celeb-DF++ stub.
- builders/stubs.py: documented stubs for dfd, in_the_wild, lavdf, fakeavceleb, wilddeepfake, plus tested name parsers (DFD names, VoxCeleb2 ids). Deviation from the brief: one module for these five stubs, not one each.
- build_manifests.py: CLI, writes data/_manifests/<key>.manifest.csv and <key>.report.json.
- splits.py: modes official, group, fixed, auto; seeded; writes data/_splits/<split_id>.csv and .json sidecar; split id = name + first 12 hex of the CSV SHA-256.
- dedup.py and hashing.py: D-026 content de-duplication of test_x against train, val and test_in, and the SHA-256 cache (tests in tests/data/test_data_dedup.py).
- leakage.py and check_leakage.py: checks and CLI; identity_set and identity_overlap helpers (LAV-DF vs FakeAVCeleb).

Tests (tests/data/): data_test_helpers.py (synthetic fixtures), test_data_manifest.py, test_data_builders.py, test_data_splits.py, test_data_leakage.py.

## 2. How to run

The package is not installed, so set the path (pytest does it through pyproject):

    PYTHONPATH=src .venv/Scripts/python.exe -m dfd.data.build_manifests --dataset ffpp      (or all)
    PYTHONPATH=src .venv/Scripts/python.exe -m dfd.data.splits --manifest data/_manifests/ffpp.manifest.csv --name ffpp --mode official
    PYTHONPATH=src .venv/Scripts/python.exe -m dfd.data.splits --manifest M --name N --mode group --seed 1 --fractions train=0.7,val=0.15,test_in=0.15
    PYTHONPATH=src .venv/Scripts/python.exe -m dfd.data.check_leakage --manifest M1 M2 --split S1 S2 [--hash-files] [--strict-duplicates] [--merge-family a,b]
    .venv/Scripts/python.exe -m pytest tests/data -q -rs

Exit codes: build_manifests 0 ok, 2 stub or failure; check_leakage 0 clean, 1 leakage, 2 input error.

## 3. Measured results (commands as above, run on the real files, 2026-10-24)

Tests: `pytest tests/data -q -rs` gave 42 passed, 1 skipped before section 3b was added; the current result is 48 passed, 1 skipped (section 3b).

A7 status:
- ASVspoof 2019 LA: RAN and passed. Manifest 121,461 rows = train 25,380 + dev 24,844 + eval 71,237. Bona fide / spoof: train 2,580 / 22,800, dev 2,548 / 22,296, eval 7,355 / 63,882. Every listed file exists. Extra flac not in the CM protocols: dev 142, eval 696, train 0 (excluded). Speakers: 20 train, 20 dev, 67 eval, 107 groups, 0 speakers in more than one partition (documented speaker-disjointness confirmed). Note: DATA_LOG.md says 20 / 10 / 48 speakers (from the paper); the real protocol files give 20 / 20 / 67.
- ProGAN val and test: RAN and passed. Each 8,000 images, 4,000 real and 4,000 fake, 20 classes, 200 real and 200 fake per class.
- FF++ c23: SKIPPED by the real-data test (data incomplete). At the last run: 628 originals found (631 at the last test run) of 1,000; Deepfakes 913, Face2Face 373, FaceSwap 404, NeuralTextures 1,000 of 1,000 each. The manifest holds 3,318 rows, 500 groups, 0 components spanning two official splits, 0 components that are not exactly one official pair, 0 ids without an official split. The complete-data logic is exercised by a synthetic tree built from the real official pair lists (1,000 originals, 4,000 manipulated, 500 groups, passes). Rerun `pytest tests/data -q -rs` once the download is complete; the real test then runs.
- Official FF++ split files fetched (3 small files) into data/_manifests/ffpp_official_splits/: 360 / 70 / 70 pairs.

Other manifests built: Celeb-DF v2 complete (Celeb-real 590, YouTube-real 300, Celeb-synthesis 5,639; test list 518 lines, all present, 518 rows marked official test; 59 identity groups). CNN synthesis subset 56,003 images (biggan 2,000+2,000; cyclegan 1,321+1,321; deepfake 2,707 real + 2,698 fake; gaugan 5,000+5,000; stargan 1,999+1,999; stylegan 5,991+5,991; stylegan2 7,988+7,988; whichfaceisreal 1,000+1,000), 18 groups. Diffusion test 10,000 images in 10 folders of 1,000: eight fake-only folders (dalle, glide_100_10, glide_100_27, glide_50_27, guided, ldm_100, ldm_200, ldm_200_cfg) and two real-only folders (imagenet, laion); the real-to-generator pairing is not in the folder names and is not assumed.

Splits written (data/_splits/): ffpp (official), asvspoof2019_la (official; train / dev as val / eval as test_in), progan_val (group, seed 20261024, train 14 classes 5,600 / val 6 classes 2,400), progan_test (fixed test_in), cnn_subset and diffusion_test and celebdf_v2 (fixed test_x). Leakage check without hashing: PASS for ffpp, asvspoof2019_la, and ffpp together with celebdf_v2.

Leakage check with `--hash-files` over progan_val, progan_test, cnn_subset and diffusion_test (82,003 files, 3 min 45 s): FAIL, which is a real finding, not a bug. 800 content hashes sit in different splits, all label 0 (real). 400 hashes are shared between progan_val (train/val) and the CNN subset (test_x), and 400 between progan_test and the CNN subset; they are the LSUN cat real images that also appear as real images in the cnn_synth_subset stylegan and stylegan2 `cat` folders (file names such as 00091.png). No hash is shared between progan_val and progan_test. In addition 3,604 hashes are duplicated inside the CNN subset (stylegan vs stylegan2 real images, 3,594+ hashes, and a few in cyclegan and whichfaceisreal); these are only warnings. Consequence: the stylegan/stylegan2 real images are not independent of the ProGAN training reals. Not checked: file hashes for ASVspoof, FF++ and Celeb-DF (slow, not run).

## 3b. D-026 content de-duplication (added after the coordinator's reply)

Implemented in src/dfd/data/dedup.py (CLI `python -m dfd.data.dedup`) and src/dfd/data/hashing.py (SHA-256 with an optional cache, data/_manifests/content_hash_cache.json; `check_leakage --hash-files` uses the same cache and has a `--cache` option). Rule: a test_x sample is dropped when its file content equals a train, val or test_in sample (test_in is included, because the progan_test reals are also in the CNN subset and the check would otherwise still fail). Duplicates inside one split stay warnings. Each split file that loses rows is rewritten under a new content-derived id (sidecar has `dedup_of`, `dropped`, `exclusions_file`); the exclusions file is data/_splits/images-exclusions.csv with a counts file images-exclusions.json.

Command: `PYTHONPATH=src .venv/Scripts/python.exe -m dfd.data.dedup --manifest <progan_val, progan_test, cnn_subset, diffusion_test manifests> --split <their split files> --name images`

Measured result (82,003 files hashed on the first run, about 4 minutes; later runs use the cache):
- 1,200 samples excluded, 800 distinct hashes, all in forensynths_cnn_subset: stylegan/cat 400, stylegan2/cat 400, stylegan2/horse 400. (The coordinator's count was cat only; stylegan2/horse also has 400 reals equal to progan reals. Each of the 800 hashes equals a progan_val or progan_test real; the cat hashes are shared by stylegan and stylegan2.)
- New cnn_subset split cnn_subset-29c6b1f0633b: 54,803 rows kept of 56,003. The old cnn_subset split files were deleted from data/_splits.
- `check_leakage --hash-files` on progan_val + progan_test + cnn_subset + diffusion_test with the new splits: PASS, rc 0, 80,803 files hashed; warnings: 1,200 manifest rows have no split (the excluded ones) and 3,604 hashes duplicated inside one split (CNN subset: stylegan vs stylegan2 reals, and a few in cyclegan and whichfaceisreal).
- Tests added in tests/data/test_data_dedup.py (synthetic fixture with duplicated files, CLI, cache, and the real-data check that runs on data/_splits). `pytest tests/data -q -rs`: 48 passed, 1 skipped (FF++ A7, incomplete: at that run 1,000 originals, Deepfakes 913, Face2Face 1,000, FaceSwap 636, NeuralTextures 1,000).

Decisions D-026 applied: ProGAN val split 14 / 6 classes accepted; Celeb-DF fake group by first identity accepted.

## 4. Skipped or not verified

- FF++ real-data A7: skipped, data not complete.
- Stubs: celebdf_pp, dfd, in_the_wild, lavdf, fakeavceleb, wilddeepfake raise LayoutNotVerified with the exact assumption; never run on real files.
- Celeb-DF v2 builder: name patterns and counts verified on the real folders; the meaning of the 0/1 token in the test list is not used.
- Video decodability, ffprobe checks and SHA-256 manifests of the data are outside this task.

## 5. Ambiguities and the choice made

1. Celeb-DF fake group: first id (target) names the group, both ids in notes (`ids=id0|id16`). The strict component over the identity graph would give 3 components of 38, 10 and 10 identities (measured), too coarse for the bootstrap. The first id being the target is an assumption.
2. The official Celeb-DF test list covers 54 of the 59 identity groups, so it is not identity-disjoint from the rest; irrelevant because Celeb-DF is test_x only.
3. ProGAN group = LSUN class, namespaced by dataset, so class disjointness holds inside one set only. A train/val split of progan_val by class gives class-disjoint validation, while progan_test (test_in) contains the same 20 class names with different images. Strictly this is weaker than "no source on both sides".
4. FF++ groups are built from the files present (as specified), not from the official pairs, so with partial data a group can be smaller than the final one. The official-pair consistency is only reported.
5. Duplicate content: across different splits fails; inside one split a warning unless `--strict-duplicates`. The brief said "any".
6. Family rule: a family in train or val and also in test_x fails; test_in plus test_x for one family is allowed. Unassigned manifest rows are a warning, split rows without a manifest row are a failure.
7. official_split names: train, val, test (FF++, Celeb-DF), train, dev, eval (ASVspoof); splits.py maps dev to val, eval and test to test_in. Fraction-based splits are my addition for datasets without an official split.
8. Sample ids: `ffpp:original:008`, `ffpp:Deepfakes:008_990`, `asvspoof2019_la:LA_T_1138215`, `<dataset>:<path inside the dataset folder>` for images and Celeb-DF.
9. In the Celeb-DF++ and FakeAVCeleb stubs the family names are `celebdf`, `lavdf`, `fakeavceleb`; `POSSIBLE_SHARED_FAMILIES` in leakage.py makes the check compare VoxCeleb2 ids (idNNNNN) between train/val and test_x of the two and fail on overlap, warn when no ids can be read.

## 6. Questions for the main agent

Questions 1 to 4 of the first version (duplicate reals, ProGAN class split, ASVspoof speaker counts, Celeb-DF grouping) were answered by D-026 and implemented (section 3b). Open: the FF++ rerun.

5. Rerun `python -m dfd.data.build_manifests --dataset ffpp`, the FF++ split and the tests when FF++ is complete (the existing FF++ split file is from partial data and its hash will change).
