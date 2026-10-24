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

Tests: `pytest tests/data -q -rs` now gives 49 passed, 0 skipped (all A7 checks ran on the real files). Earlier states: 42 passed, 1 skipped, then 48 passed, 1 skipped (FF++ incomplete).

A7 status:
- ASVspoof 2019 LA: RAN and passed. Manifest 121,461 rows = train 25,380 + dev 24,844 + eval 71,237. Bona fide / spoof: train 2,580 / 22,800, dev 2,548 / 22,296, eval 7,355 / 63,882. Every listed file exists. Extra flac not in the CM protocols: dev 142, eval 696, train 0 (excluded). Speakers: 20 train, 20 dev, 67 eval, 107 groups, 0 speakers in more than one partition (documented speaker-disjointness confirmed). Note: DATA_LOG.md says 20 / 10 / 48 speakers (from the paper); the real protocol files give 20 / 20 / 67.
- ProGAN val and test: RAN and passed. Each 8,000 images, 4,000 real and 4,000 fake, 20 classes, 200 real and 200 fake per class.
- FF++ c23, FINAL (rerun after the download completed; the real-data test RAN, not skipped, and passed): `build_manifests --dataset ffpp` wrote 5,000 rows = 1,000 originals (label 0) + 1,000 each of Deepfakes, Face2Face, FaceSwap, NeuralTextures (4,000 fakes). Groups (connected components): 500, every one with exactly 2 original videos and 10 samples (2 originals + 4 methods x 2 directions), so the size distribution is 500 groups of 10, and each component is exactly one official pair. Official split: train 360 pairs (3,600 rows: 720 real, 2,880 fake), val 70 pairs (700 rows: 140 / 560), test 70 pairs (700 rows: 140 / 560). 0 components span two official splits, 0 ids lack an official split, 0 files ignored. Split written as ffpp-037e2976fade (train 3,600, val 700, test_in 700); the stale partial split was deleted and the manifest overwritten. Leakage check without hashing: PASS (500 groups). Superseded earlier state: partial data (628 originals) was skipped by the test then.
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

Questions 1 to 4 of the first version (duplicate reals, ProGAN class split, ASVspoof speaker counts, Celeb-DF grouping) were answered by D-026 and implemented (section 3b). None open. The FF++ rerun is done (see section 3). Old note kept: rerun `python -m dfd.data.build_manifests --dataset ffpp`, the FF++ split and the tests when FF++ is complete (the existing FF++ split file is from partial data and its hash will change).

## 7. D-035 task: split dev_x, slices, In-the-Wild and DFD builders (2026-10-24)

Implemented (src/dfd/data): `dev_x` in splits.py (SPLIT_NAMES, `assign_dev_slice`, `draw_dev_units`, CLI mode `dev_slice` with `--dev-k`, `--dev-unit group|generator`, `--dev-identity-only`); leakage.py (a group in both dev_x and test_x fails as GROUP_OVERLAP plus GROUP_OVERLAP_DEV_X_TEST_X; equal content across dev_x and test_x fails like any cross-split content; the family rule is unchanged, so dev_x plus test_x of one family passes and dev_x never counts against test_x); dedup.py (two steps, see below); builders/in_the_wild.py and builders/dfd.py (real builders; the stubs for these two were removed, `parse_dfd_name` moved to dfd.py). Slice rule exactly as D-035: sort the group names, `numpy.random.default_rng(20261024).permutation`, first k, all samples of a drawn unit to dev_x, the rest test_x (tested against a direct numpy computation). The scope of "score file reader accepts dev_x" in A9 is not in my folders; only the data side is done and tested.

Tests: `pytest tests/data -q -rs` gives 64 passed, 0 skipped (new files test_data_devx.py and test_data_dfd_itw.py, plus dedup additions).

Measured counts (manifests rebuilt with `build_manifests`, splits with `splits`, then `dedup`):
- Celeb-DF v2: split file `celebdf_v2-82a3a70594e9` (dev_x slice written; hashing found no duplicate content, so the file is unchanged by dedup). k=15 of the 59 identity groups: dev_x = id0, id1, id10, id12, id16, id20, id22, id26, id31, id41, id42, id47, id48, id50, id53; dev_x 1,728 videos (146 real, 1,582 fake), 15 groups; test_x 4,801 videos (744 real, 4,057 fake), 344 groups = 44 identity groups plus 300 YouTube-real videos. Choice: the draw is over the 59 identity groups only (`--dev-identity-only`); the 300 YouTube-real videos have no identity, are not drawn and all stay test_x (a seeded draw over them would have been a new rule). Risk: the same people may appear in YouTube-real and in a dev identity, unknowable from the names.
- CNN subset: k=2 of 8 generator folders (unit = generator folder; every class group of a drawn generator moves together): stylegan (3 class groups) and whichfaceisreal. After de-duplication: split `cnn_subset-a1dd48b1a8bc`, 51,208 rows. dev_x 13,582 (stylegan 5,591 real + 5,991 fake; whichfaceisreal 1,000 + 1,000), test_x 37,626 (biggan 2,000/2,000; cyclegan 1,321/1,321; deepfake 2,707/2,698; gaugan 5,000/5,000; stargan 1,999/1,999; stylegan2 3,593 real/7,988 fake).
- De-duplication (images.exclusions.csv/json in data/_splits, command `dedup ... --name images`): 4,795 samples dropped, 4,394 distinct hashes, all in the CNN subset: dev_x 400 (stylegan/cat reals equal to ProGAN reals), test_x 4,395 (stylegan2/car 1,997, stylegan2/cat 1,997 and stylegan2/horse 401: the whole real folders of stylegan2 car and cat are byte-identical to the stylegan reals now in dev_x, plus ProGAN overlaps). Consequence for the coordinator: stylegan2 car and cat now contribute fakes only to test_x, so a per-generator stylegan2 AUC rests on church and horse reals (3,593 reals vs 7,988 fakes). Extension of D-026 that I made: dev_x rows equal to train, val or test_in content are dropped too, otherwise the hash check fails (dev_x stylegan cat reals equal progan_val train reals); tell me if you prefer otherwise.
- Exclusion file naming changed to `<name>.exclusions.csv/json` (a `<name>-*.csv` glob used to pick the exclusions file up as a split file).
- In-the-Wild: 31,779 clips (bona-fide 19,963, spoof 11,816), all with a wav; 54 speakers (DATA_LOG.md and D-035 say 58; the real meta.csv has 54), all 54 speakers appear with both labels, clips per speaker 14 to 3,636. Folder also holds attribution.txt (ignored). Slice k=12 of the 54 speakers (not of 58): split `in_the_wild-5cf2ea54a724`; dev_x 6,991 clips (3,804 real, 3,187 spoof) = 2Pac, Alan Watts, Alexandria Ocasio-Cortez, Ayn Rand, Bill Clinton, Bob Ross, Dave Chappelle, Gilbert Gottfried, Kanye West, Louis C.K., Martin Luther King, Nelson Mandela; test_x 24,788 clips (16,159 real, 8,629 spoof), 42 speakers. Hashing found no duplicates between dev_x and test_x.
- DFD: 363 originals + 3,068 manipulated = 3,431 rows, 28 actors (all with originals, 9 to 16 originals per actor; fakes per target actor 29 to 226), 293 distinct target-source pairs, no fake with target equal to source, split `dfd-f5ff5576814a` (all test_x, 28 groups). Rule: group = first actor of the name (the target, assumed), source actor and experiment id in notes; the strict actor-graph component is 3 components of 18, 5 and 5 actors, too coarse.
- Leakage with `--hash-files`: images (progan_val, progan_test, cnn_subset, diffusion_test; 77,208 files) PASS; Celeb-DF + In-the-Wild + DFD (41,739 files) PASS, warnings only (10 and 7 duplicate hashes inside one split). Not hashed: ASVspoof, FF++.

Questions: (a) the YouTube-real choice above; (b) the dev_x extension of D-026; (c) k for In-the-Wild is 12 of 54, DATA_LOG.md still says 58 speakers.
