# DATA_LOG

Last updated: 2026-10-24 (Session 0)

Every dataset used in the project gets one row here before any file is downloaded into the project, and its row is completed when the download finishes. Licence and access terms are copied from the dataset's own page or agreement, not from a paper's description. Data files are stored on D: outside git (see .gitignore).

## Status at the end of Session 0

No dataset has been downloaded. Which datasets will be used is open decision O-003. The table below is the empty template.

| Dataset | Version | Source URL | Licence | Access terms | Size | File hash | Date downloaded | Role (train / val / test) | Notes |
|---|---|---|---|---|---|---|---|---|---|
| none yet | | | | | | | | | |

## Rules for entries

1. File hash: SHA-256 of each archive as downloaded, plus the hash of any manifest, so that the same bytes can be verified later.
2. Role: assigned only after the split policy (O-004) is decided. A dataset can be training data in one experiment and held-out test data in another only if the experiment record says so and no leakage check fails.
3. Notes: record anything affecting use, for example known duplicate identities, missing audio, or access conditions such as a request form.
4. Datasets mentioned in papers read in Session 0 for which access terms have not been checked: see RESEARCH.md, entries R-001, R-002, R-003, R-013 and R-014. One of them, FakeAVCeleb (R-014), is obtained through a request form according to its paper; the others have not been checked. All access terms must be confirmed at the source before use.
