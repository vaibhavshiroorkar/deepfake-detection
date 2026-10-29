# License and use terms

This file covers the code in this repository and the data the code was run on.
They have different terms.

## Code

Copyright (c) 2026 Vaibhav Shiroorkar. All rights reserved.

This covers everything under `src/`, `scripts/`, `tests/`, `configs/` and
`hosting/`, and the documentation under `docs/`. No license is granted to copy,
modify or redistribute it. Ask the author for permission.

## Data

**No dataset is included in this repository and none may be redistributed
through it.** `data/` is ignored by Git and always has been.

Every corpus used here was obtained under its own terms and those terms survive
this file. They are research-only in every case, and several require an
individual agreement that cannot be transferred to someone who clones this
repository.

| Corpus | Role here | How it is obtained |
| --- | --- | --- |
| FakeAVCeleb | Primary development set | Author request form, research use only |
| LAV-DF | Cross-modal stream training | Author release terms |
| Celeb-DF-v2 | Cross-dataset generalization | Signed request to the authors |
| DFDC | Cross-corpus test | Competition and dataset terms |
| FaceForensics++ | Declared visual experiments | Signed EULA with the authors |
| MNW | Locked external benchmark | Benchmark release terms, evaluation only |
| DF26 | Open-world generated media | Automatic gate on the hosting platform |

Anyone reproducing this work must obtain each corpus themselves under its own
terms. The scripts under `scripts/` that fetch data do not bypass any gate;
they automate a download the user is already entitled to make.

## Derived artifacts

Checkpoints, cached views, feature stores and run outputs are ignored by Git and
are not distributed here. They are derived from the corpora above and inherit
those corpora's terms, including the biometric and personal-data handling rules
in [the data card](docs/data-card.md). A trained checkpoint is not a clean-room
artifact: publishing one may be a redistribution of the data it was trained on
under some of these agreements, and that has not been checked.

## What this system may be used for

The [model card](docs/model-card.md) and [threat model](docs/threat-model.md)
govern use, and they are narrower than any code license. In short: this is
research evidence, it should not be deployed, and its score must not by itself
decide identity, law-enforcement, employment or moderation outcomes.
