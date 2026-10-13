# License and use terms

This file covers two different things under two different sets of terms: the
code in this repository, and the data the code was run on. They are not the
same and the second is the restrictive one.

**This is a draft and is not yet in force.** The roadmap requires the license to
be chosen after checking university and dataset terms, and the university check
has not happened. Until a person confirms the code terms below, treat this
repository as all rights reserved. What follows is a starting point for that
conversation, not a grant.

## Code

Everything under `src/`, `scripts/`, `tests/` and `configs/`, and the
documentation under `docs/`.

Proposed terms: MIT, subject to the university check below. The intent is that
the research platform, the evaluation protocol and the validity checks can be
reused and audited freely, because that reuse is what this project argues is
its contribution.

TODO before this is real:

- Confirm with the awarding institution who owns the copyright in work produced
  for a dissertation, and whether the student may license it. Institutions
  differ and several claim ownership or require approval.
- Confirm no supervisor, funder or industrial partner agreement restricts
  release.
- Once confirmed, add the chosen license text verbatim as `LICENSE` and reduce
  this file to the data section.

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
