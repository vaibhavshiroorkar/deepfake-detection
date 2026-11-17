# P-001 amendment A17: speaker overlap with AV-HuBERT's pretraining

**Exploratory.** Fixed in [amendment A17](../preregistration/P-001-shortcut-neutral-av-study.md) before this result existed.

| FakeAVCeleb | VoxCeleb2 dev | VoxCeleb2 test | Neither |
|---|---|---|---|
| Source identities (500) | 500 | 0 | 0 |
| Neutral test partition: clips (2,993) | 2,993 | 0 | 0 |

## Reading

- **Every FakeAVCeleb speaker is a VoxCeleb2 dev speaker.** AV-HuBERT Base was pretrained on LRS3 and the English part of VoxCeleb2 dev (assumed from the AV-HuBERT paper; the English filter can only lower the overlap). So every speaker in our training and test data may have been heard by the frozen encoder during pretraining, and no FakeAVCeleb speaker is known to be new to it.
- **No comparison by group is possible.** The rule fixed in A17 required at least 5 test identities in each of dev and test; the test group is empty, so only counts are reported.
- **LAV-DF:** built from VoxCeleb2 (its README), with no speaker ids, so its overlap is unknown.
- **What it means for the paper.** The limitation "possible speaker overlap" becomes "complete overlap with VoxCeleb2 dev for FakeAVCeleb". The sync path's in-domain score may benefit from an encoder that has heard these speakers. Its effect is not measured, since no unseen speaker exists in this corpus.

## Source

VoxCeleb2's dev and test speaker lists come from the public dataset [`yangwang825/vox2-veri-3s`](https://huggingface.co/datasets/yangwang825/vox2-veri-3s) (`train.csv`, `test.csv`), because the official metadata now needs a login. Its counts match the published split: 5,994 dev and 118 test speakers, 36,237 test utterances. `vox2-speakers.csv` keeps the derived list (speaker id, split), and `a17.json` records the SHA-256 of both source files. VoxCeleb metadata is released under CC BY 4.0 (assumed; not checked tonight).

    .venv/Scripts/python.exe scripts/run_a17.py      # rerun from vox2-speakers.csv
