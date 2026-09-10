# Experiment matrix

## Controls

All comparisons use one frozen source split, preprocessing hash, cache index,
training budget, early-stopping rule, and evaluation implementation. MNW is
evaluation-only. It cannot select a model or threshold.

The frozen artefacts for the current program are:

| Artefact | Value |
|---|---|
| Manifest | `runs/full-20260904/fakeavceleb-all.csv`, 21,544 rows from 21,566 meta rows |
| Split | 70/15/15 by source identity, seed 17, train 15,091 / val 3,288 / test 3,165 |
| Split hash | `10bb1f6fb7f469c86de23cd88c3d2ca1c32e7f8812d8124a550d0582cbe16033` |
| Preprocessing hash | `ac79f71e7614c96610e897eb011e01129b193da4482f415d037c6cc8c17638ec` |
| Source overlap | none between any pair of partitions |

## Tiers

The pilot and the full tier share one split and therefore one split hash. The
pilot trains on a stratified subsample of the same partitions, recorded by
`ddf split subsample`, so the two are directly comparable and the pilot's cache
is reused rather than rebuilt.

| Tier | Train | Val | Test | Purpose |
|---|---:|---:|---:|---|
| pilot | 2,800 | 600 | 600 | End-to-end evidence in about a day |
| full | 15,091 | 3,288 | 3,165 | The reported result |

The pilot keeps every real clip and thins fakes to three per real clip. That is
not cosmetic: for the visual branch the full train partition holds 14,391 fake
against 700 real videos, because FakeAVCeleb contains only 500 genuine
identities in total. Real clips are the scarce resource in every partition.

## Fixed seeds

Research branch comparisons use seeds 17, 29, and 43. A candidate remains
incomplete until all three runs finish or retain an explained failure. Audio and
sync run seed 17 first because each seed costs roughly nine and twelve hours at
full scale; the remaining seeds are queued behind the visual results.

## Experiment stages

| ID | Stage | Candidates | Selection evidence | Status |
| --- | --- | --- | --- | --- |
| DET-01 | Detector | MTCNN, YuNet | Reviewed training-only benchmark | planned |
| VIS-01 | Visual | EfficientNet-B0 plus GRU, ConvNeXt-Tiny | Validation and method-holdout metrics | planned |
| AUD-01 | Audio | Wav2Vec2 Base, WavLM, AASIST | Validation and method-holdout metrics | planned |
| SYN-01 | Sync | Current temporal branch, SyncNet-style baseline | Offset and mismatch metrics | planned |
| FUS-01 | Fusion | Logistic regression, small MLP | Out-of-fold validation metrics | planned |
| GEN-01 | Generalization | Frozen visual branch on Celeb-DF-v2 | Official 518-clip test list, zero-shot | running |
| EXT-01 | External | Frozen selected system on MNW | Locked zero-shot metrics | running |
| XM-01 | Cross-modal | Lip-sync stream, audio queries mouth | Diagonal attention mass plus validation metrics | running |
| XM-02 | Cross-modal | Emotion stream, voice queries face | Same, once the audio window alignment is fixed | planned |
| XM-03 | Cross-corpus | Lip-sync stream on DFDC | Zero-shot, the only audio-bearing corpus not built on VoxCeleb2 | planned |
| VST-01 | Visual stream | DINOv3 ViT-S/16, backbone frozen | In-domain test and DFDC ROC-AUC | superseded |
| VST-02 | Visual stream | EfficientNet-B0, fine-tuned after 2 frozen epochs | Same | superseded |
| VST-03 | Visual stream | Xception, fine-tuned after 2 frozen epochs | Same | queued |
| VST-04 | Visual stream | All three, checkpoint selected on ROC-AUC | Same | running |

VIS-01 needs code before it can run: `branches/visual.py` builds only
EfficientNet-B0, and `train visual` has no architecture flag, so the
ConvNeXt-Tiny comparison is not currently expressible. The VST rows are the
Design B answer to the same question: `ddf train visual-stream` does carry an
architecture flag, so the three backbones are one command apart.

VST-01 and VST-02 are superseded, and the reason is worth keeping. Both selected
their checkpoint on validation BCE, which kept the wrong epoch. VST-02 reached
its lowest loss at epoch 1 with the backbone still frozen, and that checkpoint
measured 0.4668 ROC-AUC in-domain, below chance. Their recorded numbers describe
the selection rule, not the architectures:

| Superseded run | In-domain | DFDC |
| --- | --- | --- |
| VST-01 DINOv3, frozen | 0.9085 | 0.5106 |
| VST-02 EfficientNet-B0 | 0.4668 | 0.6343 |

Do not cite either as evidence about a backbone. The claim VST-01 appeared to
refute, that a frozen self-supervised backbone trades in-domain accuracy for
cross-corpus transfer, is untested again and is what VST-04 re-runs.

VST-04 selects on ROC-AUC. See the early-stopping section of
[obstacles](../obstacles.md) for why the two metrics disagree, and why a rising
validation loss beside a falling training loss is not sufficient evidence of
overfitting.

## Cross-modal stream protocol

The lip-sync stream trains on LAV-DF rather than FakeAVCeleb, and the reason is
a measured one. FakeAVCeleb manipulates whole clips, so a stream can recognise
generator artifacts instead of learning alignment; a full run there held
diagonal attention mass at chance (0.0592 against 0.060) for every epoch while
validation loss reached 0.645. LAV-DF's forgeries are 0.66 seconds at the median
inside 7.3 second clips, and each one yields a matched pair from the same file:
a window on the manipulated span and a window that misses it, holding speaker,
codec and re-encoding constant.

Diagonal attention mass is reported for every stream run and is never optimised.
A falling loss beside a flat diagonal mass is a shortcut, and its accuracy must
not be quoted as evidence of synchronisation.

Corpus independence governs which evaluations count as cross-dataset:

| Corpus | Source | Audio | Role |
|---|---|---|---|
| FakeAVCeleb | VoxCeleb2 | yes | visual baseline training |
| LAV-DF | VoxCeleb2 | yes | cross-modal stream training |
| DFDC | paid actors | yes | cross-corpus test for the streams |
| Celeb-DF-v2 | YouTube | **none** | visual-only cross-dataset |
| MNW | mixed | **none** | visual-only locked external |

Scoring LAV-DF on FakeAVCeleb changes the generator, not the corpus. Only DFDC
is both independent and complete in both modalities.

## Ablation groups

Every run carries `ablation_group`, `tier` and `dataset` tags, so a study can be
queried out of MLflow instead of being inferred from run names. The groups the
CLI already supports without new code:

| Group | Mechanism |
|---|---|
| Branch combinations | `train fusion --branches` |
| Fusion model | `train fusion --model logistic\|mlp` |
| Sync label mode | `train sync --label-mode authentic-offset\|global-fake` |
| Leading silence | `cache build --keep-leading-silence` |
| Method holdout | `split method-holdout --methods` |
| Seeds | `--seed 17\|29\|43` |

## External evaluation constraints

MNW cannot be read the way an in-domain set is read, and the reasons are
properties of the data rather than choices:

- Its lab half has no audio track, so only the visual branch can be scored. The
  cache marks every MNW clip `missing_audio` and writes no audio view.
- Its lab half has no real videos, so ROC-AUC is undefined. The reported figure
  is the per-generator detection rate at the frozen threshold.
- 120 lab clips and 11 usable in-the-wild clips support wide intervals and no
  subgroup analysis.

See [docs/data-card.md](../data-card.md) for the full inventory and the
`FakeVideo-RealAudio` schema caveat.

## Status rules

Use only `planned`, `running`, `failed`, `accepted`, or `superseded`. Add MLflow
run IDs only after runs exist. Never convert smoke fixture metrics into a
research result.
