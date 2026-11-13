# Generated inventory

This file is written by `ddf handoff update` from the working tree. Do not edit
the blocks by hand; run the command again after any dataset or run changes.
It exists because an earlier version of this file said `data/` was empty while
both primary datasets were on disk.

What the project is, where it stands and what to do next is in the
[handbook](HANDBOOK.md).

## Datasets

<!-- BEGIN GENERATED DATASETS -->
| Dataset directory | Role | State | Videos | Size |
|---|---|---|---:|---:|
| `data/FakeAVCeleb_v1.2` | Primary development set | present | 21,544 | 6.2 GB |
| `data/LAV-DF` | Cross-modal stream training, localized forgeries | present | 136,304 | 23.6 GB |
| `data/Celeb-DF-v2` | Cross-dataset generalization, no audio | present | 6,529 | 9.5 GB |
| `data/DFDC` | Cross-corpus test, the only one with audio not from VoxCeleb2 | present | 3,032 | 20.9 GB |
| `data/FaceForensics++` | Declared visual experiments | present | 7,000 | 16.7 GB |
| `data/MNW` | Locked external benchmark, evaluation only | present | 134 | 424.0 MB |
<!-- END GENERATED DATASETS -->

## Evaluation reports

Every evaluation report under `runs/`. A row reading `undefined` for ROC-AUC is
a single-class set, where no ranking metric exists. Older reports are not
results; see [handbook section 5](HANDBOOK.md#5-what-we-have-measured).

<!-- BEGIN GENERATED EVIDENCE -->
| Report | Dataset | Scope | Rows | ROC-AUC | Balanced accuracy |
|---|---|---|---:|---:|---:|
| `runs/comparison-20260904/visual-efficientnet-b0-ep8-seed17-validation-metrics.json` | FakeAVCeleb | development_comparison | 400 | 0.9992 | 0.9975 |
| `runs/comparison-20260904/visual-efficientnet-b0-initial-seed29-validation-metrics.json` | FakeAVCeleb | development_comparison | 400 | 1.0000 | 0.9975 |
| `runs/comparison-20260904/visual-efficientnet-b0-initial-seed43-validation-metrics.json` | FakeAVCeleb | development_comparison | 400 | 0.9998 | 0.9925 |
| `runs/comparison-20260904/visual-efficientnet-b0-lr3e-4-seed17-validation-metrics.json` | FakeAVCeleb | development_comparison | 400 | 1.0000 | 0.9950 |
| `runs/design-b-20260910/evaluation/gate-fusion-dfdc-metrics.json` | DFDC | ? | 2,410 | 0.5488 | 0.5495 |
| `runs/design-b-20260910/evaluation/gate-fusion-in-domain-metrics.json` | FakeAVCeleb | ? | 1,648 | 0.9987 | 0.9921 |
| `runs/full-20260904/evaluation/baseline-visual-mnw-metrics.json` | MNW | external_mnw | 85 | 0.0357 | 0.0952 |
| `runs/full-20260904/evaluation/pilot-audio-validation-metrics.json` | FakeAVCeleb | development_validation | 598 | 1.0000 | 0.9984 |
| `runs/full-20260904/evaluation/pilot-visual-celebdf-metrics.json` | Celeb-DF-v2 | generalization_celebdf | 462 | 0.6476 | 0.5741 |
| `runs/full-20260904/evaluation/pilot-visual-mnw-metrics.json` | MNW | external_mnw | 85 | 0.0238 | 0.0952 |
| `runs/full-20260904/evaluation/pilot-visual-test-metrics.json` | FakeAVCeleb | development_test | 597 | 1.0000 | 1.0000 |
| `runs/full-20260904/evaluation/pilot-visual-validation-metrics.json` | FakeAVCeleb | development_validation | 598 | 1.0000 | 1.0000 |
| `runs/hosting/public-demo-20261024T103218/runtime/runs/program-20260906/evaluation/visual-celebdf-metrics.json` | Celeb-DF-v2 | generalization_celebdf | 462 | 0.6682 | 0.5435 |
| `runs/hosting/public-demo-20261024T103218/runtime/runs/program-20260906/evaluation/visual-celebdf-seed29-metrics.json` | Celeb-DF-v2 | generalization_celebdf | 462 | 0.6698 | 0.5821 |
| `runs/hosting/public-demo-20261024T103218/runtime/runs/program-20260906/evaluation/visual-celebdf-seed43-metrics.json` | Celeb-DF-v2 | generalization_celebdf | 462 | 0.6611 | 0.5628 |
| `runs/hosting/public-demo-20261024T103218/runtime/runs/program-20260906/evaluation/visual-convnext-celebdf-metrics.json` | Celeb-DF-v2 | generalization_celebdf | 462 | 0.6155 | 0.5956 |
| `runs/hosting/public-demo-20261024T103218/runtime/runs/program-20260906/evaluation/visual-convnext-celebdf-seed29-metrics.json` | Celeb-DF-v2 | generalization_celebdf | 462 | 0.6184 | 0.6302 |
| `runs/hosting/public-demo-20261024T103218/runtime/runs/program-20260906/evaluation/visual-convnext-celebdf-seed43-metrics.json` | Celeb-DF-v2 | generalization_celebdf | 462 | 0.6287 | 0.5913 |
| `runs/hosting/public-demo-20261024T103218/runtime/runs/program-20260906/evaluation/visual-convnext-dfdc-metrics.json` | DFDC | generalization_celebdf | 2,410 | 0.7272 | 0.5066 |
| `runs/hosting/public-demo-20261024T103218/runtime/runs/program-20260906/evaluation/visual-convnext-dfdc-seed29-metrics.json` | DFDC | generalization_celebdf | 2,410 | 0.6776 | 0.5358 |
| `runs/hosting/public-demo-20261024T103218/runtime/runs/program-20260906/evaluation/visual-convnext-dfdc-seed43-metrics.json` | DFDC | generalization_celebdf | 2,410 | 0.6301 | 0.5720 |
| `runs/hosting/public-demo-20261024T103218/runtime/runs/program-20260906/evaluation/visual-convnext-in-domain-test-metrics.json` | FakeAVCeleb | development_test | 1,648 | 0.9992 | 0.9973 |
| `runs/hosting/public-demo-20261024T103218/runtime/runs/program-20260906/evaluation/visual-convnext-in-domain-test-seed29-metrics.json` | FakeAVCeleb | development_test | 1,648 | 0.9986 | 0.9957 |
| `runs/hosting/public-demo-20261024T103218/runtime/runs/program-20260906/evaluation/visual-convnext-in-domain-test-seed43-metrics.json` | FakeAVCeleb | development_test | 1,648 | 0.9032 | 0.7919 |
| `runs/hosting/public-demo-20261024T103218/runtime/runs/program-20260906/evaluation/visual-convnext-mnw-metrics.json` | MNW | external_mnw | 85 | 0.0000 | 0.0238 |
| `runs/hosting/public-demo-20261024T103218/runtime/runs/program-20260906/evaluation/visual-convnext-mnw-seed29-metrics.json` | MNW | external_mnw | 85 | 0.0238 | 0.0476 |
| `runs/hosting/public-demo-20261024T103218/runtime/runs/program-20260906/evaluation/visual-convnext-mnw-seed43-metrics.json` | MNW | external_mnw | 85 | 0.3571 | 0.5655 |
| `runs/hosting/public-demo-20261024T103218/runtime/runs/program-20260906/evaluation/visual-dfdc-metrics.json` | DFDC | generalization_celebdf | 2,410 | 0.7611 | 0.5677 |
| `runs/hosting/public-demo-20261024T103218/runtime/runs/program-20260906/evaluation/visual-dfdc-seed17-metrics.json` | DFDC | generalization_celebdf | 2,410 | 0.7611 | 0.5677 |
| `runs/hosting/public-demo-20261024T103218/runtime/runs/program-20260906/evaluation/visual-dfdc-seed29-metrics.json` | DFDC | generalization_celebdf | 2,410 | 0.6963 | 0.5354 |
| `runs/hosting/public-demo-20261024T103218/runtime/runs/program-20260906/evaluation/visual-dfdc-seed43-metrics.json` | DFDC | generalization_celebdf | 2,410 | 0.6878 | 0.5216 |
| `runs/hosting/public-demo-20261024T103218/runtime/runs/program-20260906/evaluation/visual-ensemble-celebdf-metrics.json` | Celeb-DF-v2 | ? | 462 | 0.6677 | 0.6143 |
| `runs/hosting/public-demo-20261024T103218/runtime/runs/program-20260906/evaluation/visual-ensemble-dfdc-metrics.json` | DFDC | ? | 2,410 | 0.7739 | 0.5181 |
| `runs/hosting/public-demo-20261024T103218/runtime/runs/program-20260906/evaluation/visual-ensemble-in-domain-test-metrics.json` | FakeAVCeleb | ? | 1,648 | 0.9994 | 0.9990 |
| `runs/hosting/public-demo-20261024T103218/runtime/runs/program-20260906/evaluation/visual-in-domain-test-metrics.json` | FakeAVCeleb | development_test | 1,648 | 0.9988 | 0.9990 |
| `runs/hosting/public-demo-20261024T103218/runtime/runs/program-20260906/evaluation/visual-in-domain-test-seed17-metrics.json` | FakeAVCeleb | development_test | 1,648 | 0.9988 | 0.9990 |
| `runs/hosting/public-demo-20261024T103218/runtime/runs/program-20260906/evaluation/visual-in-domain-test-seed29-metrics.json` | FakeAVCeleb | development_test | 1,648 | 0.9991 | 0.9990 |
| `runs/hosting/public-demo-20261024T103218/runtime/runs/program-20260906/evaluation/visual-in-domain-test-seed43-metrics.json` | FakeAVCeleb | development_test | 1,648 | 0.9993 | 0.9987 |
| `runs/hosting/public-demo-20261024T103218/runtime/runs/program-20260906/evaluation/visual-mnw-metrics.json` | MNW | external_mnw | 85 | 0.0000 | 0.1429 |
| `runs/hosting/public-demo-20261024T103218/runtime/runs/program-20260906/evaluation/visual-mnw-seed29-metrics.json` | MNW | external_mnw | 85 | 0.0952 | 0.1131 |
| `runs/hosting/public-demo-20261024T103218/runtime/runs/program-20260906/evaluation/visual-mnw-seed43-metrics.json` | MNW | external_mnw | 85 | 0.0000 | 0.1190 |
| `runs/hosting/public-demo-20261024T103916/runtime/runs/program-20260906/evaluation/visual-celebdf-metrics.json` | Celeb-DF-v2 | generalization_celebdf | 462 | 0.6682 | 0.5435 |
| `runs/hosting/public-demo-20261024T103916/runtime/runs/program-20260906/evaluation/visual-celebdf-seed29-metrics.json` | Celeb-DF-v2 | generalization_celebdf | 462 | 0.6698 | 0.5821 |
| `runs/hosting/public-demo-20261024T103916/runtime/runs/program-20260906/evaluation/visual-celebdf-seed43-metrics.json` | Celeb-DF-v2 | generalization_celebdf | 462 | 0.6611 | 0.5628 |
| `runs/hosting/public-demo-20261024T103916/runtime/runs/program-20260906/evaluation/visual-convnext-celebdf-metrics.json` | Celeb-DF-v2 | generalization_celebdf | 462 | 0.6155 | 0.5956 |
| `runs/hosting/public-demo-20261024T103916/runtime/runs/program-20260906/evaluation/visual-convnext-celebdf-seed29-metrics.json` | Celeb-DF-v2 | generalization_celebdf | 462 | 0.6184 | 0.6302 |
| `runs/hosting/public-demo-20261024T103916/runtime/runs/program-20260906/evaluation/visual-convnext-celebdf-seed43-metrics.json` | Celeb-DF-v2 | generalization_celebdf | 462 | 0.6287 | 0.5913 |
| `runs/hosting/public-demo-20261024T103916/runtime/runs/program-20260906/evaluation/visual-convnext-dfdc-metrics.json` | DFDC | generalization_celebdf | 2,410 | 0.7272 | 0.5066 |
| `runs/hosting/public-demo-20261024T103916/runtime/runs/program-20260906/evaluation/visual-convnext-dfdc-seed29-metrics.json` | DFDC | generalization_celebdf | 2,410 | 0.6776 | 0.5358 |
| `runs/hosting/public-demo-20261024T103916/runtime/runs/program-20260906/evaluation/visual-convnext-dfdc-seed43-metrics.json` | DFDC | generalization_celebdf | 2,410 | 0.6301 | 0.5720 |
| `runs/hosting/public-demo-20261024T103916/runtime/runs/program-20260906/evaluation/visual-convnext-in-domain-test-metrics.json` | FakeAVCeleb | development_test | 1,648 | 0.9992 | 0.9973 |
| `runs/hosting/public-demo-20261024T103916/runtime/runs/program-20260906/evaluation/visual-convnext-in-domain-test-seed29-metrics.json` | FakeAVCeleb | development_test | 1,648 | 0.9986 | 0.9957 |
| `runs/hosting/public-demo-20261024T103916/runtime/runs/program-20260906/evaluation/visual-convnext-in-domain-test-seed43-metrics.json` | FakeAVCeleb | development_test | 1,648 | 0.9032 | 0.7919 |
| `runs/hosting/public-demo-20261024T103916/runtime/runs/program-20260906/evaluation/visual-convnext-mnw-metrics.json` | MNW | external_mnw | 85 | 0.0000 | 0.0238 |
| `runs/hosting/public-demo-20261024T103916/runtime/runs/program-20260906/evaluation/visual-convnext-mnw-seed29-metrics.json` | MNW | external_mnw | 85 | 0.0238 | 0.0476 |
| `runs/hosting/public-demo-20261024T103916/runtime/runs/program-20260906/evaluation/visual-convnext-mnw-seed43-metrics.json` | MNW | external_mnw | 85 | 0.3571 | 0.5655 |
| `runs/hosting/public-demo-20261024T103916/runtime/runs/program-20260906/evaluation/visual-dfdc-metrics.json` | DFDC | generalization_celebdf | 2,410 | 0.7611 | 0.5677 |
| `runs/hosting/public-demo-20261024T103916/runtime/runs/program-20260906/evaluation/visual-dfdc-seed17-metrics.json` | DFDC | generalization_celebdf | 2,410 | 0.7611 | 0.5677 |
| `runs/hosting/public-demo-20261024T103916/runtime/runs/program-20260906/evaluation/visual-dfdc-seed29-metrics.json` | DFDC | generalization_celebdf | 2,410 | 0.6963 | 0.5354 |
| `runs/hosting/public-demo-20261024T103916/runtime/runs/program-20260906/evaluation/visual-dfdc-seed43-metrics.json` | DFDC | generalization_celebdf | 2,410 | 0.6878 | 0.5216 |
| `runs/hosting/public-demo-20261024T103916/runtime/runs/program-20260906/evaluation/visual-ensemble-celebdf-metrics.json` | Celeb-DF-v2 | ? | 462 | 0.6677 | 0.6143 |
| `runs/hosting/public-demo-20261024T103916/runtime/runs/program-20260906/evaluation/visual-ensemble-dfdc-metrics.json` | DFDC | ? | 2,410 | 0.7739 | 0.5181 |
| `runs/hosting/public-demo-20261024T103916/runtime/runs/program-20260906/evaluation/visual-ensemble-in-domain-test-metrics.json` | FakeAVCeleb | ? | 1,648 | 0.9994 | 0.9990 |
| `runs/hosting/public-demo-20261024T103916/runtime/runs/program-20260906/evaluation/visual-in-domain-test-metrics.json` | FakeAVCeleb | development_test | 1,648 | 0.9988 | 0.9990 |
| `runs/hosting/public-demo-20261024T103916/runtime/runs/program-20260906/evaluation/visual-in-domain-test-seed17-metrics.json` | FakeAVCeleb | development_test | 1,648 | 0.9988 | 0.9990 |
| `runs/hosting/public-demo-20261024T103916/runtime/runs/program-20260906/evaluation/visual-in-domain-test-seed29-metrics.json` | FakeAVCeleb | development_test | 1,648 | 0.9991 | 0.9990 |
| `runs/hosting/public-demo-20261024T103916/runtime/runs/program-20260906/evaluation/visual-in-domain-test-seed43-metrics.json` | FakeAVCeleb | development_test | 1,648 | 0.9993 | 0.9987 |
| `runs/hosting/public-demo-20261024T103916/runtime/runs/program-20260906/evaluation/visual-mnw-metrics.json` | MNW | external_mnw | 85 | 0.0000 | 0.1429 |
| `runs/hosting/public-demo-20261024T103916/runtime/runs/program-20260906/evaluation/visual-mnw-seed29-metrics.json` | MNW | external_mnw | 85 | 0.0952 | 0.1131 |
| `runs/hosting/public-demo-20261024T103916/runtime/runs/program-20260906/evaluation/visual-mnw-seed43-metrics.json` | MNW | external_mnw | 85 | 0.0000 | 0.1190 |
| `runs/initial-20260902/visual-validation-metrics.json` | FakeAVCeleb | development_validation | 400 | 0.9992 | 0.9975 |
| `runs/program-20260906/evaluation/audio-in-domain-test-metrics.json` | FakeAVCeleb | development_test | 1,648 | 1.0000 | 1.0000 |
| `runs/program-20260906/evaluation/audio-in-domain-test-seed29-metrics.json` | FakeAVCeleb | development_test | 1,648 | 1.0000 | 0.9962 |
| `runs/program-20260906/evaluation/audio-in-domain-test-seed43-metrics.json` | FakeAVCeleb | development_test | 1,648 | 0.9986 | 0.9906 |
| `runs/program-20260906/evaluation/audio-wavlm-in-domain-test-metrics.json` | FakeAVCeleb | development_test | 1,648 | 1.0000 | 0.9981 |
| `runs/program-20260906/evaluation/fusion-dfdc-metrics.json` | FakeAVCeleb | fusion | 2,351 | 0.7500 | 0.6203 |
| `runs/program-20260906/evaluation/fusion-test-metrics.json` | FakeAVCeleb | fusion | 1,648 | 0.9990 | 0.9990 |
| `runs/program-20260906/evaluation/sync-final-sync-seed17-metrics.json` | FakeAVCeleb | ? | 0 | 0.4364 | 0.5000 |
| `runs/program-20260906/evaluation/sync-final-sync-seed29-metrics.json` | FakeAVCeleb | ? | 0 | 0.4570 | 0.5000 |
| `runs/program-20260906/evaluation/sync-final-sync-seed43-metrics.json` | FakeAVCeleb | ? | 0 | 0.4931 | 0.5000 |
| `runs/program-20260906/evaluation/visual-celebdf-metrics.json` | Celeb-DF-v2 | generalization_celebdf | 462 | 0.6682 | 0.5435 |
| `runs/program-20260906/evaluation/visual-celebdf-seed29-metrics.json` | Celeb-DF-v2 | generalization_celebdf | 462 | 0.6698 | 0.5821 |
| `runs/program-20260906/evaluation/visual-celebdf-seed43-metrics.json` | Celeb-DF-v2 | generalization_celebdf | 462 | 0.6611 | 0.5628 |
| `runs/program-20260906/evaluation/visual-convnext-celebdf-metrics.json` | Celeb-DF-v2 | generalization_celebdf | 462 | 0.6155 | 0.5956 |
| `runs/program-20260906/evaluation/visual-convnext-celebdf-seed29-metrics.json` | Celeb-DF-v2 | generalization_celebdf | 462 | 0.6184 | 0.6302 |
| `runs/program-20260906/evaluation/visual-convnext-celebdf-seed43-metrics.json` | Celeb-DF-v2 | generalization_celebdf | 462 | 0.6287 | 0.5913 |
| `runs/program-20260906/evaluation/visual-convnext-dfdc-metrics.json` | DFDC | generalization_celebdf | 2,410 | 0.7272 | 0.5066 |
| `runs/program-20260906/evaluation/visual-convnext-dfdc-seed29-metrics.json` | DFDC | generalization_celebdf | 2,410 | 0.6776 | 0.5358 |
| `runs/program-20260906/evaluation/visual-convnext-dfdc-seed43-metrics.json` | DFDC | generalization_celebdf | 2,410 | 0.6301 | 0.5720 |
| `runs/program-20260906/evaluation/visual-convnext-in-domain-test-metrics.json` | FakeAVCeleb | development_test | 1,648 | 0.9992 | 0.9973 |
| `runs/program-20260906/evaluation/visual-convnext-in-domain-test-seed29-metrics.json` | FakeAVCeleb | development_test | 1,648 | 0.9986 | 0.9957 |
| `runs/program-20260906/evaluation/visual-convnext-in-domain-test-seed43-metrics.json` | FakeAVCeleb | development_test | 1,648 | 0.9032 | 0.7919 |
| `runs/program-20260906/evaluation/visual-convnext-mnw-metrics.json` | MNW | external_mnw | 85 | 0.0000 | 0.0238 |
| `runs/program-20260906/evaluation/visual-convnext-mnw-seed29-metrics.json` | MNW | external_mnw | 85 | 0.0238 | 0.0476 |
| `runs/program-20260906/evaluation/visual-convnext-mnw-seed43-metrics.json` | MNW | external_mnw | 85 | 0.3571 | 0.5655 |
| `runs/program-20260906/evaluation/visual-dfdc-metrics.json` | DFDC | generalization_celebdf | 2,410 | 0.7611 | 0.5677 |
| `runs/program-20260906/evaluation/visual-dfdc-seed17-metrics.json` | DFDC | generalization_celebdf | 2,410 | 0.7611 | 0.5677 |
| `runs/program-20260906/evaluation/visual-dfdc-seed29-metrics.json` | DFDC | generalization_celebdf | 2,410 | 0.6963 | 0.5354 |
| `runs/program-20260906/evaluation/visual-dfdc-seed43-metrics.json` | DFDC | generalization_celebdf | 2,410 | 0.6878 | 0.5216 |
| `runs/program-20260906/evaluation/visual-ensemble-celebdf-metrics.json` | Celeb-DF-v2 | ? | 462 | 0.6677 | 0.6143 |
| `runs/program-20260906/evaluation/visual-ensemble-dfdc-metrics.json` | DFDC | ? | 2,410 | 0.7739 | 0.5181 |
| `runs/program-20260906/evaluation/visual-ensemble-in-domain-test-metrics.json` | FakeAVCeleb | ? | 1,648 | 0.9994 | 0.9990 |
| `runs/program-20260906/evaluation/visual-in-domain-test-metrics.json` | FakeAVCeleb | development_test | 1,648 | 0.9988 | 0.9990 |
| `runs/program-20260906/evaluation/visual-in-domain-test-seed17-metrics.json` | FakeAVCeleb | development_test | 1,648 | 0.9988 | 0.9990 |
| `runs/program-20260906/evaluation/visual-in-domain-test-seed29-metrics.json` | FakeAVCeleb | development_test | 1,648 | 0.9991 | 0.9990 |
| `runs/program-20260906/evaluation/visual-in-domain-test-seed43-metrics.json` | FakeAVCeleb | development_test | 1,648 | 0.9993 | 0.9987 |
| `runs/program-20260906/evaluation/visual-mnw-metrics.json` | MNW | external_mnw | 85 | 0.0000 | 0.1429 |
| `runs/program-20260906/evaluation/visual-mnw-seed29-metrics.json` | MNW | external_mnw | 85 | 0.0952 | 0.1131 |
| `runs/program-20260906/evaluation/visual-mnw-seed43-metrics.json` | MNW | external_mnw | 85 | 0.0000 | 0.1190 |
| `runs/retrain-20261022/evaluation/blend-ffpp-seed17-celebdf-metrics.json` | Celeb-DF-v2 | generalization_celebdf | 462 | 0.6509 | 0.5907 |
| `runs/retrain-20261022/evaluation/blend-ffpp-seed17-dfdc-metrics.json` | DFDC | generalization_celebdf | 2,410 | 0.6756 | 0.5886 |
| `runs/retrain-20261022/evaluation/blend-ffpp-seed17-ffpp-test-metrics.json` | FaceForensics++ | development_comparison | 954 | 0.9103 | 0.8262 |
| `runs/retrain-20261022/evaluation/blend-ffpp-seed17-in-domain-test-metrics.json` | FakeAVCeleb | development_test | 1,648 | 0.9904 | 0.9596 |
| `runs/retrain-20261022/evaluation/blend-ffpp-seed17-validation-metrics.json` | FakeAVCeleb | development_validation | 1,644 | 0.9978 | 0.9729 |
| `runs/retrain-20261022/evaluation/blend-seed17-celebdf-metrics.json` | Celeb-DF-v2 | generalization_celebdf | 462 | 0.6762 | 0.6487 |
| `runs/retrain-20261022/evaluation/blend-seed17-dfdc-metrics.json` | DFDC | generalization_celebdf | 2,410 | 0.6496 | 0.6060 |
| `runs/retrain-20261022/evaluation/blend-seed17-ffpp-test-metrics.json` | FaceForensics++ | development_comparison | 954 | 0.7210 | 0.6737 |
| `runs/retrain-20261022/evaluation/blend-seed17-in-domain-test-metrics.json` | FakeAVCeleb | development_test | 1,648 | 0.9978 | 0.9767 |
| `runs/retrain-20261022/evaluation/blend-seed17-validation-metrics.json` | FakeAVCeleb | development_validation | 1,644 | 0.9989 | 0.9800 |
| `runs/retrain-20261022/evaluation/compress-seed17-celebdf-metrics.json` | Celeb-DF-v2 | generalization_celebdf | 462 | 0.6525 | 0.5460 |
| `runs/retrain-20261022/evaluation/compress-seed17-dfdc-metrics.json` | DFDC | generalization_celebdf | 2,410 | 0.5720 | 0.5051 |
| `runs/retrain-20261022/evaluation/compress-seed17-ffpp-test-metrics.json` | FaceForensics++ | development_comparison | 954 | 0.6960 | 0.6002 |
| `runs/retrain-20261022/evaluation/compress-seed17-in-domain-test-metrics.json` | FakeAVCeleb | development_test | 1,648 | 0.9959 | 0.9790 |
| `runs/retrain-20261022/evaluation/compress-seed17-validation-metrics.json` | FakeAVCeleb | development_validation | 1,644 | 0.9980 | 0.9806 |
| `runs/retrain-20261022/evaluation/plain-seed17-celebdf-metrics.json` | Celeb-DF-v2 | generalization_celebdf | 462 | 0.6682 | 0.5435 |
| `runs/retrain-20261022/evaluation/plain-seed17-dfdc-metrics.json` | DFDC | generalization_celebdf | 2,410 | 0.7611 | 0.5677 |
| `runs/retrain-20261022/evaluation/plain-seed17-ffpp-test-metrics.json` | FaceForensics++ | development_comparison | 954 | 0.7235 | 0.6814 |
| `runs/retrain-20261022/evaluation/plain-seed17-in-domain-test-metrics.json` | FakeAVCeleb | development_test | 1,648 | 0.9988 | 0.9990 |
| `runs/retrain-20261022/evaluation/plain-seed17-validation-metrics.json` | FakeAVCeleb | development_validation | 1,644 | 0.9995 | 0.9993 |
<!-- END GENERATED EVIDENCE -->

## MLflow store

<!-- BEGIN GENERATED MLFLOW -->
| Experiment | Runs |
|---|---|
| `design-b-20260910` | 5 (5 finished) |
| `design-b-frozen-bn` | 4 (4 finished) |
| `design-b-live-bn` | 2 (2 finished) |
| `ffpp-20260913` | 1 (1 finished) |
| `full-20260904` | 5 (5 finished) |
| `initial-baseline-20260902` | 10 (10 finished) |
| `program-20260906` | 21 (21 finished) |
| `prototype-gpu-20260902` | 7 (2 failed, 5 finished) |
| `smoke-fixture-fusion` | 10 (10 finished) |
| `streams-20260905` | 3 (3 finished) |
<!-- END GENERATED MLFLOW -->
