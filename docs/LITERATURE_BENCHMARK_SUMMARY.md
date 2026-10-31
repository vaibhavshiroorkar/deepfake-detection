# LITERATURE BENCHMARK SUMMARY

Written 2026-10-30. Short version of docs/LITERATURE_BENCHMARK_COMPARISON.md for the report and the viva. Every external number is traced in docs/LITERATURE_BENCHMARK_EVIDENCE.md. Before citing any paper below in the paper or report, add it to docs/RESEARCH.md (CLAUDE.md rule 6).

## Our primary cross-dataset result

Video-level AUROC **0.8218 [0.8020, 0.8393]** on the sealed Celeb-DF family test (Celeb-DF v2 test_x plus Celeb-DF++ fakes; 744 real, 6,906 fake; 344 identity groups; 3 seeds; E-014, confirmatory test T1). Model: SigLIP base, fully fine-tuned on FaceForensics++ c23. At the frozen threshold (post hoc): accuracy 0.6188, balanced accuracy 0.7399, sensitivity 0.5896, specificity 0.8902, precision 0.9804, F1 0.7360. The set is 90.3 percent fake, so accuracy alone is misleading.

## Best directly comparable baseline

Only one exists: the Xception we re-ran under the same pipeline, data and test items. Celeb-DF family AUROC 0.8042 [0.7841, 0.8223]. Difference +0.0176 [+0.0025, +0.0327]; at the Bonferroni level [-0.0011, +0.0355]. The re-run Xception is higher on the Celeb-DF++ block (0.7351 against 0.7105) and on DFD (0.8847 against 0.8797).

No published paper reports results on our test population (the Celeb-DF family pooling and the exclusion of 15 development identities are this project's protocol).

## Closest published results (partially comparable, Celeb-DF v2, video level, trained on FF++)

GenD 96.0 (WACV 2026; 5-seed mean 94.6), Forensics Adapter 95.7 (CVPR 2025), Effort 95.6 (ICML 2025; reproduced elsewhere at 93.2), LAA-Net 95.40 (CVPR 2024), DFD-FCG 95.0 (CVPR 2025), SBI 93.18 (CVPR 2022), TALL 90.79 (ICCV 2023), AltFreezing 89.5 (CVPR 2023), FTCN 86.9 (ICCV 2021), RealForensics 86.9 (CVPR 2022), LipForensics 82.4 (CVPR 2021), Xception as re-trained by LipForensics 73.7. Our closest matching number: **0.8478** on the 399 official-list test videos that lie in our test_x (post hoc). Differences: test items (at most 399 of 518 shared), 16 against 32 frames, face detector and crop, training compression and paradigm, no intervals on the published side.

## Our numerical position

- **Directly comparable group (2 models):** first by point estimate; not distinguished from Xception after the Bonferroni correction; non-inferior with margin 0.0100.
- **Published literature:** no defensible rank. Descriptively, 0.8478 lies above LipForensics and the LipForensics Xception and below the other ten verified video-level methods, about 0.10 AUROC below the 2025 to 2026 CLIP ViT-L/14 detectors.

## Strongest limitations of the comparison

1. No published result shares our test items; the primary set does not exist in the literature.
2. Published values are single runs without intervals; independent reproductions differ by up to 2.4 points.
3. Frame-level values (DeepfakeBench, 0.61 to 0.77) are not comparable with our video-level values; video level is about 4 to 6 points higher for the same model.
4. DeepfakeBench selects its models on the test set, and several papers do not state their test list, unit or training compression.
5. Our official-list value (0.8478) is post hoc and covers 77 percent of that list.

## Defensible conclusions

1. Under our own sealed protocol the fine-tuned SigLIP base detector is non-inferior to a re-run Xception baseline (margin 0.0100); superiority does not survive the Bonferroni correction.
2. It is not competitive with current published state of the art on Celeb-DF v2: the closest available evidence places it about 0.10 AUROC below 2025 to 2026 foundation-model detectors, and below SBI, TALL, AltFreezing and FTCN.
3. It is in the range of earlier methods (above the LipForensics re-training of Xception and LipForensics itself in point estimate), but test sets differ, so this is a placement, not a rank.
4. Newer manipulation methods (Celeb-DF++) are where it is weakest (0.7105), matching the published finding that Celeb-DF++ is harder than Celeb-DF v2 for every detector.
5. The decision threshold does not transfer across datasets (false alarms 1.4 percent in FF++ against 11 to 14 percent elsewhere), so ranking (AUROC) and deployability (error rates) must be reported separately.
