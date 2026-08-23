# Research design

## Research question

Under source-disjoint and shortcut-controlled evaluation, does cue-specific fusion of visual artifacts, audio spoofing, and mouth-audio alignment generalize better than a strong visual baseline?

The main contribution is the evaluation method and evidence. The project does not claim that multimodal fusion is new.

Prior work already covers [audio-video feature fusion](https://openaccess.thecvf.com/content/CVPR2024/html/Oorloff_AVFF_Audio-Visual_Feature_Fusion_for_Video_Deepfake_Detection_CVPR_2024_paper.html), [shortcut-controlled learning](https://openaccess.thecvf.com/content/CVPR2025/html/Smeu_Circumventing_Shortcuts_in_Audio-visual_Deepfake_Detection_Datasets_with_Unsupervised_Learning_CVPR_2025_paper.html), [synchronization models](https://openaccess.thecvf.com/content/ICCV2025/html/Anshul_Intra-modal_and_Cross-modal_Synchronization_for_Audio-visual_Deepfake_Detection_and_Temporal_ICCV_2025_paper.html), and [holistic coherence learning](https://openaccess.thecvf.com/content/CVPR2026F/html/Peng_Leave_No_Stone_Unturned_Uncovering_Holistic_Audio-Visual_Intrinsic_Coherence_for_CVPRF_2026_paper.html).

## System design

```text
manifest -> shared timeline -> visual, audio, and sync views
         -> cue-specific branches -> calibrated late fusion
         -> prediction, coverage, provenance, and evaluation
```

The visual branch uses EfficientNet-B0 and a GRU. It learns `video_fake`.

The audio branch uses Wav2Vec2 Base and attentive pooling. It learns `audio_fake`. Waveforms use the normalization expected by the pretrained encoder.

The sync branch uses a framewise ResNet-18 mouth encoder and a separate Wav2Vec2 Base encoder. It keeps temporal tokens. Authentic clips provide aligned pairs, shifted pairs, and cross-identity mismatches. The training offsets are 80, 160, and 320 milliseconds in both directions. Shifted audio comes from a wider decoded context. Padding cannot reveal the offset class.

The primary fusion model is regularized logistic regression. Each branch receives Platt calibration before fusion. Quality values enter as separate features. A small feature MLP is an optional ablation. Fusion training accepts only rows marked as out-of-fold predictions. Cross-fitting runs inside the frozen training partition. Validation and test identities never enter it.

## Data protocol

The primary split is 70/15/15 by source identity. Demographic strata guide source allocation. Training may use weighted sampling. Validation and test retain their natural class and method frequencies.

Target identities can cross the source split. FakeAVCeleb's source-target graph connects every identity. A full split that isolates both roles would remove much of the data. The protocol therefore includes a smaller identity-strict stress subset. Its reduced method coverage must appear beside its metrics.

Every manifest row contains separate clip, video, and audio labels. Conflicting duplicates enter quarantine. The split audit records source overlap, all-identity overlap, method counts, strict subset sizes, and a stable split hash.

For method holdout runs, training and validation exclude the held-out method. The test partition contains real clips and fake clips from that method. This prevents validation tuning on the claimed unseen method.

## Evaluation protocol

Required evaluations are:

- Full source-disjoint FakeAVCeleb test.
- Identity-strict filtered stress test.
- Leave-one-method-family-out tests.
- External zero-shot evaluation on [Deepfake-Eval](https://arxiv.org/html/2503.02857v5).
- Leading-silence, compression, noise, and resolution stress tests.
- Race and gender subgroups when sample counts support them.

Report ROC-AUC, PR-AUC, balanced accuracy, F1, precision, recall, FPR, FNR, EER, FPR at 95 percent TPR, Brier score, expected calibration error, coverage, and abstention rate.

Use 1,000 source-identity bootstrap samples for 95 percent confidence intervals. Compare fusion with the visual baseline through a paired source bootstrap. Run three fixed training seeds.

Choose all model settings and thresholds on training and validation data. Run the locked test set once. Do not tune on Deepfake-Eval.

## Required ablations

- Each branch alone.
- Visual plus audio.
- Visual plus sync.
- All three branches.
- Authentic-only sync learning compared with global fake-label tuning.
- Leading-silence removal enabled and disabled.
- Quality-aware abstention compared with silent fallback.
- Late fusion compared with the optional small MLP.

The CLI exposes these through fusion branch selection, the fusion model option, the sync label mode, and the cache leading-silence option.

## Decisions

| Decision | Reason |
|---|---|
| Use a clean repository | The prototype's contracts, documentation, and Git state disagree. |
| Use three branches | Audio spoofing is a direct cue. Emotion lacks enough labeled training data. |
| Train sync on authentic pairs | The global fake label does not mean every fake has broken lip sync. |
| Preserve temporal tokens | Attention over one pooled token cannot measure alignment. |
| Use late fusion first | It is easier to audit and less likely to overfit the available data. |
| Abstain on missing evidence | Silent full-frame crops and inner joins bias evaluation. |
| Keep the dashboard thin | The research claim depends on evidence, not interface complexity. |

## Exit criteria

The project is complete when a clean environment can reproduce the frozen split, one training smoke run, feature export, fusion, evaluation, and one-video inference.

Each branch must beat its cue-specific label-prior baseline on validation data. The sync offset confidence interval must exclude chance. Fusion does not need to beat the visual model. If it fails, report the negative result and its confidence interval.
