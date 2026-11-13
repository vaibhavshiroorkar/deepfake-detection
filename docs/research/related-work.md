# AV deepfake novelty check

Checked 2026-11-13. Source notes are in [related-work/](related-work/). Each entry
there carries its URL and the table its number came from. Numbers below are as the papers
report them. We have not reproduced them.

## Verdicts

| Objective | Verdict | Claim we can make |
|---|---|---|
| O1 three-path fusion vs visual-only | Partially covered | Not new as a method. New as a measurement: does fusion still beat visual-only after the codec, silence and identity shortcuts are removed from train and test? |
| O2 which-modality attribution | Partially covered | The head is not new. Attribution that transfers between FakeAVCeleb and LAV-DF, under shortcut control, is unreported. So is the finding that clip-level AUC hides what per-modality labels reveal. |
| O3 real-only conformal FPR control | Partially covered | The method is standard, and concurrent work does it for images and speech. Audio-visual video under codec re-encoding is untested, as are real-clip sample counts and the per-modality TPR cost. |
| O4 shortcut-neutral protocol | Most open | No one has released neutralized manifests. No one has published a per-class codec census for AV datasets, a trailing-silence fix, or an identity probe. |

O4 is the strongest backbone for the paper. O1, O2 and O3 are measured inside it.

## O1: fusion under shortcut control

- **AVH-Align** (Smeu et al., CVPR 2025, https://arxiv.org/abs/2412.00175). It trains AV-HuBERT alignment on real clips only. A leading-silence classifier alone reaches 98.4 AUC on FakeAVCeleb and 98.2 on AV-Deepfake1M. The real-only model loses almost nothing when the silence is trimmed. Code is at github.com/bit-ml/AVH-Align, licence CC BY-NC-SA 4.0. Its checkpoint was trained on 45k real AV-Deepfake1M videos.
- **AVFF** (CVPR 2024) reports FakeAVCeleb to DFDC 86.2 and to KoDF 95.5. There is no official code, only an unofficial port. It has no shortcut control.
- **HAVIC** (Peng et al., CVPR 2026 Findings, github.com/tuffy-studio/HAVIC, MIT). It reports FakeAVCeleb to KoDF 98.9, but 75.77 on HiFi-AVDF. Its FakeAVCeleb split is random with no fixed seed.
- **X-AVDT** (CVPR 2026). Weights are public, but the repo has no licence file.
- **Protocol gap.** No paper we read uses an identity-disjoint FakeAVCeleb split. The common split is a random 70/30 inherited from AVoiD-DF. FakeAVCeleb has only about 500 real videos, so identities almost certainly cross that split.

Baselines we can reproduce: AVH-Align, HAVIC, RealForensics (MIT, with training code), plus LipForensics and AVAD (inference only).

## O2: attribution

- **Yu et al.**, modality-agnostic dual-label detection (arXiv 2307.14491, IH&MMSec 2025). Closest prior work. Its cross-dataset tests run DFDC to FakeAVCeleb and back only. Audio accuracy falls to 48.89 there.
- **Multimodaltrace** (CVPRW 2023). A multi-label head on FakeAVCeleb. Its cross-dataset tests report only binary accuracy.
- **MRDF** (ICASSP 2024). It shows clip-level labels harm fused learning, in-dataset only: 30% versus 12.6 to 14% error on real-audio/fake-video clips.
- **Naming clash.** AMDD (2026) uses "attribution" to mean naming the generator. Call our head "manipulated-modality prediction".

## O3: false-alarm control

- **Concurrent work:** Mahdaoui (Zenodo preregistration, 1 Nov 2026) covers real-photo recalibration for still images under camera-processing shift. PGA-G (Zenodo, Oct 2026) sets conformal operating points from real speech.
- **Already established:** thresholds fail to transfer even when AUC is high (Nguyen et al. 2026; LAION-Mobile 2026). Do not claim this as a finding.
- **Standard method:** Bates et al. (Annals of Statistics 2023) and the Neyman-Pearson umbrella method (Tong et al. 2018).
- **Still open:** AV video, H.264 and H.265 re-encoding, how many real clips are needed for a given (alpha, delta), and the TPR cost per modality.

## O4: documented shortcuts

| Shortcut | Evidence | Status |
|---|---|---|
| Leading silence, FakeAVCeleb and AV-Deepfake1M | 98.4 and 98.2 AUC from silence alone (Smeu 2025) | Documented. Fixed only at test time. |
| Trailing silence | Over 99 AUC on FakeAVCeleb, 65 on LAV-DF (Smeu 2025) | Documented. Never neutralized. |
| Silence in ASVspoof 2019 LA | 15.1% EER from duration alone (Mueller 2021) | Documented |
| Codec or container per class | Our own measurement: 0.9985 AUC from metadata on FakeAVCeleb | Unpublished anywhere |
| Wav2Lip leak in the leave-one-out protocol | DeepFake Doctor (arXiv 2506.05851) | Documented |
| Identity leakage in AV detectors | Not measured | Open |
| Speaker overlap with AV-HuBERT pretraining | Not measured | Open |

## Gaps in this check

- We could not read the Anshul et al. ICCV 2025 PDF (403).
- We found no paper called Mamba-AVGuard.
- We opened the PGA-G manuscript and the Mahdaoui results only at record level.
- Some tables in Boldisor et al. 2026 look wrong. Check the PDF before citing them.
