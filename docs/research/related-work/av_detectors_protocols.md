# Audio-visual deepfake detectors 2023 to Nov 2026: protocols, cross-dataset results, code

Research date: 2026-11-13. All numbers below were read from the paper page cited next to them (arXiv HTML or project/GitHub page). Where a fetch gave an ambiguous table layout, the note says so. Format: fact, then (source link).

## Q1. Per-method facts: data, test sets, cross-dataset AUC, shortcut control, code, licence

### Takeaway
Almost every 2024 to 2026 AV detector trains on a random 70/30 FakeAVCeleb split and reports cross-dataset AUC on KoDF (subset), DFDC (subset) or its own new dataset. Only the Bucharest group (AVH-Align, Boldisor et al.) tests the leading-silence shortcut. Reproducible with official code and weights: AVH-Align, HAVIC, X-AVDT, AVAD (inference only), RealForensics, LipForensics, SpeechForensics. AVFF has no official code.

### Comparison table

| Method (venue) | Train data | Test sets | Key cross-dataset AUC (table) | Shortcut control | Code / weights / licence |
|---|---|---|---|---|---|
| AVAD, Feng, Chen, Owens (CVPR 2023) | Real only (LRS2 or LRS3) | FAVC, KoDF subset | KoDF 86.9 (AVFF Table 3); FAVC cross-manip AVG-FV 94.5 (LRS2) / 92.8 (LRS3) (AVFF Table 2); AV1M val 54.3 (AVH-Align Table 2) | None | github.com/cfeng16/audio-visual-forensics, MIT; inference checkpoints yes, training code not evident |
| AVFF, Oorloff et al. (CVPR 2024) | Stage 1 LRS3 real; Stage 2 FAVC 70/30 | FAVC, KoDF subset, DF-TIMIT HQ, DFDC subset (3,215) | KoDF 95.5; DFDC 86.2; DF-TIMIT 100.0 (Tables 3, 5) | None (codec only as unseen perturbation) | No official code. Unofficial: github.com/JoeLeelyf/OpenAVFF (cited by Boldisor et al.), licence not stated |
| AVH-Align, Smeu et al. (CVPR 2025) | Real only: 45k AV1M real (VoxCeleb2 origin) | FAVC (70/30), AV1M val 10k, AV1M official test | AV1M val 85.9 untrimmed / 83.5 trimmed; FAVC 94.6 / 94.6 (Table 2); AV1M official test 85.24 (Table 3) | Yes: finds leading silence (25 to 30 ms) in FAVC and AV1M; reports trimmed results | github.com/bit-ml/AVH-Align, CC BY-NC-SA 4.0; checkpoint AVH-Align_AV1M.pt |
| AVH-Align/sup (same paper, supervised) | AV1M or FAVC real+fake | FAVC, AV1M | FAVC->AV1M 69.0 / 63.6; AV1M->FAVC 77.5 / 70.8 (Table 2) | Shows supervised drops from 100.0 to 83.1 on trimmed AV1M | same repo |
| SpeechForensics, Liang et al. (NeurIPS 2024) | Real only, AV-HuBERT features | (from AVH-Align) | FAVC 98.8; AV1M val 68.8 / 68.2 trimmed (AVH-Align Table 2) | Not in own paper (not read) | github.com/Eleven4AI/SpeechForensics; licence not checked |
| Anshul et al. intra/cross-modal sync (ICCV 2025) | Two stage: sync pretraining on real videos, then detection | Not read (CVF PDF returned 403) | Not read | Not known | No code link found on author page |
| Anshul et al. next-frame prediction (arXiv 2511.10212, follow-up) | FAVC 70/30 | FAVC, KoDF, LAV-DF (localization), CREMA | KoDF 100 (Table III); FAVC cross-manip AVG-FV 95.6 (Table II) | None stated | No code URL |
| HAVIC "Leave No Stone Unturned", Peng et al. (CVPR 2026 Findings) | Pretrain LRS2 real; finetune FAVC 70/30 | KoDF subset, HiFi-AVDF (new, 3,810 videos) | KoDF AUC 98.9 / AP 99.2 (Table 3); HiFi-AVDF avg AUC 75.77 (Table 4) | None described | github.com/tuffy-studio/HAVIC, MIT; finetuned weights at huggingface.co/JielunPeng/HAVIC; split script has no fixed seed |
| X-AVDT, Kim et al. (CVPR 2026) | MMDF (new): Hallo2, LivePortrait, FaceAdapter fakes | MMDF unseen generators, FAVC, FF++ | FAVC 99.69; FF++ 89.55 (Table 5); MMDF unseen avg 95.29 (Table 4) | None described | github.com/youngseo0526/X-AVDT, no licence file; weights via Google Drive; data huggingface.co/datasets/zaqxsw0526/MMDF |
| SAVe, Shahzad et al. (arXiv 2603.25140, 2026) | Real only, FAVC or LipSyncTIMIT | FAVC, LipSyncTIMIT (orig, c23, c40) | LipSyncTIMIT->FAVC AVSync branch 0.8462 (Table IV); fused 0.9633 / 0.9688 / 0.7680 (Table V, column meaning ambiguous) | Compression studied, no silence control | No code found |
| T-AVFD, Liu et al. (arXiv 2605.27944, 2026) | Real talking videos only | AVLips, FAVC (500 real + 1,000 fake), TalkingHeadBench, SHDF (singing, new) | FAVC 95.6; AVLips 87.7; THB 93.0; SHDF 80.2 (Table 2) | None | Project page only (LiuKe3068LikWix.github.io/SingingHead-DeepFake), no code |
| CAD, Du et al. (arXiv 2505.15233) | IDForge-v2; FAVC 70/30 | FAVC, IDForge, FaceShifter, Celeb-DF | IDForge-v2->FaceShifter 99.6, ->Celeb-DF 94.2 (Tables 4, 5); FAVC cross-manip AVG-FV 100 (Table 3) | None | No code URL |
| Boldisor et al., SSL representations (CVPR 2026) | FAVC, AV1M, AVLips, DFE-2024 | Cross pairs among these | Table 2 (layout unclear in fetch, see Gaps) | Analyses leading silence as a learned cue | Project page bit-ml.github.io/ssr-dfd; repo not confirmed |
| Local temporal inconsistencies (arXiv 2501.08137) | DFDC subset (15,300 train) | DFDC, FAVC (70 real + 70 fake) | DFDC->FAVC 87.0 (Table 4) | None | No code URL |
| RealForensics, Haliassos et al. (CVPR 2022, pre-2023 baseline) | Stage 1 LRW; Stage 2 FF++ | many | KoDF 93.6, DFDC 83.7 (AVFF Tables 3, 5); FAVC->HiFi-AVDF 61.91 (HAVIC Table 4) | None | github.com/ahaliassos/RealForensics, MIT; weights + training code |
| LipForensics, Haliassos et al. (CVPR 2021, baseline) | FF++ | many | KoDF 86.6, DFDC 77.4 (AVFF Tables 3, 5) | None | github.com/ahaliassos/LipForensics, MIT; FF++ weights, no training script |
| Mamba-AVGuard (2026) | Not found | | | | Not found |

### Cited Findings
- AVH-Align trains on 45k real AV1M samples and uses a 70/30 FAVC split; AV1M official test uses different speakers ([AVH-Align arXiv](https://arxiv.org/html/2412.00175v3), Sec. 4.2).
- Table 2 of AVH-Align (untrimmed / trimmed AUC): AVH-Align FAVC 94.6/94.6, AV1M 85.9/83.5; AVAD 84.5/84.7 and 54.3/54.3; SpeechForensics 98.8/98.8 and 68.8/68.2; AVH-Align/sup FAVC->AV1M 69.0/63.6, AV1M->FAVC 77.5/70.8 ([AVH-Align arXiv](https://arxiv.org/html/2412.00175v3)).
- Silence-only classifier reaches AUC 98.4 on FAVC and 98.2 on AV1M, falling to 54.8 and 50.6 after trimming. LAV-DF has a trailing-silence AUC of about 65%. AVLips and DFDC show none of the three biases (Sec. 3.2, Table 1) ([AVH-Align arXiv](https://arxiv.org/html/2412.00175v3)).
- AV1M official test AUC: AVH-Align 85.24, AVH-Align/sup 99.90, silence classifier 98.44 (Table 3) ([AVH-Align arXiv](https://arxiv.org/html/2412.00175v3)).
- AVH-Align code is CC BY-NC-SA 4.0 with a checkpoint trained on 45,000 AV1M real videos ([GitHub bit-ml/AVH-Align](https://github.com/bit-ml/AVH-Align)).
- AVFF: Stage 1 on LRS3 real, Stage 2 on FAVC 70/30 following AVoiD-DF; FAVC intra 98.6 ACC / 99.1 AUC (Table 1); KoDF AUC 95.5 vs RealForensics 93.6, AVAD 86.9, LipForensics 86.6 (Table 3); DFDC 86.2 vs RealForensics 83.7 (Table 5); no code link ([AVFF arXiv](https://arxiv.org/html/2406.02951v1)).
- AVFF cross-manipulation AVG-FV AUC: AVFF 99.5, RealForensics 97.1, AVAD (LRS2) 94.5, LipForensics 91.1; RVFA: AVFF 92.4, AVAD (LRS3) 80.5 (Table 2) ([AVFF arXiv](https://arxiv.org/html/2406.02951v1)).
- HAVIC pretrains on LRS2 real, finetunes on FAVC 70/30, tests on KoDF subset and HiFi-AVDF. KoDF AUC: HAVIC 98.9, AVFF 95.5, PIA 95.0, RealForensics 93.6, AVH-Align 90.1, AVAD 86.9 (Table 3). HiFi-AVDF avg AUC: HAVIC 75.77, PIA 66.40, AVFF 65.28, Effort 64.20, LipFD 62.31, RealForensics 61.91, AVAD 61.73, AVH-Align 58.32 (Table 4) ([HAVIC arXiv](https://arxiv.org/html/2603.23960v1)).
- HAVIC repo is MIT, finetuned weights on Hugging Face, FAVC split made by a random script ([GitHub tuffy-studio/HAVIC](https://github.com/tuffy-studio/HAVIC)). Venue: CVPR 2026 Findings ([CVPR poster](https://cvpr.thecvf.com/virtual/2026/poster/41156)).
- X-AVDT trains on MMDF (Hallo2, LivePortrait, FaceAdapter). MMDF unseen-generator avg AUROC: X-AVDT 95.29, RealForensics retrained 92.42, AVH-Align retrained 81.44 (Table 4). MMDF->FAVC AUROC: X-AVDT 99.69, RealForensics retrained 83.67, AVH-Align retrained 54.20, LipFD retrained 53.59; official LipForensics checkpoint 98.40 (Table 5) ([X-AVDT arXiv](https://arxiv.org/html/2603.08483v1)).
- X-AVDT code at github.com/youngseo0526/X-AVDT, no licence file, detector weights on Google Drive ([GitHub](https://github.com/youngseo0526/X-AVDT); [project page](https://youngseo0526.github.io/X-AVDT)).
- SAVe trains only on real videos of FAVC or LipSyncTIMIT. LipSyncTIMIT->FAVC AVSync AUC 0.8462 (Table IV). Table V lists the fused model at 0.9945 (FakeAVCeleb-LS column) and AVAD at 0.94 / 0.96 / 0.94 / 0.81; column meaning is unclear. No code link ([SAVe arXiv](https://arxiv.org/html/2603.25140v1)).
- T-AVFD (in "From Talking to Singing") trains on real talking samples only. Table 2 AUC on FAVC: T-AVFD 95.6, AVH-Align 93.0, RealForensics 88.2, AVAD 84.8, LipFD 77.0; on SHDF: T-AVFD 80.2, AVH-Align 37.4 ([T-AVFD arXiv](https://arxiv.org/html/2605.27944v1)).
- CAD uses FAVC 70/30 following AVoiD-DF and AVFF; IDForge-v2->FaceShifter 99.6 and ->Celeb-DF 94.2 (Tables 4, 5); no code URL ([CAD arXiv](https://arxiv.org/html/2505.15233)).
- Anshul et al. ICCV 2025: two-stage framework that learns intra- and cross-modal sync on real videos (DOI 10.1109/ICCV51701.2025.01283) ([ICCV poster](https://iccv.thecvf.com/virtual/2025/poster/220); [ML Anthology](https://mlanthology.org/iccv/2025/anshul2025iccv-intramodal/)). The follow-up 2511.10212 reports KoDF AUC 100 and FAVC intra AUC 96.3 (Tables I, III), 70/30 split, no code ([arXiv 2511.10212](https://arxiv.org/html/2511.10212v1)).
- Local temporal inconsistencies paper: DFDC->FAVC AUC 87.0 vs FGI 84.5, AVoiD-DF 82.8 (Table 4) ([arXiv 2501.08137](https://arxiv.org/html/2501.08137v3)).
- AVAD repo is MIT with inference checkpoints; sync_model.pth on Google Drive; no training script visible ([GitHub cfeng16/audio-visual-forensics](https://github.com/cfeng16/audio-visual-forensics)). AVH-Align states AVAD training code is not available ([AVH-Align arXiv](https://arxiv.org/html/2412.00175v3)).
- RealForensics: MIT, Stage 2 weights and training code public ([GitHub](https://github.com/ahaliassos/RealForensics)). LipForensics: MIT, FF++ weights, no training script ([GitHub](https://github.com/ahaliassos/LipForensics)).
- SpeechForensics (NeurIPS 2024) code at github.com/Eleven4AI/SpeechForensics ([arXiv 2508.09913](https://arxiv.org/html/2508.09913v1); [ML Anthology](https://mlanthology.org/neurips/2024/liang2024neurips-speechforensics)).
- Boldisor et al. (CVPR 2026) find that no SSL feature generalizes reliably across deepfake datasets, and that audio models attend to leading silence ([arXiv 2511.17181v2](https://arxiv.org/html/2511.17181v2); [CVPR poster](https://cvpr.thecvf.com/virtual/2026/poster/37119)).
- Other 2026 AV detectors found but not read in full: AMDD reports 99.8 AUC on FAVC ([arXiv 2604.26453](https://arxiv.org/abs/2604.26453)); AV-LMMDetect on Qwen 2.5 Omni ([arXiv 2602.23393](https://arxiv.org/html/2602.23393v1)); DATS-AV claims AV1M->LAV-DF AUC 0.979 ([ResearchGate](https://www.researchgate.net/publication/404975785_DATS-AV_A_Dissonance-Aware_Two-Stage_Framework_for_Audio-Visual_Deepfake_Detection)).

### Inferences
- Reproducible baselines with official code and weights: AVH-Align (non-commercial licence), HAVIC (MIT, weights), X-AVDT (no licence, so legally unclear), RealForensics (MIT), LipForensics (MIT, inference only), AVAD (MIT, inference only), SpeechForensics (licence unchecked). AVFF needs the unofficial OpenAVFF port, so any AVFF number from a rerun is not the paper's number.
- KoDF is the de facto cross-dataset test for FAVC-trained models, but every paper uses "a KoDF subset following AVAD", so the subset itself is the protocol to copy.
- The HiFi-AVDF and MMDF results show that methods near 95+ on KoDF fall to 58 to 76 AUC on modern generators. That gap is where a new paper can claim novelty.
- None of the supervised methods (AVFF, HAVIC, X-AVDT, CAD, SAVe) report silence-trimmed results, so their FAVC numbers may include the leading-silence shortcut that AVH-Align measured at 98.4 AUC alone.

### Gaps
- Mamba-AVGuard: no paper by that name found on arXiv or in search results. It may not exist under this name.
- Anshul et al. ICCV 2025 tables: CVF PDF returned HTTP 403; numbers not read.
- AVAD's own paper (Feng et al. CVPR 2023) not fetched; its numbers above come from AVFF and AVH-Align tables.
- Boldisor et al. Table 2: fetch suggested AV-HuBERT AV1M->FAVC 100 and FAVC->AV1M 99.0, but the column layout was ambiguous and these values look implausible next to AVH-Align's Table 2. Verify against the PDF before citing.
- SAVe Table V column meanings are unclear.
- SpeechForensics own paper numbers and licence not read. X-AVDT appendix (FAVC split) not read.
- DeepSpeak: no detector in this set reports results on it.

## Q2. Identity-disjoint or official FakeAVCeleb splits

### Takeaway
None of the papers read uses an identity-disjoint FakeAVCeleb split. FakeAVCeleb has no official split; the community standard is a random 70/30 split inherited from AVoiD-DF. AV1M (official test with unseen speakers) and LAV-DF (identity-independent splits) are the datasets with official identity-disjoint splits.

### Cited Findings
- AVFF uses a 70/30 split "following AVoiD-DF" and does not state identity disjointness ([AVFF arXiv](https://arxiv.org/html/2406.02951v1)).
- CAD uses a random 70/30 split following AVoiD-DF and AVFF ([CAD arXiv](https://arxiv.org/html/2505.15233)).
- HAVIC uses 70/30 "following prior works"; the repo script splits randomly with no stated seed ([HAVIC arXiv](https://arxiv.org/html/2603.23960v1); [GitHub](https://github.com/tuffy-studio/HAVIC)).
- AVH-Align uses 70/30 on FAVC and says the AV1M official test has a different set of speakers ([AVH-Align arXiv](https://arxiv.org/html/2412.00175v3)).
- Boldisor et al. state FAVC has no predefined split and use 63/7/30, keeping only RVRA and FVFA samples ([arXiv 2511.17181v2](https://arxiv.org/html/2511.17181v2)).
- Anshul et al. follow-up uses a 70/30 split ([arXiv 2511.10212](https://arxiv.org/html/2511.10212v1)).
- T-AVFD tests on 500 real + 1,000 selected FAVC fakes; 2501.08137 tests on 70 real + 70 fake ([T-AVFD](https://arxiv.org/html/2605.27944v1); [2501.08137](https://arxiv.org/html/2501.08137v3)).

### Inferences
- With only 500 real FAVC videos, a random 70/30 split puts the same identities in train and test. FAVC in-domain AUCs of 99+ (AVFF 99.1, HAVIC 99.9, AMDD 99.8) are likely inflated by identity overlap plus the silence shortcut. An identity-disjoint FAVC protocol would be a real contribution, but no prior number exists to compare against on it.

### Gaps
- Did not find any paper that publishes identity-disjoint FAVC split files.

## Q3. Best reported cross-dataset AUC

### Takeaway
FAVC->other: best is HAVIC 98.9 on KoDF (the 100 on KoDF from arXiv 2511.10212 is unreviewed and has no code). On harder modern sets the best is HAVIC 75.77 on HiFi-AVDF and T-AVFD 80.2 on SHDF. AV1M->other: AVH-Align (real-only AV1M) reaches 94.6 on FAVC; supervised AV1M->FAVC drops to 77.5 (70.8 trimmed).

### Cited Findings
- FAVC->KoDF AUC: 2511.10212 100 ([arXiv](https://arxiv.org/html/2511.10212v1)); HAVIC 98.9 ([arXiv](https://arxiv.org/html/2603.23960v1)); AVFF 95.5 ([arXiv](https://arxiv.org/html/2406.02951v1)).
- FAVC->DFDC: AVFF 86.2 ([arXiv](https://arxiv.org/html/2406.02951v1)).
- FAVC->HiFi-AVDF: HAVIC 75.77 ([arXiv](https://arxiv.org/html/2603.23960v1)).
- FAVC->AV1M: AVH-Align/sup 69.0 (63.6 trimmed) ([arXiv](https://arxiv.org/html/2412.00175v3)).
- AV1M (real only)->FAVC: AVH-Align 94.6; AV1M (supervised)->FAVC: 77.5 / 70.8 trimmed ([arXiv](https://arxiv.org/html/2412.00175v3)).
- AV1M->LAV-DF: DATS-AV claims 0.979 AUC; an MMMS-BA 2024 preprint reports 0.968 ([ResearchGate DATS-AV](https://www.researchgate.net/publication/404975785_DATS-AV_A_Dissonance-Aware_Two-Stage_Framework_for_Audio-Visual_Deepfake_Detection); [arXiv 2408.01532](https://arxiv.org/html/2408.01532v2)). Not read in full.
- Real-talking-only->FAVC: T-AVFD 95.6 ([arXiv](https://arxiv.org/html/2605.27944v1)). MMDF->FAVC: X-AVDT 99.69 ([arXiv](https://arxiv.org/html/2603.08483v1)).

### Inferences
- LAV-DF->other numbers are rare. AV1M->LAV-DF is near-same-source (both VoxCeleb2-based), so it is a weak generalization test.

### Gaps
- No LAV-DF-trained ->FAVC or ->KoDF AUC found in the papers read.
- DATS-AV and MMMS-BA numbers not verified against their tables.
