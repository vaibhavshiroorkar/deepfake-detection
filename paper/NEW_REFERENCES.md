# NEW_REFERENCES

Written 2026-10-25 by the paper stream. For the main agent to merge into docs/RESEARCH.md. Part A lists sources that are NOT in RESEARCH.md and were verified here from the publisher, the proceedings page, the DOI record (Crossref) or arXiv. The abstracts and first pages were NOT read for these (status NR for content): they are cited only for what the title, venue and the published abstract line state (what the encoder or detector is, that it exists, its venue). Part B lists venue corrections for papers already in RESEARCH.md, found in the same way. Part C is a discrepancy to settle.

Verification tools: HTTP fetch of the proceedings or publisher page, the Crossref REST API (api.crossref.org, record fields title, authors, container, volume, issue, pages, date, DOI) and the arXiv abstract page (comments and journal-reference fields). DBLP and OpenReview were not reachable (bot protection).

## A. New sources (key in paper/refs.bib)

| Key | Author, title | Venue, year, pages | Verified from |
|---|---|---|---|
| oquab2024dinov2 | Maxime Oquab, Timothee Darcet, Theo Moutakanni and 23 co-authors. DINOv2: Learning Robust Visual Features without Supervision | Transactions on Machine Learning Research, 2024 | ML Anthology record (mlanthology.org/tmlr/2024/oquab2024tmlr-dinov2), a web search that returned the HAL record "Transactions on Machine Learning Research, 2024", and the arXiv abstract page (arXiv:2304.07193v2, authors). The OpenReview page was not reachable. No page range (journal without pages) |
| radford2021clip | Alec Radford, Jong Wook Kim, Chris Hallacy and 9 co-authors. Learning Transferable Visual Models From Natural Language Supervision | Proc. 38th ICML, PMLR 139, 2021, pp. 8748-8763 | https://proceedings.mlr.press/v139/radford21a.html |
| zhai2023siglip | Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, Lucas Beyer. Sigmoid Loss for Language Image Pre-Training | Proc. IEEE/CVF ICCV, 2023, pp. 11975-11986 | CVF open access page, ICCV 2023 |
| tan2019efficientnet | Mingxing Tan, Quoc V. Le. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks | Proc. 36th ICML, PMLR 97, 2019, pp. 6105-6114 | https://proceedings.mlr.press/v97/tan19a.html |
| chollet2017xception | Francois Chollet. Xception: Deep Learning With Depthwise Separable Convolutions | Proc. IEEE CVPR, 2017, pp. 1251-1258 | CVF open access page, CVPR 2017 |
| baevski2020wav2vec2 | Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, Michael Auli. wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations | Advances in Neural Information Processing Systems 33 (NeurIPS 2020) | proceedings.neurips.cc abstract page. Page range not printed there, so none is given in the bib |
| hsu2021hubert | Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai and 3 co-authors. HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units | IEEE/ACM Trans. Audio, Speech, and Language Processing, vol. 29, 2021, pp. 3451-3460, doi 10.1109/TASLP.2021.3122291 | Crossref record |
| chen2022wavlm | Sanyuan Chen, Chengyi Wang, Zhengyang Chen and 16 co-authors. WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing | IEEE J. Selected Topics in Signal Processing, vol. 16, no. 6, 2022, pp. 1505-1518, doi 10.1109/JSTSP.2022.3188113 | Crossref record |
| radford2023whisper | Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, Ilya Sutskever. Robust Speech Recognition via Large-Scale Weak Supervision | Proc. 40th ICML, PMLR 202, 2023, pp. 28492-28518 | https://proceedings.mlr.press/v202/radford23a.html (the page prints "Mcleavey"; the arXiv listing and the authors' usage give McLeavey) |
| hu2022lora | Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen. LoRA: Low-Rank Adaptation of Large Language Models | ICLR 2022 | iclr.cc virtual poster page 6319 (title ICLR 2022 Poster, authors) and arXiv:2106.09685. RESEARCH.md R-036 already lists the paper (venue was "to verify"; now ICLR 2022) |
| wu2023yunet | Wei Wu, Hanyang Peng, Shiqi Yu. YuNet: A Tiny Millisecond-level Face Detector | Machine Intelligence Research, vol. 20, no. 5, 2023, pp. 656-665, doi 10.1007/s11633-023-1423-y | Crossref record (authors with given names, journal, volume, issue, pages, date 2023-04-19) |

## B. Venue corrections for papers already in RESEARCH.md

| RESEARCH.md ID | Key | Result |
|---|---|---|
| R-001 FaceForensics++ | rossler2019faceforensics | ICCV 2019, pp. 1-11 (CVF page) |
| R-002 Celeb-DF | li2020celebdf | CVPR 2020, pp. 3207-3216 (CVF page) |
| R-003 DFDC | dolhansky2020dfdc | no venue found; cited as arXiv:2006.07397 |
| R-004 Face X-ray | li2020facexray | CVPR 2020, pp. 5001-5010 (CVF page) |
| R-005 FakeCatcher | ciftci2020fakecatcher | IEEE TPAMI, doi 10.1109/TPAMI.2020.3009287; the arXiv comment says "To appear in IEEE TPAMI, accepted July 2020"; Crossref lists the record without volume or pages (early access), so none is given |
| R-006 Hearts of Deep Fakes | ciftci2020hearts | IEEE IJCB 2020, pp. 1-10, doi 10.1109/IJCB48548.2020.9304909 (Crossref) |
| R-007 DeepFakesON-Phys | hernandez2020deepfakeson | the arXiv record carries the journal reference "Proc. 35th AAAI Conference on Artificial Intelligence Workshops, 2021"; cited so, without pages |
| R-010 Lin et al. | lin2023ppg | EUSIPCO 2023, pp. 615-619 (Crossref; the paper number 615 equals the first page) |
| R-011 Ojha et al. | ojha2024universal | CVPR 2023, pp. 24480-24489 (CVF page). The key keeps "2024" (the arXiv v2 year the file was read in); the entry carries 2023 |
| R-012 AASIST | jung2021aasist | ICASSP 2022, pp. 6367-6371 (Crossref). Key keeps 2021 (arXiv year) |
| R-013 ASVspoof 2019 | wang2020asvspoof | Computer Speech and Language, vol. 64, art. 101114, 2020 (Crossref) |
| R-014 FakeAVCeleb | khalid2022fakeavceleb | NeurIPS 2021 Datasets and Benchmarks Track (arXiv comment field). Key keeps 2022 (arXiv v4) |
| R-015 Not made for each other | chugh2021notmade | Proc. 28th ACM International Conference on Multimedia, 2020, pp. 439-447, doi 10.1145/3394171.3413700 (Crossref). Key keeps 2021 (arXiv v3) |
| R-016 Guo et al. | guo2017calibration | ICML 2017, PMLR 70, pp. 1321-1330 (PMLR page) |
| R-017 DeepfakeBench | yan2023deepfakebench | NeurIPS 2023 Datasets and Benchmarks Track, Advances in Neural Information Processing Systems 36, pp. 4534-4565 (Crossref record; the project README title line reads "NeurIPS 2023 D&B"). See part C |
| R-018 Kornblith et al. | kornblith2019better | CVPR 2019, pp. 2661-2671 (CVF page) |
| R-024 LAV-DF | cai2022lavdf | DICTA 2022, doi record "2022 International Conference on Digital Image Computing: Techniques and Applications (DICTA)", pp. 1-10 (Crossref); the arXiv comment field says "DICTA 2022" |
| R-029 Tak et al. | tak2022wav2vecspoof | Speaker and Language Recognition Workshop (Odyssey 2022), pp. 112-119, doi 10.21437/Odyssey.2022-16 (Crossref) |
| R-033 Muller et al. | muller2022inthewild | Interspeech 2022, pp. 2783-2787 (Crossref). Author spelling: Crossref prints "Franziska Diekmann", RESEARCH.md (from the file) prints "Dieckmann"; the bib follows the publisher record |
| R-044 Dong et al. | dong2023implicit | CVPR 2023, pp. 3994-4004 (CVF page and Crossref) |
| R-045 Cozzolino et al. | cozzolino2024raising | CVPR Workshops 2024, pp. 4356-4366, doi 10.1109/CVPRW63382.2024.00439 (Crossref) |
| R-047 Wang et al. | wang2020cnnspot | CVPR 2020, pp. 8695-8704 (CVF page) |
| R-053 Celeb-DF++ | li2025celebdfpp | no venue found; cited as arXiv:2507.18015 |
| R-057 POS / rPPG principles | wang2017pos | IEEE Transactions on Biomedical Engineering, vol. 64, no. 7, pp. 1479-1491, July 2017, doi 10.1109/TBME.2016.2609282 (Crossref) |

## C. Discrepancy to settle

The task text called DeepfakeBench "ICML 2023". The evidence found says NeurIPS 2023, Datasets and Benchmarks Track: the paper's own PDF (arXiv v2) as recorded in R-017, the project README ("NeurIPS 2023 D&B") and the Crossref record (Advances in Neural Information Processing Systems 36). The paper uses NeurIPS 2023.

## D. Check of the two quoted DeepfakeBench numbers

arXiv:2307.01426v2 (28 Oct 2023), PDF page 6, Table 3 ("Within-domain and cross-domain evaluations using the AUC metric. All detectors are trained on FF-c23 and evaluated on other data"), row "Naive, Xception, backbone Xception": within-domain Avg 0.9450 (columns FF-c23, FF-c40, FF-DF, FF-F2F, FF-FS, FF-NT: 0.9637, 0.8261, 0.9799, 0.9785, 0.9833, 0.9385) and cross-domain Avg 0.7718 (eight columns CDFv1, CDFv2, DF-1.0, DFD, DFDC, DFDCP, Fsh, UADFV: 0.7794, 0.7365, 0.8341, 0.8163, 0.7077, 0.7374, 0.6249, 0.9379). The text of section 4.2 quotes 94.50 percent for Xception within-domain. Protocol (section 4.1): frame-level AUC, face crops from DLIB aligned to 256 by 256, 32 frames per video, Adam with learning rate 0.0002, batch 32, each entry the average of the top three values of the metric. The same cross-domain average, 0.7718, is also printed for the EfficientNet-B4 row. Both numbers are confirmed.
