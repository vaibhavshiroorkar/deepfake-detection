# P-002: choosing the visual and audio backbones

A decision record for the [P-002 design](P-002-design.md). Checked on 2026-11-16. Each cell is labelled **verified** (read in the cited source), **assumed** (believed, not checked) or **unknown**.

## Criteria, set before looking at options

| Criterion | Weight | Why |
|---|---|---|
| Cross-dataset evidence, measured in published work | High | The project's whole problem is transfer to unseen data |
| Fits our GPU (16 GB) for training or caching | High | One RTX 5070 Ti |
| Training cost | Medium | Hours, not days, per seed |
| Licence allows research use | Must | Research only, no commercial use planned |
| Available as weights we can load | Must | No reimplementation from scratch |

One caution applies to every published number below. The benchmarks were scored on raw files, with the file shortcuts this project measured still in them, so part of each score may be shortcut. P-002 must re-score the chosen backbone on neutralized copies before relying on it.

## Visual path

| Option | Cross-dataset evidence | Fits 16 GB | Cost | Licence |
|---|---|---|---|---|
| **CLIP ViT-L/14, LayerNorm-only tuning (LNCLIP-DF)** | Verified: trained on FF++ c23 only, mean video AUROC 94.2 over 18 benchmarks (DFDC 87.0, KoDF 85.9, FakeAVCeleb 96.3, DeepSpeak v2 80.0) [1] | Assumed: about 0.3B parameters in the image encoder | Verified: "a few hours" on one A100 [1] | Assumed: MIT (OpenAI CLIP) |
| DINOv3 ViT-L/16, frozen, linear probe | Verified, images only: face-image accuracy 0.32 to 0.91 across four shifted sets, mean 0.661 [2] | Assumed | Assumed: low, probe only | Unknown: DINOv3 licence terms not checked |
| C-RADIOv4-H, frozen, linear probe | Verified, images only: most resilient of three frozen backbones, mean accuracy 0.718 [2] | Unknown: a huge model | Assumed: low, probe only | Unknown: NVIDIA licence not checked |
| EfficientNet-B0 with GRU (the current P-001 path, the simplest option) | Verified here: about 0.99 in-domain on FakeAVCeleb; cross-corpus result arrives with H2 | Verified: trains here | Verified: about 30 min per seed here | Verified: in use |

**Pick: CLIP ViT-L/14 with LayerNorm-only tuning.** It is the only option with strong video-level, cross-dataset evidence, including two of our corpora, and it is cheap to train. Training on paired real and fake clips of the same source mattered in that work [1], and LAV-DF and HiFi-AVDF give us such pairs. **Runner-up:** a frozen DINOv3 or C-RADIOv4 probe; the evidence for these is on still images only. **What would change the pick:** if CLIP's score on our neutralized copies drops far below its raw-file score while a DINOv3 probe holds up.

## Audio path

Evidence from one controlled benchmark: 20 frozen speech front-ends, each with the same light classifier, trained on ASVspoof 2019 LA and tested elsewhere. The table gives EER in %, lower is better [3].

| Option | In-the-Wild EER | Mean EER, 4 sets | Licence |
|---|---|---|---|
| **XLS-R** | Verified: 20.1 | Verified: 17.4, the best | Assumed: Apache 2.0 (Hugging Face card) |
| UniSpeech-SAT | Verified: 16.8, the best | Verified: 19.5 | Assumed: MIT (Microsoft UniLM) |
| WavLM Large | Verified: 24.3 | Verified: 20.6 | Assumed: MIT |
| wav2vec 2.0 Base (the current P-001 path) | Verified: 40.9 | Verified: 28.4 | Verified: in use |

**Pick: XLS-R as a frozen front end** with a weighted sum of its layers. It has the lowest mean EER across four test sets. UniSpeech-SAT has the best In-the-Wild EER, but a higher mean. **Runner-up:** UniSpeech-SAT. **Fits 16 GB:** assumed (about 0.3B parameters). The current wav2vec 2.0 Base front end is near chance on In-the-Wild in this benchmark, which supports replacing it. A back end in the AASIST style is the common next step; one study found mixed results between XLS-R and WavLM back ends on social-media speech [4].

## Data constraint found in the same check

AV-Deepfake1M++ is **1.46 TB** in total (verified [5]), and this PC has about 519 GB free. P-002 needs one of two things:

- a subset, for example a stratified share of the 1.10M training clips;
- an external disk.

Its fake segments last **0.33 s** on average, with up to 5 per clip (verified, secondary source [6]). That confirms the P-001 lesson (A9): scoring must be temporal, window by window, not one fixed window per clip.

## Sources

1. LNCLIP-DF, "Deepfake Detection that Generalizes Across Benchmarks", arXiv 2508.06248: <https://arxiv.org/html/2508.06248v1>
2. Frozen vision foundation models for facial deepfake detection, arXiv 2605.24965: <https://arxiv.org/html/2605.24965>
3. SSL front ends for audio deepfake detection, arXiv 2603.01482: <https://arxiv.org/html/2603.01482v1>
4. Fake Speech Wild, arXiv 2508.10559: <https://arxiv.org/pdf/2508.10559>
5. AV-Deepfake1M++ dataset card: <https://huggingface.co/datasets/ControlNet/AV-Deepfake1M-PlusPlus>
6. AV-Deepfake1M dataset summary: <https://www.emergentmind.com/topics/av-deepfake1m-dataset>
