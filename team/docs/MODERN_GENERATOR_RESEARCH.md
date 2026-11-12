# Modern generator research (images and video)

Working record of the modern-generator experiments of 2026-10-27 (D-079, D-080, D-081). Each section is owned by one agent (docs/LIVE_STATUS.md); numbers come only from the logged experiments named in each section (docs/EXPERIMENTS.md).

## Image

(E-028 and E-029; completed 2026-10-27; every number below is from docs/EXPERIMENTS.md, intervals are 95 percent bootstrap intervals over images; independently recomputed by the main agent.)

- E-028: the installed generated-image head (CLIP ViT-L/14, linear, trained on ProGAN) on 11,550 images of a test-only subset of Community Forensics-Eval (21 generators): macro AUC 0.7039 [0.6855, 0.7229], pooled false-negative rate 0.7395 [0.7300, 0.7500], pooled false-positive rate 0.0418; INADEQUATE. GANs 0.9580, commercial 0.7351, latent diffusion against RAISE reals 0.5808. Format cue: format-only score 0.9222; PNG-only AUC 0.6123. Face-routed images (3,119): this head 0.7859, the face-swap path 0.6020, paired +0.1839 [+0.1596, +0.2074]; neither usable.
- E-029: the same kind of head trained on 23,807 images of 125 generators (Community Forensics-Small), validated on 31 other generators and other real sources, frozen, scored once: macro AUC seed mean 0.8839 [0.8732, 0.8942], paired +0.1800 [+0.1643, +0.1959] over the installed head; +0.1675 [+0.1540, +0.1803] on the 18 generators without a related model in training; PNG-only 0.8365. At its validation threshold 0.5213 [0.5091, 0.5339] of the real Eval images are flagged (LAION 0.5885, COCO 0.0328): ranking improves, calibration to unseen real sources does not; not eligible for the product (D-086).
- Reading: generator diversity in training helps ranking on unseen generators at about 1 percent of the published scale, as the published work argues; it does not by itself give a usable threshold; the evaluation set differs in format and size between real and generated images, only partly controlled.

## Video

(owned by the video agent; written 2026-10-27 after E-027 was completed; every number below is from docs/EXPERIMENTS.md E-020, E-026 or E-027, intervals are 95 percent cluster-bootstrap intervals over source groups unless marked Wilson)

### What the project now covers and what it misses

| Generator | Where | Role | Measured |
|---|---|---|---|
| BDAnimateDiffLightning, CogVideoX5B, RunwayML, StableDiffusion, Veo, VideoPoet | DeepAction (E-020) | training and LOGO development of the installed E-020 head | E-020 development only |
| HunyuanVideo 1.5 14B i2v and t2v, LTX 2.3 distilled i2v and t2v, Wan 2.2 14B i2v and t2v | DF26 open subset | E-020: evaluation; E-027: training, leave-one-family-out and leave-one-generator-out development | E-020, E-027 |
| Grok Imagine, Kling 3.0, Veo 3.1, Wan 2.6 | DF26 closed subset | evaluation only (E-020 look 1, E-027 look 2) | E-020, E-027 |

Missed: every generator not in the table (no clip of any other text-to-video or image-to-video system has been scored), every content type other than DF26's 5 s, 720p talking heads and DeepAction's short action clips, anything after the first 2.0 s of a clip (canonical-v3 window), and the audio track (these heads are visual only). Three open families give only three held-out units for leave-one-family-out.

### What E-020, E-026 and E-027 show

- E-020 (installed experimental head: DINOv2-small mean embedding, logistic regression, trained on DeepAction): DeepAction LOGO 0.8281 [0.7908, 0.8662], but DF26 closed subset 0.4880 [0.4534, 0.5204] against all 271 DF26 reals: it did not transfer to modern talking-head generators.
- E-026 (source confound of the E-020 head): at its threshold it flags 0.7682 [0.7265, 0.8052] (Wilson) of 440 unmodified real clips from FF++ and LAV-DF, against 0.19 of DeepAction's own reals out of fold. Its DeepAction result mostly separates one stock library from everything else.
- E-027 (heads trained on DF26's six open generators, 3 fold seeds, source-group folds):
  - Development, leave-one-family-out macro AUC: the best candidate is CLIP ViT-L/14, mean of 8 frames, logistic regression (P8-IMG-13), at 0.7453 [0.7174, 0.7710]. It beats the pixel-statistics control C3 by +0.2532 [0.2259, 0.2811] and the installed E-020 head by +0.2458 [0.2150, 0.2745]. Leave-one-generator-out is only +0.0075 [0.0037, 0.0117] above leave-one-family-out. H-X3 is SUPPORTED and the preregistered selection chose P8-IMG-13. With the E-020 encoder (DINOv2-small) and DF26 training, the result is 0.6746 [0.6513, 0.6956], so most of the gain comes from in-domain training data and a further +0.0707 [0.0439, 0.0960] from the larger CLIP encoder.
  - Temporal question: there is no evidence that temporal features add to pooled appearance. Adding frame-to-frame change and spread to the mean lowers the AUC on CLIP (-0.0404 [-0.0650, -0.0165]). 14 frames are no better than 8. Change-only features reach 0.53 to 0.57. In this setup (2 s, 7 fps, frozen image encoders, linear head) the signal is per-frame appearance.
  - Closed look (frozen fold models, counted once): 0.7548 [0.7189, 0.7885] against the 200 scenario-matched reals, and 0.7800 [0.7437, 0.8109] against all 271, compared with E-020's 0.4880. H-X3c is SUPPORTED by its preregistered line (lower bound above 0.70). Per generator: Wan 2.6 (same developer as the training Wan 2.2) 0.8758 [0.8435, 0.9060], Veo 3.1 0.7599 [0.7181, 0.7969], Grok Imagine 0.7360 [0.6910, 0.7808], Kling 3.0 0.6483 [0.5992, 0.6951]. At the out-of-fold balanced-accuracy threshold, 0.40 [0.3346, 0.4692] (Wilson) of the 200 reals are flagged, more than the 0.2841 the threshold was set at. ECE is 0.2016 raw and 0.0479 after Platt scaling.
  - Source confound: CONFOUND PERSISTS. The deployable head flags 0.8023 [0.7625, 0.8368] (Wilson) of the same 440 FF++ and LAV-DF real clips. Inside DF26 its false positives concentrate on YouTube-sourced real clips (0.3062) rather than TalkingCelebs clips (0.0484).
  - X4: 16 frames spread over the whole clip slightly improve the 8-frame canonical view (+0.0188 [0.0060, 0.0321]). Letterboxing hurts (-0.0613 [-0.0803, -0.0423]).
- What this means. Training a linear head on frozen CLIP features from matched real and generated talking heads gives a usable separation of DF26 real from DF26-style generated clips, including four closed commercial generators it never saw. But the same head scores most real footage from other sources as generated-like, so on an arbitrary uploaded video its score cannot distinguish "generated" from "real footage that does not look like DF26's reals". The closed-look reals are the same 271 clips used during development (only the generated side is new), and the subset had already been looked at once.

### Product-integration recommendation

Candidate for experimental head under roadmap section 8.5 promotion rules (decided autonomously - needs user review; this is not an approval, and the product decision belongs to the main agent and the user). The frozen deployable head is checkpoints/modern-E027/deploy_P8-IMG-13.npz (SHA-256 778cb271...5ce3; threshold t_BA 0.6587, out-of-fold). It passed its own preregistered criteria (H-X3, H-X3c) and dominates the installed E-020 head on DF26. It may only enter as another experimental head: new head_id, new folder, gate closed, `APPROVED_HEADS` unchanged, finding never shown as definitive. Its tested-scope text must quote the source-confound result (0.8023 of other-source real clips flagged), the reuse of DF26 reals in the closed look, and the CONDITIONAL licence wording of the CLIP ViT-L/14 weights (research use). It also costs a 304 M-parameter encoder (peak 3.6 GB VRAM in fp16 extraction) instead of 22 M. Promotion to a definitive finding would need an independent preregistered validation on fresh real and generated clips from other sources, and that does not exist yet. If the main agent prefers not to add a second experimental head before 30 October, "keep experimental / do not install" is the safe alternative; nothing in E-027 requires installation.
