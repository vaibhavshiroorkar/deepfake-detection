"""The written half of the app and the report: the math, the comparison with
prior work, and the failures.

Each claim carries the record it comes from, so the app and the PDF never say
more than `docs/research/findings.md` and the run files support. Equations are
LaTeX for KaTeX and follow the handbook's notation.
"""

from __future__ import annotations

MATH: tuple[dict, ...] = (
    {
        "title": "From a score to a probability",
        "source": "docs/handbook/02-deep-learning-foundations.md",
        "items": (
            {
                "tex": r"z = \mathbf{w}^\top \mathbf{x} + b, \qquad p = \sigma(z) = \frac{1}{1 + e^{-z}}",
                "note": "Every model ends in one number z, the logit. The sigmoid turns it into a probability that the clip is fake.",
            },
            {
                "tex": r"\mathcal{L}(y, z) = \log\left(1 + e^{z}\right) - y\,z = \max(z, 0) - z\,y + \log\left(1 + e^{-|z|}\right)",
                "note": "Binary cross-entropy with logits. The second form is the one PyTorch computes, because it never exponentiates a large positive number.",
            },
            {
                "tex": r"\frac{\partial \mathcal{L}}{\partial z} = \sigma(z) - y",
                "note": "The gradient is the gap between the predicted probability and the label, which is why training pushes fakes up and genuine clips down.",
            },
        ),
    },
    {
        "title": "Visual branch: face crops through time",
        "source": "docs/handbook/06-visual-branch.md",
        "items": (
            {
                "tex": r"\begin{aligned} r_t &= \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{t-1} + b_{hr}) \\ z_t &= \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{t-1} + b_{hz}) \\ n_t &= \tanh\big(W_{in} x_t + b_{in} + r_t \odot (W_{hn} h_{t-1} + b_{hn})\big) \\ h_t &= (1 - z_t) \odot n_t + z_t \odot h_{t-1} \end{aligned}",
                "note": "A CNN (EfficientNet-B0 or ConvNeXt-Tiny) turns each of 16 face crops into a feature x_t. A GRU reads them in order, so a flicker between frames can count as evidence.",
            },
            {
                "tex": r"\mathcal{L}_i = -\left[w_{\text{pos}}\, y_i \log \sigma(l_i) + (1 - y_i) \log\left(1 - \sigma(l_i)\right)\right]",
                "note": "Trained on the video label: a clip with a cloned voice over genuine video counts as genuine for this branch.",
            },
        ),
    },
    {
        "title": "The ensemble served in this app",
        "source": "scripts/score_visual_ensemble.py",
        "items": (
            {
                "tex": r"p_{\text{ens}} = \sigma\left(\frac{1}{M} \sum_{m=1}^{M} z_m\right), \qquad M = 6",
                "note": "Six visual models (two backbones, three seeds each) score the same crops. Their logits are averaged with equal weight and nothing is fitted, so no test set shaped the rule.",
            },
        ),
    },
    {
        "title": "Audio branch: attentive pooling",
        "source": "docs/handbook/07-audio-branch.md",
        "items": (
            {
                "tex": r"e_t = \mathbf{w}_a^\top \mathbf{z}_t + b_a, \quad \alpha_t = \frac{e^{e_t}}{\sum_{k=1}^{U} e^{e_k}}, \quad \mathbf{h} = \sum_{t=1}^{U} \alpha_t \mathbf{z}_t, \quad l = \mathbf{w}_c^\top \mathbf{h} + b_c",
                "note": "Wav2Vec2 turns the waveform into U tokens. A learned softmax weights them, so the clip score can rest on the few frames where the voice sounds synthetic.",
            },
        ),
    },
    {
        "title": "Sync branch: does the mouth match the sound?",
        "source": "docs/handbook/08-sync-branch.md",
        "items": (
            {
                "tex": r"s_t = \frac{\mathbf{v}_t^\top \mathbf{a}_t}{\lVert \mathbf{v}_t \rVert_2 \, \lVert \mathbf{a}_t \rVert_2}",
                "note": "Cosine similarity between the mouth token and the audio token at the same moment.",
            },
            {
                "tex": r"\mathcal{L}_{\text{offset}} = -\frac{1}{N} \sum_i \log \frac{e^{o_{i, y_i}}}{\sum_{c=0}^{7} e^{o_{i,c}}}",
                "note": "The audio is shifted by one of eight offsets during training, and the head learns which one was applied.",
            },
            {
                "tex": r"q_{ij} = \frac{\mathbf{v}_i^\top \mathbf{a}_j}{\tau}, \quad \mathcal{L}_{\text{contrast}} = \tfrac{1}{2}\left(\mathcal{L}_{v \to a} + \mathcal{L}_{a \to v}\right), \quad \mathcal{L}_{\text{total}} = \lambda_{\text{offset}} \mathcal{L}_{\text{offset}} + \lambda_{\text{contrast}} \mathcal{L}_{\text{contrast}}",
                "note": "A two-way InfoNCE term pulls each clip's mouth and voice together. Defaults: tau = 0.07, lambda_offset = 1, lambda_contrast = 0.1.",
            },
            {
                "tex": r"\text{anomaly} = \log \sum_{c \neq 3} e^{o_c} - o_3",
                "note": "The log-odds that the clip is at any offset other than aligned (class 3), passed to fusion as one logit.",
            },
        ),
    },
    {
        "title": "Fusion and calibration",
        "source": "docs/handbook/09-fusion-and-calibration.md",
        "items": (
            {
                "tex": r"p_{ij} = \sigma(a_j l_{ij} + b_j), \qquad c_{ij} = \log \frac{p_{ij}}{1 - p_{ij}}",
                "note": "Platt scaling puts each branch on the same log-odds scale before they are combined.",
            },
            {
                "tex": r"\eta_i = \beta_0 + \sum_{j=1}^{J} \beta_j c_{ij} + \beta_f f_i + \beta_k k_i + \beta_d d_i, \qquad p_i = \sigma(\eta_i)",
                "note": "Logistic fusion over the calibrated branches plus three quality features: face coverage f, audio clipping k and audio-video duration gap d.",
            },
            {
                "tex": r"J(\boldsymbol\beta, \beta_0) = \sum_{i=1}^{N} \log\left(1 + e^{-t_i (\boldsymbol\beta^\top \mathbf{x}_i + \beta_0)}\right) + \frac{\lVert \boldsymbol\beta \rVert_2^2 + \beta_0^2}{2C}",
                "note": "The head is fitted on out-of-fold branch scores, so no branch was scored on clips it trained on.",
            },
        ),
    },
    {
        "title": "Cross-modal streams",
        "source": "docs/handbook/10-crossmodal-streams.md",
        "items": (
            {
                "tex": r"\operatorname{Attn}(Q, K, V) = \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V, \qquad \text{mismatch} = A - \operatorname{Attn}(A, M, M)",
                "note": "Audio tokens A query mouth tokens M. What the mouth cannot explain about the sound is the mismatch residual the stream classifies.",
            },
            {
                "tex": r"\text{diagonal mass} = \frac{1}{T} \sum_{t} \sum_{|s - t| \le 1} \alpha_{ts}, \qquad \text{chance} = \frac{3}{50} = 0.06",
                "note": "The share of attention within one step of the diagonal, so chance on a 50 by 50 map is 0.06. Recorded every epoch and never trained on. It stayed at chance while accuracy rose, which is how the project found the streams were not using lip-sync at all (F1).",
            },
        ),
    },
    {
        "title": "How the numbers are measured",
        "source": "src/deepfake_detection/evaluation/metrics.py",
        "items": (
            {
                "tex": r"\text{AUC} = P\left(s_{\text{fake}} > s_{\text{real}}\right)",
                "note": "ROC-AUC: the chance that a random fake scores above a random genuine clip. It measures ranking and ignores the threshold.",
            },
            {
                "tex": r"\text{EER} = \text{FPR}(\theta^\ast) \text{ where } \theta^\ast = \arg\min_\theta \left|\text{FPR}(\theta) - \text{FNR}(\theta)\right|",
                "note": "Equal error rate: the point where false alarms and misses are equally likely.",
            },
            {
                "tex": r"\text{ECE} = \sum_{b=1}^{10} \frac{n_b}{N} \left| \bar{y}_b - \bar{p}_b \right|",
                "note": "Expected calibration error over ten probability bins: does a 0.8 mean fake eight times in ten?",
            },
            {
                "tex": r"D_r = m(F, E_r) - m(V, E_r), \qquad \text{95\% CI} = \left[D_{(2.5\%)},\, D_{(97.5\%)}\right]",
                "note": "Paired bootstrap: resample speaker identities with replacement 1,000 times and score both systems on the same draw. Resampling identities, not clips, matches the question of new people.",
            },
        ),
    },
)


# The four targets the project set itself from the literature, and where they
# stand. docs/handoff.md, "Where the four stated targets stand".
TARGETS: tuple[dict, ...] = (
    {
        "target": "0.991 ROC-AUC visual detection in-domain",
        "literature": "0.991",
        "ours": "0.9994 (ensemble), 0.9988 (single model)",
        "status": "met",
    },
    {
        "target": "0.844+ from audio-visual cues on genuine-frame fakes",
        "literature": "0.844",
        "ours": "0.9969 lip-sync stream on LAV-DF",
        "status": "met on the number, failed on the mechanism",
    },
    {
        "target": "Fusion beats the strongest single branch by about 0.099",
        "literature": "+0.099",
        "ours": "+0.0159 in-domain; -0.0085, -0.0215, -0.0417 on DFDC at three seeds",
        "status": "not met",
    },
    {
        "target": "0.64+ on unseen generators without fine-tuning",
        "literature": "0.64",
        "ours": "0.7527 DF26 leave-one-generator-out (Perception Encoder probe)",
        "status": "met",
    },
)

PRIOR_WORK: tuple[dict, ...] = (
    {
        "title": "DF26 benchmark",
        "comparison": (
            "Our first DF26 probe read 0.6942, level with the best published "
            "FF++-trained detector on DF26 (0.697). With the frame-rate leak "
            "removed and a Perception Encoder probe it reads 0.7527."
        ),
        "difference": (
            "The published ranking has open-source generators easier to catch "
            "than commercial ones. Ours is the reverse, 0.81 against 0.68. The "
            "reason is untested."
        ),
        "source": "findings.md F15, F33",
    },
    {
        "title": "Standard cross-dataset protocol (train FF++, test Celeb-DF and DFDC)",
        "comparison": (
            "Trained on FF++ c23: 0.8723 FF++ test, 0.6348 Celeb-DF, 0.4920 "
            "DFDC. Only roughly comparable to published tables, because our "
            "DFDC set is a subset, not the official test split."
        ),
        "difference": (
            "Training on the more varied corpus made DFDC transfer worse than "
            "training on FakeAVCeleb (0.7611 to 0.4920)."
        ),
        "source": "findings.md F10",
    },
    {
        "title": "Cross-attention lip-sync detectors (DF-TransFusion, LIPINC-V2)",
        "comparison": (
            "We use the same idea, audio querying mouth motion. Our emotion "
            "stream reaches 0.9978 in-domain and the lip-sync stream 0.9969 on "
            "LAV-DF."
        ),
        "difference": (
            "We also logged diagonal attention mass every epoch. It sat at "
            "chance, so the accuracy did not come from lip-sync. We did not "
            "find this check reported in that literature."
        ),
        "source": "findings.md F1, contributions.md item 1",
    },
    {
        "title": '"When AUC Misleads" (polarization under domain shift)',
        "comparison": (
            "Makes the general argument that AUC hides threshold failure. We "
            "measured it three times in one system: JPEG at quality 75 takes "
            "false alarms from 0 to 76 of 150 while AUC drops 0.0054."
        ),
        "difference": "We claim the paired measurements, not the idea.",
        "source": "findings.md F20, F22, F24",
    },
    {
        "title": "Seed sensitivity",
        "comparison": (
            "Known in deep learning generally. Here seed alone moves DFDC "
            "ROC-AUC by 0.0733 for the visual branch and 0.1078 for fusion, "
            "and the seed every result quotes is the best of three."
        ),
        "difference": (
            "Not found reported for deepfake cross-dataset transfer. The "
            "ensemble served here removes the seed choice from the claim."
        ),
        "source": "findings.md F21, F27",
    },
    {
        "title": "Frozen self-supervised audits (arXiv 2606.26384)",
        "comparison": (
            "Covers ground close to our region-of-interest result: a full-frame "
            "probe is at chance on FF++ while face crops are not."
        ),
        "difference": "Read before claiming F13 as new.",
        "source": "findings.md F13, contributions.md",
    },
)

CONTRIBUTIONS: tuple[str, ...] = (
    "An accurate detector shown not to use the cue it is named for: attention alignment stays at chance while ROC-AUC reaches 0.9991 (F1).",
    "A benchmark defect found before training: on DF26, whether a clip has an audio track predicts its label almost perfectly (F14).",
    "Four shortcuts caught by cheap checks before they became results (F14, F15).",
    "A published list of eight withdrawn claims, each with the measurement that replaced it.",
    "The cue-label trap: the same visual logits read 0.9988 or 0.9742 depending on which label is scored, which hides a fusion gain of +0.0159 (F19).",
    "A measured ceiling for fusion: on DFDC, 1,585 of 2,351 clips are wrong under both fusion and its visual branch, so no head can fix them (F23).",
    "Seed variance larger than any effect reported, and a six-model ensemble that removes the seed choice: DFDC 0.7739, +0.047 over the seed mean, though it gains nothing on Celeb-DF.",
)

FAILURES: tuple[dict, ...] = (
    {
        "group": "Negative results",
        "items": (
            {
                "title": "Fusion never beats the visual branch on unseen data",
                "detail": "On DFDC the fused system is -0.0085, -0.0215 and -0.0417 against its own visual branch at three seeds. The research question was answered no.",
                "source": "F19, F27",
            },
            {
                "title": "Fusing all five streams is worse than three of its inputs",
                "detail": "Deep fusion of every Design B stream reads 0.5411 on DFDC. The best subset, emotion plus lip-sync, reads 0.6076.",
                "source": "F4, F8",
            },
            {
                "title": "Audio and sync carry nothing across datasets",
                "detail": "The audio branch is at chance on DFDC (0.5054). The sync branch is below chance on FakeAVCeleb at every seed (0.4364 to 0.4931).",
                "source": "F26, F30",
            },
            {
                "title": "A more varied training set made transfer worse",
                "detail": "Training on FaceForensics++ instead of FakeAVCeleb took DFDC from 0.7611 to 0.4920.",
                "source": "F10",
            },
            {
                "title": "Capture conditions decide transfer, not the forgery type",
                "detail": "A lip-sync stream at 0.9969 on LAV-DF reads 0.7407 on FakeAVCeleb and 0.5059 on DFDC.",
                "source": "F12",
            },
            {
                "title": "One unseen forgery family is at chance",
                "detail": "Holding out a FaceForensics++ family costs 0.12 on average, and FaceSwap held out reads 0.5195.",
                "source": "F34",
            },
            {
                "title": "The ensemble's gain is DFDC only",
                "detail": "Averaging six face models adds +0.047 over the seed mean on DFDC. On Celeb-DF it reads 0.6677 against 0.6682 for the seed-17 model. On MNW at 0.5 it catches 8 of 84 fakes against 24, because the ConvNeXt members score those fakes low. One member, ConvNeXt seed 43, diverged in training and shipped never fine-tuned (F29). It stays in because removing it lowers DFDC to 0.7666 and Celeb-DF to 0.6553.",
                "source": "visual-ensemble-*-metrics.json, F29",
            },
        ),
    },
    {
        "group": "Where the score looks fine and the decision is not",
        "items": (
            {
                "title": "Light compression breaks the operating point",
                "detail": "JPEG at quality 75 costs 0.0054 of AUC and takes false alarms from 0 to 76 of 150 genuine clips. At quality 50 every genuine clip is called fake.",
                "source": "F24",
            },
            {
                "title": "Audio noise fails the other way",
                "detail": "At 20 dB SNR the audio branch misses 857 of 859 fakes while AUC drops only 0.0227.",
                "source": "F25",
            },
            {
                "title": "Confidence is higher when wrong",
                "detail": "On DFDC mean confidence is 0.8742 on errors against 0.8299 on correct calls, so abstaining on low confidence cannot help.",
                "source": "F22",
            },
            {
                "title": "False alarms range from 0 to 91 percent",
                "detail": "Genuine clips called fake: 0 percent in-domain, 4 percent on DFDC, 91 percent on Celeb-DF (141 of 155).",
                "source": "F16",
            },
            {
                "title": "Calibration collapses off-domain",
                "detail": "Calibration error goes from 0.0022 in-domain to 0.6366 on DFDC for fusion, and 0.87 for the ensemble. Its probabilities rank well but are not literal off-domain.",
                "source": "F18, ensemble report",
            },
        ),
    },
    {
        "group": "Evaluation traps caught",
        "items": (
            {
                "title": "A benchmark whose audio track gives the answer away",
                "detail": "On DF26 every commercial generator has audio and almost no genuine clip does. An audio model would have scored near 1.0 by reading the container.",
                "source": "F14",
            },
            {
                "title": "A frame-rate leak through one generator",
                "detail": "Normalising frame rate moved the DF26 result from 0.6942 to 0.6786.",
                "source": "F15",
            },
            {
                "title": "Checkpoint selection on validation loss",
                "detail": "Early stopping on loss kept a checkpoint that scores below chance. Two checkpoints were published and withdrawn.",
                "source": "F3",
            },
            {
                "title": "A flag that built the wrong model",
                "detail": "The WavLM challenger flag silently built a truncated Wav2Vec2.",
                "source": "F28",
            },
            {
                "title": "A backbone that diverged at one seed",
                "detail": "ConvNeXt-Tiny reads 0.9032 in-domain at seed 43 against 0.999 at the other two.",
                "source": "F29",
            },
        ),
    },
    {
        "group": "Claims withdrawn",
        "items": (
            {
                "title": "Freezing BatchNorm is a clean win",
                "detail": "It is a trade. The controlled effect is +0.0177, not +0.1276.",
                "source": "F2",
            },
            {
                "title": "Emotion attention is six times chance",
                "detail": "It was compared against a baseline at a different query length. At 8 steps 0.3438 is chance.",
                "source": "F1",
            },
            {
                "title": "Camera motion causes false alarms",
                "detail": "Measured with the opposite sign: motion lowers the fake score and causes misses.",
                "source": "F6",
            },
            {
                "title": "The density gate stops 79.4 percent of unfamiliar genuine clips",
                "detail": "One unseeded draw. The exact fit reads 74.4 percent, and 67.3 percent over twenty splits.",
                "source": "F17",
            },
        ),
    },
    {
        "group": "Engineering failures",
        "items": (
            {
                "title": "Out of memory with 15 GB free",
                "detail": "Windows commit charge, not RAM: 126.7 of 127.9 GB committed after repeated forced kills of CUDA processes.",
                "source": "paper.md section 6",
            },
            {
                "title": "A cache loader that read every view",
                "detail": "It loaded about 20 MB per clip where 9.19 MB was needed, and killed a run after two and a half hours.",
                "source": "paper.md section 6",
            },
            {
                "title": "A chain script that failed silently",
                "detail": "A wrapper waiting on another job exited without training anything.",
                "source": "docs/obstacles.md",
            },
            {
                "title": "A crash on single-model uploads, fixed for this release",
                "detail": "The visual-only prediction path built a record without the sync start field the preprocessor reads, so it raised on every upload.",
                "source": "src/deepfake_detection/inference/predictor.py",
            },
            {
                "title": "Never run",
                "detail": "The leading-silence ablation, the MTCNN against YuNet detector benchmark, the AASIST and SyncNet baselines, and a clean-environment reproduction.",
                "source": "ROADMAP.md, F31",
            },
        ),
    },
)

LIMITS: tuple[str, ...] = (
    "Not a deployable detector. On footage unlike its training data it can call most genuine videos fake.",
    "Trained on one corpus, FakeAVCeleb: frontal, studio-like YouTube speakers.",
    "Face models judge the video track only. A cloned voice over genuine video is invisible to them.",
    "Probabilities are calibrated in-domain only. Off-domain, read them as a ranking, not as odds.",
    "Every Design B number is from one training seed.",
)
