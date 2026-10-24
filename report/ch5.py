"""Chapter 5, nomenclature, appendices."""


def conclusions(B):
    B.h1("Conclusions and Further Work")
    B.box("This chapter presents the conclusions that the evidence of the project supports at this stage, with the important points in list form, and the further work that completes "
          "the planned path from the working version to the frozen system.")
    B.h2("Conclusions", trace="CLAUDE.md rules 1 and 8; DOC_FACTS; STATUS")
    B.p("The project built, from zero and in one repository, the foundations of a multimodal deepfake detector and the protocol under which it is judged. A group-disjoint evaluation "
        "harness exists, with a development slice for ranking, sealed test sets behind a guard and a ledger, bootstrap intervals over groups and seeds, paired and non-inferiority "
        "comparisons, and an acceptance test that the harness passes. The data sets were obtained, verified and split, with their licences recorded, and the Celeb-DF family test set was "
        "corrected so that no source video is counted twice. A shared preprocessing package, a measured choice of face detector, a checkpoint format that refuses a mismatch, an "
        "inference library and a local web page exist. The comparison of eleven pretrained encoders and the iteration ladder were designed with hypotheses, success criteria and failure "
        "criteria written before any run.")
    B.p("What the project does not yet have is a final measured performance. The ranking of the encoders, the metrics of the working version and the verdicts of the iterations are "
        "marked as pending in this report because their runs are not frozen, and the report makes no claim about detection accuracy, calibration or comparability with a published detector. "
        "Where numbers of other papers are quoted, they are quoted with their own protocol and as context.")
    B.p("The important points are the following.")
    B.bullets([
        "The protocol is fixed before any model: the headline metric is the ROC-AUC on a data set never used in training with a 95 percent group bootstrap interval, the threshold and the temperature are fitted on validation data only, and the sealed test sets are touched once per frozen iteration.",
        "The splits are group-disjoint, the group being the source video together with the person identity; the leakage check passes on 80,803 image files and on 41,739 files of Celeb-DF, In-the-Wild and DFD, and the Celeb-DF family test set of 744 real and 6,906 fake videos has about 44 independent identity groups, so its intervals are wide.",
        "The evaluation harness passes its acceptance checks, including a bootstrap coverage of 192 of 200 simulated data sets against a requirement of 90 percent.",
        "Eleven pretrained encoders, seven for images and four for speech, plus two small networks trained from scratch as references, are compared under one rule that prefers the cheaper candidate when the paired interval includes zero.",
        "The working version is defined by seven checkable conditions, and its code path, checkpoint format and web page exist; a visible banner keeps a development stand-in from being mistaken for a result.",
        "The pulse experiment has its thresholds written before any run, and it reports a negative outcome as an outcome.",
        "Two limitations are stated with the design: small development slices, and weights of two encoders that are research-only.",
    ])
    B.h2("Further Work", trace="ITERATIONS; DECISIONS D-038, D-042, D-045; DOC_FACTS section 13")
    B.p("The remaining work follows the plan of the iteration ladder and the notes of the decisions.")
    B.bullets([
        "Freeze the encoder comparison for the three tasks and fit the heads of the winners, which completes the working version and its diagnostic battery.",
        "Run iterations I-1 to I-7, each with its lever chosen from the diagnostic battery of the preceding iteration, and report an iteration that is not finished as not run.",
        "Evaluate every frozen iteration once on the sealed test sets and run the two confirmatory comparisons, which needs the published baseline to be named and re-run, and the margin $\\delta$ to be fixed from its seed spread.",
        "Fine-tune the generated-image and the speech tasks if time remains, and consider a low-rank adapter variant of the two CLIP ViT-L/14 models, which do not fit the card for full fine-tuning.",
        "Decide the combining rule for the visual and audio scores of a video once both unimodal heads are calibrated on LAV-DF.",
        "Add DINOv3 as an iteration if its access gate is accepted, add FakeAVCeleb and WildDeepfake as further tests when obtained, and add UBFC-rPPG only if the pulse experiment needs ground truth.",
        "Treat the JPEG quality of the stored crops as a candidate preprocessing iteration, and revisit the identity exclusion of Celeb-DF++ with a narrower rule (target identity only) if that is preferred.",
        "Extend the output with temporal localisation, which was excluded from the first version because temporal labels are not available in every data set.",
    ])


NOMEN = [
    ("$y_i$", "label of example i, 1 for fake and 0 for real"),
    ("$s_i,\\ \\hat p_i$", "score and predicted probability of fake"),
    ("$n_0,\\ n_1,\\ n,\\ \\pi$", "numbers of real and fake examples, their sum and the fake prevalence"),
    ("$t,\\ t^{*}$", "decision threshold and the threshold at which the two error rates are equal"),
    ("TPR, FPR, TNR, FNR", "true positive, false positive, true negative and false negative rates"),
    ("$\\sigma(u)$", "logistic function, 1 / (1 + exp(-u))"),
    ("$T$", "temperature of temperature scaling, or the number of frames of a video (context decides)"),
    ("$M,\\ B_m$", "number of calibration bins and the examples in bin m"),
    ("$B,\\ \\alpha$", "number of bootstrap resamples and the significance level"),
    ("$\\Delta,\\ \\delta$", "difference of a metric between two models and the non-inferiority margin"),
    ("$r$", "rank of a low-rank adapter"),
    ("$\\rho$", "correlation of the scores of frames or of encoders"),
    ("$\\ell$", "logit, the log-odds of the fake class"),
]

ABBR = [
    ("AUC", "area under the ROC curve"), ("AP", "average precision"), ("BA", "balanced accuracy"), ("ECE", "expected calibration error"),
    ("EER", "equal error rate"), ("ROC", "receiver operating characteristic"), ("GAN", "generative adversarial network"),
    ("rPPG", "remote photoplethysmography, the pulse visible in skin colour"), ("SNR", "signal-to-noise ratio"),
    ("POS", "plane orthogonal to skin, a projection method for the pulse signal"), ("LoRA", "low-rank adaptation"),
    ("FF++", "FaceForensics++"), ("DFD", "DeepFakeDetection data set"), ("LAV-DF", "localized audio visual deepfake data set"),
    ("dev_x", "development slice of a cross-dataset source"), ("test_in, test_x", "sealed in-dataset test and sealed cross-dataset test"),
    ("I-0 to I-8", "working version, iterations 1 to 7 and the integration and freeze step"),
    ("c23, c40", "FaceForensics++ compression levels"), ("YuNet", "face detector used by the preprocessing"),
]


def nomenclature(B):
    B.p("<b>Symbols</b>", ppr='<w:keepNext/>')
    B.simple_table(["Symbol", "Meaning"], [[a, b] for a, b in NOMEN], [4.6, 11.8], ["left", "left"], size=11)
    B.p("<b>Abbreviations</b>", ppr='<w:keepNext/>')
    B.simple_table(["Abbreviation", "Meaning"], [[a, b] for a, b in ABBR], [4.6, 11.8], ["left", "left"], size=11)


def appendices(B):
    B.h1("Fixed Settings of the Working Version", kind="appendix", trace="PROTOCOL; O007_SWEEP section 5; DECISIONS D-024, D-034")
    B.p("<ref tab:settings> lists the settings that are fixed for the working version. Changing any preprocessing setting gives a new preprocessing hash and is recorded as a parameter change.")
    B.table("settings", "Fixed settings of the working version.",
            ["Item", "Value"],
            [["Frames per video", "16, uniformly spaced"],
             ["Face detector and crop", "YuNet 2023mar; largest face; enlargement factor 1.3; 224 by 224"],
             ["Stored crop", "JPEG, quality 95"],
             ["Audio window", "first 64,600 samples at 16 kHz, repeat-padded"],
             ["Features", "stored once as 16-bit floats; standardised with the training mean and standard deviation"],
             ["Head", "one linear layer; class-weighted binary cross-entropy"],
             ["Optimiser", "AdamW, learning rate 1e-3, weight decay 1e-4"],
             ["Batch and epochs", "1,024 features; at most 200 epochs; early stopping on validation AUC with patience 10"],
             ["Seeds", "0, 1 and 2 for the head of every candidate"],
             ["Clip score", "mean of the frame probabilities"],
             ["Threshold", "maximises balanced accuracy on validation; midpoints of adjacent distinct validation scores; fixed for all test sets"],
             ["Calibration", "temperature scaling fitted on validation; ECE with 15 equal-width bins, top-label version"],
             ["Probability clipping", "to [1e-7, 1 - 1e-7] before log loss"],
             ["Bootstrap", "2,000 resamples, 95 percent percentile interval, fixed base seed 20261024, resampling unit the group, one-class resamples redrawn"]],
            [4.6, 11.8], ["left", "left"], size=10)
    B.h1("Weights Recorded Before Use", kind="appendix", trace="DATA_LOG encoder weights table; DECISIONS D-038")
    B.p("The weight files of the shortlisted encoders were recorded with their size, SHA-256 and repository revision before use. <ref tab:hashes> gives the size, the first 16 hexadecimal digits "
        "of the hash and the first 12 characters of the revision.")
    B.table("hashes", "Weight files of the shortlisted encoders.",
            ["Encoder", "File", "Size in bytes", "SHA-256, first 16 digits", "Revision, first 12"],
            [["EfficientNet-B4", "model.safetensors", "77,933,206", "8030f9c929ed71a0", "442c00e15609"],
             ["Xception", "xception-43020ad28.pth", "91,675,053", "43020ad2812c333d", "release v0.1"],
             ["DINOv2 small", "model.safetensors", "88,249,960", "ae1e99fcefd534ed", "ed25f3a31f01"],
             ["DINOv2 base", "model.safetensors", "346,345,912", "d73036b56966966d", "f9e44c814b77"],
             ["CLIP ViT-L/14, OpenAI", "model.safetensors", "1,710,540,580", "a2bf730a0c7debf1", "32bd64288804"],
             ["CLIP ViT-L/14, LAION", "model.safetensors", "1,710,540,580", "0fce40ac684817a1", "162703219714"],
             ["SigLIP base", "model.safetensors", "812,672,320", "2c63cb7d1f2e95ba", "7fd15f0689c7"],
             ["wav2vec 2.0 base", "pytorch_model.bin", "380,267,417", "3249fe98bfc62fcb", "0b5b8e868dd8"],
             ["WavLM base plus", "pytorch_model.bin", "377,617,425", "3bb273a6ace99408", "4c66d4806a42"],
             ["Whisper small", "model.safetensors", "966,995,080", "1d7734884874f1a1", "973afd24965f"],
             ["HuBERT base", "pytorch_model.bin", "377,569,754", "062249fffb353eab", "dba3bb02fda4"]],
            [3.8, 3.8, 2.8, 3.6, 2.4], ["left", "left", "right", "left", "left"], size=9)
