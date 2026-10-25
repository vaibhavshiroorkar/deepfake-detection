"""Chapter 5, nomenclature, appendices."""


def conclusions(B):
    B.h1("Conclusions and Further Work")
    B.box("This chapter presents the conclusions that the evidence of the project supports at this stage, with the important points in list form, and the further work that completes "
          "the work that remains after the sealed-test evaluation.")
    B.h2("Conclusions", trace="CLAUDE.md rules 1 and 8; DOC_FACTS; STATUS")
    B.p("The project built, from zero and in one repository, the foundations of a multimodal deepfake detector and the protocol under which it is judged. A group-disjoint evaluation "
        "harness exists, with a development slice for ranking, sealed test sets behind a guard and a ledger, bootstrap intervals over groups and seeds, paired and non-inferiority "
        "comparisons, and an acceptance test that the harness passes. The data sets were obtained, verified and split, with their licences recorded, and the Celeb-DF family test set was "
        "corrected so that no source video is counted twice. A shared preprocessing package, a measured choice of face detector, a checkpoint format that refuses a mismatch, an "
        "inference library and a local web page exist. Eleven pretrained encoders were compared on the development slice under one rule, hypotheses, success criteria and failure "
        "criteria having been written before any run, and the first working checkpoint was built from the three winners and tested end to end.")
    B.p("The results are exploratory, on validation data and on development slices. On face video, frozen DINOv2 small separated clips of unseen identities with an AUC of 0.6933 [0.6639, 0.7314], "
        "and full fine-tuning of SigLIP base raised the slice AUC to 0.9096 [0.8776, 0.9388] (validation 0.9899); a re-run Xception baseline with the same recipe, at the learning rate selected on validation data, reaches 0.8716 [0.8364, 0.9076], and the "
        "difference on the slice is +0.0380 [+0.0172, +0.0592]. Two further levers on the face task did not enter the final system: augmentation matched to compression, blur and resize with mixup gave 0.9315 [0.9055, 0.9548] "
        "on the slice and lowered the validation AUC to 0.9705 [0.9618, 0.9795], a trade-off, and the average of three fine-tuned encoders gave 0.9192 [0.8948, 0.9457], not distinguished from the single encoder. On speech, frozen encoders did not transfer reliably to In-the-Wild (the best lower bound, 0.5338, was below the line fixed in advance, so that "
        "comparison was inconclusive), and three levers took the slice AUC from 0.6100 [0.5434, 0.7256] to 0.8130 [0.7523, 0.8808]: trimming and normalising the clip, augmenting the training clips, and averaging three encoders. "
        "On generated images the OpenAI CLIP head separates the two generator units of the slice with 0.9935. On LAV-DF the frozen face head was below chance (0.3838), and a head trained on LAV-DF "
        "reached a fused AUC of 0.9935 that a control shows is driven by a cue that separates original videos from produced ones, not by a visual manipulation signal (0.5070 within the produced videos); "
        "the audio head, at 0.8574 within the produced videos, is the part that detects a manipulation there. The pulse signal was not recoverable from the data (0.301 against the required 0.6); a positive control showed that the same pipeline recovers an injected 1.2 Hz sinusoid of a relative green amplitude of 0.008 in 377 of 432 clips, "
        "which shows its sensitivity to a clean signal of that size and does not show that a real pulse exists in these videos or survives compression.")
    B.p("The sealed run was made on 25 October 2026, once for each frozen iteration, and the three confirmatory tests declared before it give the following. On the sealed Celeb-DF family set (7,650 videos, 344 groups) the final face-video system "
        "reaches an AUC of 0.8218 [0.8020, 0.8393] against 0.6238 [0.6015, 0.6486] for the working version, a difference of +0.1980 [+0.1688, +0.2242], which is improved under the declared rule and also at the 98.33 percent level "
        "([+0.1627, +0.2303]). On the sealed In-the-Wild set (24,788 clips, 42 speakers) the final speech system reaches 0.9148 [0.8896, 0.9383] against 0.7516 [0.6802, 0.8035], a difference of +0.1633 [+0.1177, +0.2266], "
        "improved, and robust at the 98.33 percent level ([+0.1085, +0.2481]). Against the re-run Xception baseline (0.8042 [0.7841, 0.8223]) the final face-video system is higher by +0.0176 [+0.0025, +0.0327]: superior at the "
        "95 percent level, not robust to the 98.33 percent level, whose interval [−0.0011, +0.0355] contains zero, and non-inferior at both levels. The report claims nothing beyond this wording.")
    B.p("What these results do not show is stated with them. The comparison with a published detector rests on one baseline family at 224 pixels and one sealed set. The sealed Celeb-DF family has 344 groups, and its fakes vary over 44 identities; "
        "the sealed In-the-Wild set has 42 speakers. The margin of 0.0100 comes from three seeds and a floor. Iterations 1b and 1c were not tested on the sealed sets. The sealed numbers are one run of each frozen iteration. "
        "Two predictions for speech failed upward, since the final speech system and the working version reached 0.9148 and 0.7516 against the predicted ranges 0.60 to 0.85 and 0.50 to 0.70, and the sealed video value is below its development value (0.8218 against 0.9096). "
        "Every sealed number outside the three tests is exploratory. On the sealed LAV-DF subset the development pattern repeats, and the report makes no claim of audio-visual detection. "
        "Where numbers of other papers are quoted, they are quoted with their own protocol and as context.")
    B.p("The important points are the following.")
    B.bullets([
        "The protocol is fixed before any model: the headline metric is the ROC-AUC on a data set never used in training with a 95 percent group bootstrap interval, the threshold and the temperature are fitted on validation data only, and the sealed test sets are touched once per frozen iteration.",
        "The splits are group-disjoint, the group being the source video together with the person identity; the leakage check passes on 80,803 image files and on 41,739 files of Celeb-DF, In-the-Wild and DFD, and the Celeb-DF family test set of 744 real and 6,906 fake videos has about 44 independent identity groups, so its intervals are wide.",
        "The evaluation harness passes its acceptance checks, including a bootstrap coverage of 192 of 200 simulated data sets against a requirement of 90 percent.",
        "Eleven pretrained encoders, seven for images and four for speech, plus two small networks trained from scratch as references, are compared under one rule that prefers the cheaper candidate when the paired interval includes zero.",
        "The working version is defined by seven checkable conditions and meets them: its checkpoint scores a file through the local web page with the source marked real, the parity between stored features and the raw-file path is within 0.0019, and a visible banner keeps a development stand-in from being mistaken for a result.",
        "Fine-tuning the face encoder removed the bias seen in the working version (training AUC 0.837 to 0.9994) and raised the development AUC by +0.1700 [+0.1228, +0.2114] over the probe of the same encoder; most of that gain comes from fine-tuning, since the Xception baseline gains +0.1783 [+0.1296, +0.2140] over the working version as well.",
        "The speech task is the weakest: three levers raised the slice AUC to 0.8130, the slice has only 12 speakers, and the calibration error stays high, so the audio score is a ranking score.",
        "The audio-visual result on LAV-DF is an in-data-set result that a production-pipeline cue of the visual head drives; the visual head has no demonstrated ability to detect a visual manipulation there, and the audio head does. The system therefore shows the two scores side by side with a warning and no fused verdict.",
        "The pulse experiment has its thresholds written before any run, and it reported a negative outcome as an outcome: the pulse signal is not recoverable from these data. A positive control with an injected sinusoid shows that the pipeline is sensitive to a clean signal of relative amplitude 0.008 and leaves open whether a real pulse exists.",
        "A version-2 checkpoint of the frozen system exists, is covered by 304 unit tests, and passed a parity check on the graphics processor (largest difference 0.0019); the web page still runs the checkpoint of the working version and has not been switched to it.",
        "The sealed run gave two improvements over the working version (face video and speech) at both levels, and a superiority over the re-run Xception baseline that holds at the 95 percent level, is not robust to the 98.33 percent level and is non-inferior at both; every other sealed number is exploratory.",
        "Limitations are stated with the results: small development slices, one baseline family at 224 pixels, 344 groups and 42 speakers in the sealed sets, a margin from three seeds, a baseline learning rate that is the highest of its grid, two extra face-video iterations that were run although their triggers were not met to the letter, and weights of two encoders that are research-only.",
    ])
    B.h2("Further Work", trace="ITERATIONS; DECISIONS D-038, D-042, D-045; DOC_FACTS section 13")
    B.p("The remaining work follows the plan of the iteration ladder and the notes of the decisions.")
    B.bullets([
        "Switch the local web page to the checkpoint of the frozen system and run its page check on it.",
        "Repeat the comparison with further families of published detectors and at other input sizes, with more seeds for the margin, because the present superiority rests on one baseline family at 224 pixels and is not robust to the 98.33 percent level.",
        "Test on sealed sets with more independent identities and speakers than the 44 identities and 42 speakers of the present sets, and explain by an experiment why the speech predictions failed upward and the sealed video value is below its development value.",
        "Evaluate iterations 1b and 1c on a sealed set if they are to enter a later system.",
        "Look for a visual manipulation signal on LAV-DF with a recipe that does not rely on the production pipeline, for example more frames per video or a temporal model, before any audio-visual claim is made.",
        "Extend the baseline grid beyond the learning rate of 3e-4 if the comparison is to be repeated, and fine-tune the generated-image task if time remains.",
        "Separate the perturbations from mixup in the augmentation lever, and test it on a held-out perturbation slice, before any claim of robustness to compression is made.",
        "Test the pulse pipeline with an injected signal that varies in frequency and is re-encoded with H.264, and measure the amplitude of a real pulse on data with ground truth, before concluding anything about a pulse in compressed video.",
        "Treat the audio calibration as open: the calibration error on the slice stays high after temperature scaling.",
        "Add DINOv3 as an iteration if its access gate is accepted, add FakeAVCeleb and WildDeepfake as further tests when obtained, and add UBFC-rPPG only if a pulse experiment is repeated with ground truth.",
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
