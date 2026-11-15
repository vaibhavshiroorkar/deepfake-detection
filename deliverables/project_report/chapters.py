"""Abstract and Chapters 1 to 10 of the v2 project report.

Every number is copied from deliverables_v2/CONTENT_MAP.md (same decimals and labels); speech EER uses the freeze report
values (CONTENT_MAP 9.2, primary record). Experiment, decision and research ids appear only in Word comments (trace=...).
"""
from build_report import PLATFORM_FIGURE, DASHBOARD_SCREENSHOTS, DASHBOARD_WIDTH_CM, FIG
import os

W = 16.4  # usable table width in cm


# ---------------------------------------------------------------------------------------------------------------
def abstract(B):
    B.h1("Abstract", kind="front")
    B.p("Detectors of manipulated media often score well on the data they were developed on and markedly worse on data from other sources. This project built a "
        "local system that checks still images, videos and the speech inside videos or audio files, and recorded every design choice and experiment so that each "
        "step can be traced to evidence. Each experiment had a written hypothesis and success criterion before its run; splits are group-disjoint, sealed test "
        "sets were read once per frozen iteration through a guarded, logged command, and intervals are 95 percent group bootstrap over three seeds.")
    B.p("The first working version used one frozen pretrained encoder per task with a linear head. Its diagnostics led to full fine-tuning of a SigLIP base encoder "
        "for face video and, for speech, to silence trimming, augmentation and an average of three encoders; audio-visual fusion, a pulse (rPPG) check and two "
        "further video levers were not adopted. On the sealed Celeb-DF family set (7,650 videos, 344 groups: 44 identities and 300 single YouTube videos) the final "
        "face-video system reached an AUC of 0.8218 [0.8020, 0.8393] against 0.6238 for the first version, and on sealed In-the-Wild speech 0.9148 [0.8896, 0.9383] "
        "against 0.7516; part of the speech gain removes a silence and level shortcut, so it is not all better detection of synthesis. Against a re-run Xception "
        "baseline the final video system is non-inferior at both levels (margin 0.0100; difference +0.0176 [+0.0025, +0.0327]); the declared 95 percent "
        "superiority rule was met but does not survive the Bonferroni correction (98.33 percent interval [-0.0011, +0.0355]), and the baseline ran at 224 px.")
    B.p("After the freeze the detectors were wrapped in an eight-module analysis platform with per-module labels and no overall score. Later studies added limits "
        "rather than claims. On the sealed half of FakeAVCeleb the frozen system's AUCs (visual 0.9552, audio 0.9263) met their pre-stated criteria, but they are not "
        "evidence of detection quality: the video codec coincides with the manipulation method (codec metadata alone separates real from fake video at 0.9985 on the "
        "development half), and the frozen thresholds do not transfer. On LAV-DF a fused audio-visual score looked near perfect, but its visual part is at chance "
        "within the produced videos (0.507 on validation, a post hoc control; 0.503 on the sealed subset, exploratory), so no audio-visual detection is claimed. The "
        "image head misses 0.7395 of the images of 21 recent generators, and generated-video heads are confounded by the video source. No Gemini-specific "
        "experiment exists.")
    B.p("<b>Keywords:</b> deepfake detection, multimodal analysis, group-disjoint evaluation, frozen and fine-tuned encoders, group bootstrap, non-inferiority, "
        "per-module reporting.")


# ---------------------------------------------------------------------------------------------------------------
def ch1(B):
    B.h1("Introduction")
    B.box("This chapter gives an overview of the project, states the problem, lists the objectives and fixes the scope: what the system claims and what it does not.")
    B.h2("Overview", trace="CONTENT_MAP 2")
    B.p("Deepfakes are images, videos and audio clips that a learned model has generated or altered: a swapped or re-enacted face, a fully generated picture, or "
        "synthesised or converted speech. Detectors trained on one benchmark are usually reported on test data from the same benchmark; when the test data come "
        "from another source their scores drop <c bench>. A person who receives a suspicious file is in the second situation.")
    B.p("This project built, from zero and on one desktop machine, a local system that analyses an uploaded image, video or audio file and reports, check by check, "
        "what each detector found within the scope on which it was tested. The work followed a documented path: a simple working version first, then iterations "
        "that each changed one thing for a stated reason, a freeze, one reading of sealed test sets, and finally a modular analysis platform with a dashboard. This "
        "report is the concise engineering and evaluation record of that path.")
    B.h2("Problem Statement", trace="CONTENT_MAP 2, 3")
    B.p("Given an image, a video with or without an audio track, or an audio file, report for each applicable check whether it found signs of manipulation, with the "
        "meaning of its score and the limits of the evidence behind it, and measure the detectors on data from sources that were never used for training or tuning. "
        "The system must not turn several weak signals into one overall verdict, and it must not present a detector score as the probability that the file is fake.")
    B.h2("Objectives", trace="CONTENT_MAP 3")
    B.numbered([
        "Define one evaluation protocol before any model exists: group-disjoint splits, development slices separate from sealed tests, sealed tests read once per "
        "frozen iteration through a guarded and logged command, three seeds and 95 percent group-bootstrap intervals.",
        "Build face-video, generated-image and speech detectors, choose their encoders by a measured comparison, and improve them in iterations that each start "
        "from a written hypothesis and success criterion.",
        "Decide in advance three confirmatory comparisons for the sealed tests, including one against a re-run published baseline network.",
        "Wrap the frozen detectors in a local analysis platform that reports one result per module with its tested scope, limitations and uncertainty, and never a fused score.",
        "Keep failed and rejected approaches in the record, with the reason they were rejected.",
    ])
    B.h2("Scope", trace="CONTENT_MAP 3; D-069, D-077")
    B.p("<ref tab:scope> separates what the evidence supports from what is outside the claims of this report.")
    B.table("scope", "Scope of the claims.", ["Claimed (on the stated data, with intervals)", "Not claimed"], [
        ["A reproducible group-disjoint protocol with sealed tests and a logged guard",
         "Audio-visual detection or a fused verdict (the LAV-DF fusion rests on a production-pipeline cue)"],
        ["Face-video and speech detectors with sealed results and three pre-declared confirmatory tests",
         "Pulse-based (rPPG) detection; lip-sync deepfake detection; temporal localisation"],
        ["A generated-image detector with sealed exploratory results on ProGAN-era test sets",
         "Generated-video detection, or generated-image detection on modern generators"],
        ["A local platform with 8 modules, per-module results and no overall score",
         "Speed or throughput claims; a ranking above published detectors (the only comparison on our own protocol is the re-run baseline network)"],
        ["Negative results kept in the record with their reasons (audio-visual fusion, pulse, generated-video heads)",
         "Any Gemini capability; robustness to social-media processing or edited media (not tested)"],
    ], [8.2, 8.2], size=10)
    B.p("Emotion analysis was removed from the scope by the user, and visible-logo or watermark detection is excluded. Chapter 2 summarises the related work, Chapters "
        "3 and 4 describe the system, Chapter 5 the data and the protocol, Chapter 6 the development history, Chapter 7 the results, Chapter 8 the website, "
        "Chapter 9 the limitations, ethics and licences, and Chapter 10 concludes.")


# ---------------------------------------------------------------------------------------------------------------
def ch2(B):
    B.h1("Background and Related Work")
    B.box("This chapter summarises the benchmarks and detector families that the project builds on, and the gap that motivated the protocol. It is deliberately "
          "compact; only sources that were read are cited.")
    B.h2("Benchmarks, Detector Families and the Gap", trace="CONTENT_MAP 17; RESEARCH.md READ and PARTIAL entries; guo2017calibration")
    B.p("<ref tab:related> lists the related work used by this project, what was taken from each source and the gap that remained.")
    B.table("related", "Related work used by the project (cited sources only).", ["Area", "Sources and what they provide", "What this project takes", "Gap left open"], [
        ["Face manipulation in video",
         "FaceForensics++ with four manipulation methods and compression levels <c ffpp>; Celeb-DF v2 <c celebdf> and Celeb-DF++ with newer methods <c celebdfpp>; "
         "DFDC <c dfdc>; DeepfakeBench reports Xception at 0.9450 average frame AUC within FaceForensics++ against 0.7718 across other sets <c bench>",
         "FF++ c23 for training; Celeb-DF family and DFD as unseen sources; Xception <c xception> as re-run baseline",
         "Published numbers rarely share a protocol, so they cannot rank detectors"],
        ["Identity leakage",
         "Face-forgery classifiers trained on FF++ learn identity cues <c identity>; artefact-level features were proposed for forgeries not seen in training <c xray>",
         "Splits that are disjoint by identity and source video",
         "Random splits measure memorisation of faces"],
        ["Generated images",
         "ProGAN training set and CNN-synthesis tests <c cnnspot>; a frozen CLIP linear probe evaluated across generators <c ojha>; a linear support vector machine on CLIP features, trained from few examples of one generator and evaluated on many generators <c cozzolino>; "
         "Community Forensics with thousands of generators <c commforensics>",
         "Frozen CLIP ViT-L/14 <c clip> with a linear probe; Community Forensics subsets after the freeze",
         "Modern commercial generators and real images from many sources"],
        ["Speech",
         "ASVspoof 2019 LA <c asvspoof>; AASIST <c aasist>; wav2vec 2.0 front ends <c w2vspoof>; frozen self-supervised encoders with a simple head <c pascu>; "
         "In-the-Wild, where 33.9 percent EER was reported <c inthewild>",
         "Frozen HuBERT <c hubert>, WavLM <c wavlm> and Whisper <c whisper> encoders with linear heads",
         "Large drop from studio corpora to recordings in the wild"],
        ["Audio-visual and physiological",
         "FakeAVCeleb <c fakeav>; LAV-DF <c lavdf>; audio-visual dissonance <c dissonance>; pulse-based claims <c fakecatcher, hearts, onphys> and the finding that "
         "high-quality deepfakes keep a pulse <c heart>; POS pulse extraction <c pos>",
         "Sealed FakeAVCeleb half; LAV-DF subset; POS for the pulse check",
         "Whether fusion or pulse adds evidence under a controlled protocol"],
        ["Generated video",
         "ReStraV perceptual straightening <c restrav>; DeepAction <c deepaction>; DF26 with closed commercial generators <c df26>",
         "Canonical re-encoding, an appearance-baseline head, DF26 as held-out set",
         "Control for the source of the real clips"],
    ], [2.6, 6.4, 4.1, 3.3], size=9)
    B.p("Temperature scaling fits one scalar on validation logits <c calib>; it calibrates scores only for data like the validation split. The gap this project "
        "addresses is one of protocol, not architecture: a few detectors measured on unseen sources, with intervals, and every rejected idea kept in the record.")


# ---------------------------------------------------------------------------------------------------------------
def ch3(B):
    B.h1("Requirements and System Architecture")
    B.box("This chapter states the requirements, describes the frozen detector pipeline and the post-freeze analysis platform with its eight modules, the report schema, "
          "the per-module verdict layer and the Evaluation Lab.")
    B.h2("Requirements", trace="CONTENT_MAP 4.2, 4.3; D-029, D-044, D-075, D-076")
    B.p("<ref tab:req> lists the requirements and the way the system meets each of them.")
    B.table("req", "Functional and non-functional requirements.", ["Requirement", "How the system meets it"], [
        ["Accept an image, a video (with or without audio) or an audio file", "A probe validates the type, checks that the file decodes and reads the streams; each module states whether it applies"],
        ["Report per check, never one overall verdict", "One result per module; the keys fused, overall_score, overall_probability, probability_fake and verdict are rejected by the contract checker"],
        ["Say what a score means", "Each result carries its score meaning, tested scope, limitations and an uncertainty level; calibration.is_probability is always false"],
        ["Isolate failures", "Modules run one at a time with a time limit; an exception, a timeout or a contract-breaking result gives failed or timed_out for that module only"],
        ["Local and private operation", "Flask server on 127.0.0.1, 512 MB upload limit, uploads deleted after analysis, one worker, refusal at 4 queued or running jobs"],
        ["Refuse a wrong model", "The adapter refuses a checkpoint whose metadata or preprocessing id does not match"],
        ["No test-set tuning", "Sealed splits readable only through a final-flag command that writes a ledger; the Evaluation Lab enforces the same guard"],
    ], [5.4, 11.0], size=10)
    B.h2("Frozen Detector Pipeline", trace="CONTENT_MAP 4.1; E-004, E-005, E-009, E-015; checkpoint metadata")
    B.p("The frozen final system has one scoring path per task (<ref tab:pipeline>). Each path ends in a temperature fitted on the detector's own validation split "
        "and a threshold chosen on that split as the point of maximum balanced accuracy. There is no fusion in the checkpoint.")
    B.table("pipeline", "Scoring paths of the frozen final system.", ["Task", "Input and preprocessing", "Encoder and head", "Temperature (fitted on)", "Threshold"], [
        ["Face video", "16 uniformly spaced frames; YuNet face detector <c yunet>, largest face enlarged 1.3, 224 x 224 crop; clip score is the mean over frames with a face",
         "SigLIP base <c siglip>, fully fine-tuned; linear head", "0.8321 (FF++ c23 validation, 700 clips)", "0.614"],
        ["Generated image", "Whole image through the encoder's processor", "CLIP ViT-L/14, frozen; linear probe", "0.6360 (ProGAN validation)", "0.2073"],
        ["Speech", "First 64,600 samples at 16 kHz after trimming leading and trailing silence and peak normalisation",
         "Mean of logits of HuBERT base, WavLM base plus and Whisper small encoders, each frozen with a linear head", "0.4543 (ASVspoof 2019 LA dev, 24,844 clips)", "0.695"],
    ], [2.0, 5.4, 4.2, 2.8, 2.0], size=9)
    B.h2("Analysis Platform and Its Eight Modules", trace="CONTENT_MAP 4.3, 4.4; D-069, D-072, D-073, D-083")
    B.p("After the freeze the detectors were wrapped, unchanged, in a modular platform (<ref fig:platform>). A coordinator asks every module whether it applies to the "
        "probed file and runs the applicable ones one at a time on the shared GPU, each with its own time limit; a total time limit skips modules not yet started, "
        "and every module appears exactly once in the report with one of the statuses completed, not_applicable, unsupported, failed, timed_out or skipped. "
        "<ref tab:modules> lists the eight modules.")
    B.figure("platform", PLATFORM_FIGURE, "Structure of the analysis platform with its eight modules.", 15.6, "Block diagram of the platform with eight modules in the sections authenticity, cross_modal and provenance", tag="schematic")
    B.table("modules", "The eight modules of the platform.", ["Module", "Section", "What it does", "Tier", "Can it give a finding?"], [
        ["face_video", "authenticity", "Largest face of a video, up to 16 frames, frozen face-video detector", "detector", "Yes (score at or above 0.6139)"],
        ["image_detector", "authenticity", "Still image: largest face crop through the face-video head if a face is found, otherwise the whole image through the image head", "limited", "Yes (0.6139 or 0.2073)"],
        ["speech", "authenticity", "About the first 4 s of an audio file or of a video's audio track, three-encoder average", "limited", "Yes (0.6954); it has no speech check of its own"],
        ["generated_image", "authenticity", "Only when image_detector took the face path: frozen image head on the whole image", "experimental", "No: always inconclusive"],
        ["generated_video", "authenticity", "Canonical re-encode of the first 2 s, 8 frames, frozen DINOv2 small <c dinov2>, mean class token, logistic regression", "experimental", "No: no head is approved"],
        ["lip_sync", "cross_modal", "Audio-visual offset of one tracked face (SyncNet port); consistent inside the band of -6 to +2 frames", "limited", "Cross-modal finding only, weak evidence"],
        ["media_metadata", "provenance", "Container and stream facts", "context", "Information only"],
        ["content_credentials", "provenance", "C2PA manifest validation (c2pa-python 0.38.0)", "provenance", "Valid, invalid or none found; never means authentic"],
    ], [3.1, 2.2, 5.9, 2.2, 3.0], size=9)
    B.h2("Report Schema and Verdict Layer", trace="CONTENT_MAP 4.3; D-076, D-082, D-087 and addendum")
    B.p("The report follows schema 2.1 with three sections (authenticity, cross_modal, provenance). The finding vocabulary is closed: authenticity modules report "
        "indicates_manipulation, no_indication_within_tested_scope or inconclusive; no value means genuine. Every report carries the fixed statement that a module "
        "which found nothing, did not run, was unsupported or was inconclusive does not show that the media is authentic, and that no single probability that the "
        "media is fake is given. Optional explanations are fixed server-side templates over the module's own fields, validated so that they contain no number "
        "absent from those fields; no language model is involved, and they are not saliency or causal explanations.")
    B.p("A verdict layer adds to every completed module an assessment: a label, a tier, the raw score with its kind and threshold, a calibration statement, an "
        "uncertainty level, reasons and evidence references. It re-derives the comparison of score and threshold, and a missing or invalid score or a disagreeing "
        "finding gives Inconclusive. <ref tab:labels> gives the labels in force. There is no report-level verdict and no fused or overall score.")
    B.table("labels", "Labels of the verdict layer and the evidence behind their wording.", ["Module and path", "Label when flagged (uncertainty)", "Label when not flagged (uncertainty)", "Evidence behind the wording"], [
        ["face_video (detector)", "Possible manipulation (high)", "No strong evidence of manipulation detected (high)", "FakeAVCeleb specificity 0.996, sensitivity 0.480; the false-alarm rate of the threshold differed between test sets (Chapter 7)"],
        ["image_detector, face crop (limited)", "Possible manipulation (weak evidence) (high)", "Inconclusive (high)", "Face-video head on real face photographs: false-positive rate 0.2152, false-negative rate 0.6621"],
        ["image_detector, whole image (limited)", "Possible AI-generation (weak evidence) (high)", "Inconclusive (high)", "Modern generators: pooled false-positive rate 0.0418, false-negative rate 0.7395"],
        ["speech (limited)", "Possible manipulation (weak evidence) (high)", "No strong evidence of manipulation detected (moderate)", "FakeAVCeleb audio specificity 0.637; In-the-Wild EER 15.78 percent"],
        ["lip_sync (limited)", "Possible audio-visual mismatch (weak evidence) (high)", "No audio-visual mismatch found (high)", "Offset band validated on 230 held-out real clips"],
        ["generated_image, generated_video (experimental)", "Experimental signal observed, no reliable verdict available (high)", "same", "Source confound and inadequate results (Chapter 6)"],
        ["media_metadata; content_credentials", "Information only; Content credentials verified, invalid or not found", "", "Provenance and context, not detection"],
    ], [3.4, 3.9, 3.9, 5.2], size=9)
    B.h2("Evaluation Lab", trace="CONTENT_MAP 4.3; D-088; TR 10")
    B.p("The Evaluation Lab is a separate dashboard page that re-analyses recorded score files with the project's metrics and group bootstrap. It runs no model, trains "
        "nothing, writes no ledger entry and never touches thresholds, checkpoints or the list of approved heads. A sealed split can only be re-analysed as a final "
        "re-analysis of an evaluation already in the ledger, and only if the score file's SHA-256 equals the ledger entry; thresholds come only from validation rows "
        "or are given with provenance; groups shared with training or validation, duplicate sample ids and duplicate file hashes refuse a final run; and comparisons "
        "use identical samples. A good lab result is not a product approval, and the lab cannot prove that training data excluded the evaluated items when the score "
        "file does not record them.")


# ---------------------------------------------------------------------------------------------------------------
def ch4(B):
    B.h1("Technology Stack and Model Integration")
    B.box("This chapter lists the software and hardware and explains how encoders, heads and checkpoints are integrated.")
    B.h2("Stack", trace="pyproject.toml; EXPERIMENTS.md ENV-DESKTOP-GPU")
    B.p("<ref tab:stack> lists the software and hardware of the system.")
    B.table("stack", "Technology stack.", ["Layer", "Choice", "Role in the system"], [
        ["Hardware", "One desktop with an NVIDIA GeForce RTX 5070 (12 GB)", "All training and inference; base-size encoders only"],
        ["Language and environment", "Python, managed by uv with a lock file", "Reproducible environment"],
        ["Libraries", "PyTorch, timm, Hugging Face Transformers; OpenCV (YuNet), soundfile, ffprobe; c2pa-python 0.38.0", "Encoders, fine-tuning and heads; frame, face and audio handling; Content Credentials validation"],
        ["Web and testing", "Flask on 127.0.0.1, static HTML, CSS and JavaScript; pytest; Microsoft Edge", "Inference page, dashboard and Evaluation Lab; engineering verification (Chapter 8)"],
    ], [3.4, 6.0, 7.0], size=10)
    B.h2("Encoders, Heads and Checkpoints", trace="CONTENT_MAP 4.1, 4.2, 7; E-001 to E-005, E-013; D-029, D-036, D-044, D-049, D-061; TR 9")
    B.p("Encoders were chosen by a measured sweep rather than by reputation. The image shortlist held EfficientNet-B4 <c effnet>, Xception <c xception>, DINOv2 small "
        "and base <c dinov2>, CLIP ViT-L/14 in the OpenAI and LAION versions <c clip> and SigLIP base <c siglip>; the speech shortlist held wav2vec 2.0 base "
        "<c wav2vec2>, WavLM base plus <c wavlm>, Whisper small <c whisper> and HuBERT base <c hubert>. Features were extracted once from the frozen encoders and a "
        "linear head was trained on them (class-balanced cross-entropy, AdamW, three head seeds). Full fine-tuning of "
        "the face-video encoder used AdamW with learning rate 1e-5, weight decay 0.05, batch 32 and 1,800 steps; only base-size encoders fit in 12 GB; "
        "low-rank adapters <c lora> were not used.")
    B.p("The frozen final system is one checkpoint folder whose metadata records each head, its temperature, its threshold and the preprocessing id of its path. "
        "The adapter refuses a checkpoint whose metadata does not match the code. The version-1 web "
        "page and the platform modules call the same adapter, so the platform adds no new detector; on 11 non-sealed "
        "files the largest absolute difference between them was 0.001919 (tolerance 0.02).")


# ---------------------------------------------------------------------------------------------------------------
def ch5(B):
    B.h1("Datasets and Experimental Methodology")
    B.box("This chapter lists the datasets and their roles, explains the group-disjoint splits, the development slices and the sealed tests with their ledger guard, "
          "and defines the metrics, the bootstrap and the three confirmatory tests.")
    B.h2("Datasets", trace="CONTENT_MAP 6; DATA_LOG")
    B.p("<ref tab:data> lists the datasets, the role of each and the terms under which it was used.")
    B.table("data", "Datasets, their use and their terms (all research-only).", ["Dataset", "Task", "Use in this project", "Terms"], [
        ["FaceForensics++ c23 <c ffpp>", "face video", "train 3,600 clips, validation 700, sealed test_in 700 (70 groups)", "non-commercial research and education"],
        ["Celeb-DF v2 <c celebdf>", "face video", "dev_x 1,728 clips (15 identities); sealed test_x 4,801", "non-commercial research; no derived data published"],
        ["Celeb-DF++ <c celebdfpp>", "face video", "sealed test_x, 2,849 fakes from 44 identities (dev identities removed)", "derived data not published"],
        ["Celeb-DF family (derived)", "face video", "sealed test_x for T1 and T3: 7,650 videos (744 real, 6,906 fake), 344 groups (44 identities and 300 single YouTube videos)", "as above"],
        ["DFD (inside FF++)", "face video", "sealed test_x 3,431 videos, 28 actors", "FF++ terms"],
        ["ForenSynths ProGAN and CNN-synthesis <c cnnspot>", "image", "ProGAN train 5,600, validation 2,400, sealed test_in 8,000; CNN-synthesis dev_x 5,998 (2 units), sealed test_x 48,805 (16 groups)", "CC BY-NC-SA 4.0"],
        ["UnivFD diffusion sets <c ojha>", "image", "sealed test_x 10,000 images (10 groups)", "image licence not established"],
        ["ASVspoof 2019 LA <c asvspoof>", "speech", "train 25,380; validation 24,844; sealed test_in 71,237 (67 speakers)", "ODC-By"],
        ["In-the-Wild <c inthewild>", "speech", "dev_x 6,991 clips (12 speakers); sealed test_x 24,788 (42 speakers)", "licence conflict unresolved"],
        ["LAV-DF <c lavdf>", "audio-visual", "subset, 500 videos per cell; sealed test_in 2,000", "CC BY-NC 4.0, non-commercial"],
        ["FakeAVCeleb v1.2 <c fakeav>", "audio-visual", "identity split: dev_x 5,041 clips; sealed test_x 5,013 (250 identities)", "request-form terms not recorded"],
        ["DeepAction v1 <c deepaction>", "generated video", "development only, 2,605 clips", "CC BY 4.0; Pexels licence for reals"],
        ["DF26 <c df26>", "generated video", "held-out evaluation; open subset for leave-one-generator-out training", "evaluation-only licence; no frames in documents"],
        ["Community Forensics <c commforensics>", "image", "Small subset for training (23,807); Eval subset for testing (11,550)", "CC BY-NC-SA 4.0"],
    ], [3.8, 2.2, 6.6, 3.8], size=9)
    B.h2("Splits, Development Slices and Sealed Tests", trace="CONTENT_MAP 6 split protocol; V-002; D-045; E-014")
    B.p("Data were split once and group-disjoint by source identity: the group is the source video together with the person identity, the speaker for speech and "
        "the generator unit for images (<ref fig:splits>). Each training dataset has train, validation and an in-dataset sealed test (test_in). Each cross-dataset "
        "source is divided into a development slice (dev_x), which ranks candidates and may be reused, and a sealed test (test_x). A script checks group "
        "disjointness including content-hash de-duplication; it passed on 80,803 image files and on 41,739 Celeb-DF, In-the-Wild and DFD files.")
    B.figure("splits", os.path.join(FIG, "fig02_splits.png"), "Group-disjoint splits with development slices and sealed tests.", 7.4, "Diagram of the split protocol", tag="schematic")
    B.p("Sealed tests are read once per frozen iteration by one command with a final flag, which writes a ledger and refuses a second read without an explicit "
        "repeat flag and a logged reason. The ledger of the freeze holds 74 entries and no repeat.")
    B.h2("Metrics, Bootstrap and Comparisons", trace="CONTENT_MAP 7; PROTOCOL.md; MATH_NOTES.md; D-012, D-024, D-025, D-061, D-062, D-066")
    B.p("The headline metric is ROC-AUC with a 95 percent group-bootstrap interval (2,000 resamples over groups); equal error rate and balanced accuracy at the "
        "validation threshold are reported where relevant, heads use three seeds and post-freeze proportions use Wilson intervals. In the encoder sweep, candidates whose paired interval against the leader includes zero form a tie set and the "
        "cheapest member wins (extraction time, then parameters).")
    B.p("Three confirmatory tests were declared before the sealed run. T1 compares the final face-video system with the first working version on the sealed Celeb-DF "
        "family set, and T2 does the same for speech on sealed In-the-Wild; each is an improvement if the lower bound of the paired 95 percent interval is above zero. "
        "T3 compares the final face-video system with the re-run Xception baseline on the Celeb-DF family: it is non-inferior if the lower bound is above minus the "
        "margin, and a declared superiority rule asks for a lower bound above zero at the 95 percent level. The margin is 0.0100, set by a floor of 0.01 against twice "
        "the baseline's seed spread. Each test also reports a 98.33 percent interval (Bonferroni over three tests). The 98.33 percent line was declared as a robustness "
        "check beside the 95 percent decision rule; Bonferroni control holds under any dependence between the three tests, so a reader who wants family-wise control "
        "should read the 98.33 percent column. Every other sealed number is exploratory.")


# ---------------------------------------------------------------------------------------------------------------
def _hist_table(B, label, caption, rows):
    B.table(label, caption, ["Step", "Previous version", "Limitation found", "Change", "Result", "Interpretation", "Remaining limitation"], rows,
            [2.3, 1.9, 2.3, 2.5, 3.1, 2.3, 2.0], size=9, keep=False)


def ch6(B):
    B.h1("Development History and Major Improvements")
    B.box("This chapter records how the system reached its frozen form and what happened after the freeze: for every step the previous version, the limitation, the "
          "change, the measured result and its reading. All values before the freeze are exploratory (validation or development slice).")
    B.h2("Before the Freeze", trace="CONTENT_MAP 8.1; E-001 to E-013b, E-016, E-017")
    B.p("<ref fig:ladder> shows the development-slice AUC along the ladder and <ref tab:hist1> records each step. Iterations that did not meet their pre-written "
        "criterion stayed in the record.")
    B.figure("ladder", os.path.join(FIG, "fig08_ladder_results.png"), "Development-slice AUC along the ladder (exploratory, three seeds).", 14.0, "Dot plot of development AUC for the video and speech iterations")
    _hist_table(B, "hist1", "Development ladder before the freeze (exploratory values).", [
        ["Encoder sweep", "none", "No evidence for an encoder choice", "7 image and 4 speech encoders, frozen, ranked under the cost rule",
         "Video: DINOv2 small 0.6933 [0.6639, 0.7314] wins the tie set; image: CLIP ViT-L/14 0.9935; speech: HuBERT 0.6100, inconclusive",
         "Frozen self-supervised and vision-language features beat CNNs on face video (+0.0994 [+0.0228, +0.1723])", "15 identities, 12 speakers: wide intervals"],
        ["I-0 working version", "sweep winners", "No end-to-end system", "Frozen encoder, linear head, temperature, validation threshold, web page",
         "Video val 0.7985, dev_x 0.6933; image dev_x 0.9935; speech val 0.9942, dev_x 0.6100", "The path works end to end",
         "Video underfits (train AUC 0.8367); speech collapses out of corpus"],
        ["I-1 fine-tuning (video)", "I-0 frozen probe", "Underfitting", "Full fine-tuning of 3 encoders at 3 learning rates; SigLIP base, lr 1e-5",
         "SigLIP base linear probe 0.7396 [0.7031, 0.7832] to fine-tuned 0.9096 [0.8776, 0.9388]; paired +0.1700 [+0.1228, +0.2114] (same encoder, development slice, 15 identities)",
         "Supported; retained", "Errors concentrate in one identity"],
        ["Xception baseline", "none", "Need a published-family baseline on the same protocol", "Xception fine-tuned with the same recipe, lr 3e-4 from validation",
         "dev_x 0.8716 [0.8364, 0.9076]; SigLIP minus Xception +0.0380 [+0.0172, +0.0592] (development slice, 15 identities)", "Baseline for T3; margin 0.0100", "Selected rate at the grid edge"],
        ["I-1b augment and mixup", "I-1", "Gap of 0.0803 between validation and dev_x", "Compression, blur and resize augmentation with mixup",
         "dev_x +0.0219 [+0.0090, +0.0382]; validation -0.0194 [-0.0251, -0.0129]", "Does not enter: validation loss beyond the allowance", "A trade-off, not a confirmed gain; not sealed-tested"],
        ["I-1c three-encoder mean (video)", "I-1", "Single encoder", "Mean logit of three fine-tuned encoders",
         "dev_x 0.9192; versus I-1 +0.0096 [-0.0020, +0.0231]", "Does not enter: no gain the interval can show", "Score correlation 0.89 to 0.91"],
        ["I-2 speech trimming", "I-0 speech", "Recording cues (silence, level) consistent with a sign-flipped shortcut; 12 speakers do not establish it", "Trim silence, peak-normalise",
         "dev_x 0.6723; gain +0.0623 [+0.0114, +0.1048]", "Supported; retained", "Loudness alone still separates In-the-Wild"],
        ["I-3 speech augment", "I-2", "Short, padded and saturated clips misclassified", "Noise, low-pass, reverberation, MP3 round trip",
         "dev_x 0.7455; gain +0.0732 [+0.0122, +0.1225]", "Supported; retained", "Gains below about +0.09 not showable with 12 speakers"],
        ["I-4 three encoders (speech)", "I-3 (HuBERT)", "Single encoder", "Mean logit of HuBERT, WavLM base plus, Whisper small",
         "dev_x 0.8130 [0.7523, 0.8808]; gain +0.0675 [+0.0300, +0.0929]", "Supported; final speech system", "Three encoder passes per clip"],
        ["I-5 and I-6 LAV-DF fusion", "frozen heads", "No audio-visual evidence", "Pre-check on LAV-DF, then in-domain heads with logistic-regression fusion",
         "Pre-check visual 0.3838; fused 0.9935, but the visual part within produced videos is at chance, 0.5070 [0.4771, 0.5361] (post hoc control)", "Fusion rests on a production-pipeline cue; rejected",
         "No audio-visual claim"],
        ["I-7 pulse (rPPG)", "none", "Candidate physiological cue", "Signal check on 432 real clips with a shuffled control",
         "0.301 [0.260, 0.346] above threshold, 0.6 required: fails; injected sinusoid recovered in 0.873 of clips", "No recoverable pulse in these data; dropped",
         "No pulse ground truth"],
    ])
    B.p("Appendix A gives the per-cell LAV-DF results behind the fusion row and the positive control behind the pulse row. The freeze (I-8) changed no model: the final system is SigLIP base fine-tuned for face video (I-1), the frozen CLIP image head of I-0 and the three-encoder "
        "speech system of I-4. Every frozen iteration was then scored once on the sealed sets (Chapter 7).")
    B.h2("After the Freeze", trace="CONTENT_MAP 8.3; E-015, E-018 to E-030; D-069 to D-088")
    B.p("<ref tab:hist2> records each study after the freeze; Chapter 7 gives the sealed numbers and their limits.")
    _hist_table(B, "hist2", "Steps after the freeze (labels as in Chapter 7: sealed exploratory, exploratory, post hoc or engineering).", [
        ["Analysis platform", "v1 page with one label per modality", "Wording overstated scores; no tested scope per check", "Modules, coordinator, schema 2.1, dashboard, no overall score",
         "8 modules; 713 tests passed", "Presentation layer, no detector change", "Per-module labels need user review"],
        ["FakeAVCeleb sealed half", "frozen", "No audio-visual sealed set", "Score the sealed identity half once",
         "Visual AUC 0.9552, audio 0.9263; visual sensitivity 0.480 at specificity 0.996; audio specificity 0.637", "Criteria met, not evidence of detection quality; thresholds do not transfer", "Codec shortcut (next row)"],
        ["Codec audit and control", "previous row", "Real and fake videos use different codecs", "Metadata and pixel audit, canonical re-encode, codec-matched control",
         "Metadata-only AUC 0.9985; frozen visual 0.9732 before, 0.8772 after re-encode; codec-matched 0.8856", "Large shortcut, but manipulation evidence remains",
         "Inflation of the sealed AUC not estimated"],
        ["Generated-video heads", "none", "Need a generated-video check", "Canonical re-encode v3; three heads; DF26 scored zero-shot (one look)",
         "Appearance head: DeepAction 0.8281; DF26 0.4957 [0.4755, 0.5168]", "At chance on DF26; installed as experimental only", "Pixel statistics still separate classes"],
        ["Source test and DF26 heads", "appearance head", "Possible source confound", "Score unmodified real clips; retrain on DF26 open generators",
         "Flags 0.7682 of 440 real clips; a head trained on DF26's open generators reached 0.7548 on the commercial generators (second counted look; the installed head was at chance there; different real sets) but flags 0.8023 of real clips", "Confound persists; nothing approved", "Needs real clips from many sources"],
        ["Modern generated images", "frozen image", "Untested on modern generators", "Score 11,550 Community Forensics-Eval images",
         "Macro AUC 0.7039 [0.6855, 0.7229]; misses 0.7395", "Inadequate by the pre-written line", "Format and size confounded with the label"],
        ["Retrained image head", "frozen image", "ProGAN-era training data", "Same recipe on 125 recent generators",
         "Macro AUC 0.8839; +0.1800 [+0.1643, +0.1959]; flags 0.5213 of real images", "Better ranking; not eligible, kept out", "Threshold does not transfer to new real sources"],
        ["Portrait whole-image check", "face path only", "Still images with a face never reached the image head", "Experimental generated_image module",
         "AUC 0.7859 on 3,119 face images; misses 0.8063", "Experimental card, no finding", "Needs a portrait validation"],
        ["Lip-sync band", "earlier tolerance rule", "False alarms on real speech 15.0 and 13.3 percent", "Band of -6 to +2 frames fitted on 230 real clips, validated on 230 others",
         "False alarms 1.3 percent [0.4, 3.8]; Wav2Lip flagged 0 of 160", "Installed as weak cross-modal evidence", "Not a lip-sync deepfake detector"],
        ["Verdict wording", "schema findings", "Text overstated scores; a stronger face label rested on one dataset", "Per-module labels; Possible manipulation for face video",
         "The false-alarm rate of the face-video threshold differed between test sets (Chapter 7)", "Presentation only; thresholds unchanged", "Labels need user review"],
        ["Evaluation Lab", "none", "Re-analysis without test tuning", "Lab on the evaluation safeguards", "57 tests passed; 0.8218 reproduced",
         "Engineering", "Cannot see unrecorded training data"],
    ])
    B.h2("Retained and Rejected Approaches", trace="CONTENT_MAP 11; D-065, D-072, D-074, D-085, D-086")
    B.p("<ref tab:kept> states, for each approach, whether the final system keeps it and why.")
    B.table("kept", "What the final system keeps and what was rejected, with the reason.", ["Approach", "Status", "Reason"], [
        ["SigLIP base, fully fine-tuned (face video)", "retained", "Largest measured gain over the frozen probe of the same encoder; non-inferior to the re-run Xception at both levels (superiority rule met at 95 percent, not after Bonferroni)"],
        ["Frozen CLIP ViT-L/14 probe (images)", "retained, labelled weak", "Strong on ProGAN-era sets; inadequate on modern generators"],
        ["Trim, augment, three-encoder average (speech)", "retained", "Three supported steps on the development slice; sealed T2 improved by the declared rule (part of the gain removes a shortcut, Chapter 7)"],
        ["Augmentation with mixup; three-encoder video mean", "rejected", "Validation loss beyond the allowance; no gain the interval can show"],
        ["Audio-visual fusion on LAV-DF", "rejected", "Driven by a production-pipeline cue (visual 0.5070 within produced videos)"],
        ["Pulse (rPPG) check", "rejected", "Signal check failed (0.301 against 0.6 required)"],
        ["Generated-video heads", "experimental, no finding", "At chance on DF26 or confounded by source; APPROVED_HEADS stays empty"],
        ["Retrained modern-image head", "not installed", "Flags 0.5213 of real images; JPEG re-encoding lowers one family's AUC by 0.1086"],
    ], [5.4, 3.2, 7.8], size=9)
    B.h2("Gemini Investigation", trace="CONTENT_MAP 12; GEMINI_FEASIBILITY.md; E-031 proposed, not run")
    B.p("Interest in modern commercial generators led to an assessment of Gemini-generated images. <b>No Gemini data, model or experiment exists in this project</b>: "
        "nothing was downloaded, trained or evaluated, and the study remains a proposal. It is an unresolved research direction, not a capability. The nearest "
        "measured generator is Google Imagen 3, one of the 21 generators of the Community Forensics-Eval subset (203 generated against 688 paired real images): the "
        "frozen image head reached AUC 0.7691 [0.7352, 0.8020] and missed 0.8719 of the generated images, and the retrained head reached 0.8646 [0.8360, 0.8917] but "
        "flagged 0.5654 of the paired real images. Imagen 3 is not Gemini. Open issues block a study: whether the terms allow Gemini outputs to train a detector, "
        "missing provenance (prompts, dates, model build) in candidate sets, the lack of matched real controls, and the fact that different Gemini versions are "
        "different generators. A future pre-registered study would need one named model with recorded provenance, about 1,000 generated images split by prompt, "
        "real images from at least three sources, format-only baselines and a threshold fixed in advance.")


# ---------------------------------------------------------------------------------------------------------------
def ch7(B):
    B.h1("Evaluation Results and Comparisons")
    B.box("This chapter gives the sealed results, the three confirmatory tests, the speech equal error rates, the FakeAVCeleb reading and the comparisons with "
          "published results. Each number carries its label: confirmatory, sealed exploratory or post hoc.")
    B.h2("Sealed Results", trace="CONTENT_MAP 9.1; E-014, D-067, D-068")
    B.p("The sealed sets were read once for each frozen iteration on 25 October 2026. <ref tab:sealed> gives AUC with 95 percent group-bootstrap intervals over three "
        "seeds. Only the comparisons T1 to T3 on the marked rows are confirmatory; every other value is sealed exploratory.")
    B.table("sealed", "Sealed results (AUC, 95 percent group bootstrap, seeds 0 to 2).", ["Set (samples, groups)", "Working version (I-0)", "Final system", "Re-run Xception", "Label"], [
        ["Celeb-DF family test_x (7,650, 344)", "0.6238 [0.6015, 0.6486]", "0.8218 [0.8020, 0.8393]", "0.8042 [0.7841, 0.8223]", "T1, T3"],
        ["Celeb-DF v2 test_x (4,801, 344)", "0.6322 [0.6018, 0.6658]", "0.8999 [0.8759, 0.9210]", "0.8527 [0.8237, 0.8777]", "exploratory"],
        ["Celeb-DF++ fakes with v2 reals (3,593, 344)", "0.6119 [0.5821, 0.6427]", "0.7105 [0.6877, 0.7331]", "0.7351 [0.7083, 0.7599]", "exploratory"],
        ["DFD test_x (3,431, 28)", "0.5927 [0.5629, 0.6258]", "0.8797 [0.8566, 0.9041]", "0.8847 [0.8655, 0.9052]", "exploratory"],
        ["FF++ c23 test_in (700, 70)", "0.7946 [0.7667, 0.8243]", "0.9899 [0.9826, 0.9959]", "0.9895 [0.9823, 0.9952]", "exploratory"],
        ["In-the-Wild test_x (24,788, 42)", "0.7516 [0.6802, 0.8035]", "0.9148 [0.8896, 0.9383]", "", "T2"],
        ["ASVspoof 2019 LA test_in (71,237, 67)", "0.9684 [0.9564, 0.9777]", "0.9835 [0.9798, 0.9872]", "", "exploratory"],
        ["ProGAN test_in (8,000, 20)", "1.0000 [0.9999, 1.0000]", "same head", "", "exploratory"],
        ["CNN-synthesis test_x (48,805, 16)", "0.9316 [0.8933, 0.9694]", "same head", "", "exploratory"],
        ["Diffusion test_x (10,000, 10)", "0.9354 [0.8967, 0.9665]", "same head", "", "exploratory"],
    ], [4.1, 3.5, 3.5, 3.5, 1.8], size=9)
    B.p("The sealed face-video AUC on the family (0.8218) is below its development-slice value (0.9096); these are different data, not a pair. Along the exploratory "
        "speech staircase on In-the-Wild (0.7516, 0.8125, 0.8672, 0.9148) the point estimates rise; adjacent intervals overlap. Two of five pre-written predictions failed, both upward for speech.")
    B.h2("Confirmatory Tests", trace="CONTENT_MAP 9.1; compare_T3.json; D-091; audit claims_stress items 2 and 3")
    B.p("<ref tab:conf> gives the three tests declared before the sealed run, with the paired differences, the 95 percent intervals and the 98.33 percent robustness line.")
    B.table("conf", "The three confirmatory tests (paired difference of AUC; sealed confirmatory).", ["Test", "Difference [95 percent]", "98.33 percent interval", "Verdict"], [
        ["T1 face video: final minus I-0, Celeb-DF family", "+0.1980 [+0.1688, +0.2242]", "[+0.1627, +0.2303]", "improved, robust"],
        ["T2 speech: final minus I-0, In-the-Wild", "+0.1633 [+0.1177, +0.2266]", "[+0.1085, +0.2481]", "improved, robust"],
        ["T3 face video: final minus re-run Xception, Celeb-DF family (margin 0.0100)", "+0.0176 [+0.0025, +0.0327]", "[-0.0011, +0.0355]",
         "non-inferior at both levels (above -0.0100); 95 percent superiority rule met, not after Bonferroni"],
    ], [5.6, 3.6, 3.0, 4.2], size=9)
    B.p("Against the re-run Xception baseline (AUC 0.8042 [0.7841, 0.8223]) the final video system is non-inferior at both levels (margin 0.0100); the declared 95 percent "
        "superiority rule was met but does not survive the Bonferroni correction (98.33 percent interval [-0.0011, +0.0355]); the baseline ran at 224 px. On the "
        "Celeb-DF++ fakes with the Celeb-DF v2 reals (0.7351 against 0.7105) and on DFD (0.8847 against 0.8797) the baseline is higher, both exploratory (<ref tab:sealed>).")
    B.p("Part of the speech gain is the removal of a shortcut rather than better detection of synthesis: in the training corpus the leading silence and the level of a clip "
        "carry a weak class signal that points the other way on In-the-Wild (single-feature AUC of leading silence 0.375 on ASVspoof development data against 0.719 on the "
        "In-the-Wild development slice; RMS energy 0.594 against 0.248; supported in sign, not in size, with 12 speakers), and the first speech lever, which removes these "
        "cues, accounts for 0.061 of the 0.163 sealed gain on the exploratory staircase (0.7516 to 0.8125); a level cue remains on In-the-Wild after normalisation (RMS AUC "
        "0.135 [0.054, 0.382] on the development slice). Test 2 shows that the final speech system ranks In-the-Wild clips better than the working version; it does not "
        "show how much of that is better detection of synthesis.")
    B.h2("Speech Equal Error Rate and Threshold Behaviour", trace="CONTENT_MAP 9.2, 9.4; E-030; D-084; frozen_rates_output.txt")
    B.p("The final speech system has an equal error rate of 6.94 percent [5.94, 7.79] on ASVspoof 2019 LA and 15.78 percent [12.75, 18.70] on In-the-Wild. At the frozen validation thresholds the final face-video system detects 0.5896 of the family's fakes, and the "
        "speech system flags 0.1716 of real In-the-Wild clips. In a post hoc re-analysis of sealed scores already in the ledger (not confirmatory), the video threshold "
        "flags about 0.11 (0.1098) of the real videos of the Celeb-DF family and about 0.14 (0.1405) of DFD. Scores order files within a source, but thresholds do not "
        "transfer between sources.")
    B.h2("FakeAVCeleb", trace="CONTENT_MAP 9.3; E-015, E-021, E-023")
    B.p("On the sealed identity half of FakeAVCeleb (sealed exploratory, one read) the frozen system reached a visual AUC of 0.9552 [0.9442, 0.9657] and an audio "
        "AUC of 0.9263 [0.9149, 0.9366], which met their pre-stated criteria. They are not evidence of detection quality. In this data set the video codec coincides with "
        "the manipulation method (fake videos are mpeg4, real ones mostly h264): codec metadata alone separates real from fake video at 0.9985 on the "
        "development half, and real and fake audio come from different recording channels. At the frozen thresholds the visual sensitivity was 0.480 at a specificity of "
        "0.996, so about half of the fake videos are missed, and the audio specificity was 0.637; thresholds do not transfer. A codec-matched control left manipulation "
        "evidence by the pre-written rule. How much the sealed 0.9552 is inflated was not estimated.")
    B.h2("Comparisons with Published Results", trace="CONTENT_MAP 10; PERFORMANCE_BENCHMARK.md; D-091; audit claims_stress item 6; E-020, E-027, E-030")
    B.p("No published number is directly comparable with ours: the only comparison on our own protocol is the re-run Xception baseline (T3), which shares the data, the "
        "split, the training recipe and the metric. <ref tab:pub> sets the published values beside ours as context only. Published values carry no intervals, so no "
        "significance test is possible, and no difference is computed.")
    B.table("pub", "Published results as context (no ranking of published values).", ["Comparison", "Ours", "Published", "Status and reading"], [
        ["ASVspoof 2019 LA pooled EER, final speech system and AASIST <c aasist>", "6.94 percent [5.94, 7.79]", "1.13 percent, as quoted by its authors", "Context only; AASIST was not re-run here"],
        ["Same, frozen XLS-R 2B with logistic regression, and RawNet2 <c pascu>", "6.94 percent", "0.5 percent; 5.9 percent", "Context only; not re-run; no sign claimed"],
        ["In-the-Wild EER, ours on 42 of 54 speakers <c inthewild>", "15.78 percent [12.75, 18.70]", "33.9 percent; 7.2 percent for XLS-R 2B <c pascu>, full set", "Different test set, no ranking"],
        ["DF26 zero-shot pooled AUROC, generated-video head <c df26>", "49.57 [47.55, 51.68]", "44.0 to 69.7 for nine FF++-trained detectors, no intervals", "Context only; training data and frame sampling differ, so no ordering is drawn"],
    ], [5.2, 3.2, 3.6, 4.4], size=9)
    B.p("AASIST was not re-run, and whether it transfers better or worse than our system was not measured on a common protocol. On DF26 without DF26 training our head "
        "is at chance (0.4957 [0.4755, 0.5168]). A head trained on DF26's open generators reached 0.7548 on the commercial generators (one counted look, the second look "
        "at that subset); the installed head was at chance there, and the two used different real sets. Other published values (UnivFD diffusion mean average precision "
        "95.00 against our 0.9387 <c ojha>; DeepfakeBench frame-level Xception 0.7365 on Celeb-DF v2 <c bench>; BA-TFD 0.990 on the full LAV-DF <c lavdf>) differ in "
        "data or protocol and are context only.")


# ---------------------------------------------------------------------------------------------------------------
def ch8(B):
    B.h1("Website and User Interface")
    B.box("This chapter describes the local dashboard, the per-module cards and labels, what a score means on the page, and the software tests that verify the "
          "engineering (not the accuracy) of the system.")
    B.h2("Dashboard", trace="CONTENT_MAP 4.3 dashboard; D-075, D-076, D-082")
    B.p("The dashboard is a Flask application on 127.0.0.1 with an upload-first page. In live mode it runs one analysis at a time, refuses new work when 4 jobs are "
        "queued or running, accepts files up to 512 MB and deletes uploads after the analysis. The report page first summarises the file, then groups the checks: "
        "merits attention, nothing found within tested scope, credential verified, inconclusive, file information, experimental, did not apply, not available and did "
        "not complete (<ref fig:dash1>). The redesigned front end (a cream, beige and warm-brown page served over the unchanged backend) asks what kind of file is "
        "analysed, which only filters the file picker; a donut counts the checks of the report by outcome and is labelled as a count, not a probability, a score or a "
        "verdict. A detector card shows its model output as a bar against its decision line, labelled not a probability; experimental numbers stay under Technical "
        "details. Demonstration scenarios (11 fixtures) run only on request and are marked MOCK. Every text colour pair meets WCAG AA contrast, checked by a script.")
    shots = DASHBOARD_SCREENSHOTS
    for i, (path, cap, alt) in enumerate(shots):
        B.figure("dash%d" % (i + 1), path, cap, DASHBOARD_WIDTH_CM, alt, tag="screenshot")
    B.h2("Per-Module Cards and Labels", trace="CONTENT_MAP 4.3 labels; D-087 addendum; TR 12")
    B.p("Each completed module has a card with its label (<ref tab:labels>), supporting signals, the model, limitations and an uncertainty statement (<ref fig:dash2>). "
        "When the face-video score reaches its threshold the card reads <b>Possible manipulation</b> with high uncertainty, because the false-alarm rate of that "
        "threshold differed between test sets (Chapter 7). "
        "Experimental modules always read Experimental signal observed, no reliable verdict available, whatever their score. No card says that media is authentic.")
    B.p("The frozen version-1 page, kept unchanged as history, showed one label per file (Likely fake, Likely real or Undetermined) and the sentence “Calibrated "
        "probability that this content is fake”. The platform replaced that wording because the later measurements showed that the scores are not probabilities on "
        "other sources.")
    B.h2("Score Semantics", trace="CONTENT_MAP 5; D-087")
    B.p("<ref tab:semantics> states what each kind of score and label means, so that no number on the page is read as the probability that a file is fake.")
    B.table("semantics", "What a score means in the system.", ["Concept", "Meaning", "Where it applies"], [
        ["Temperature-scaled score", "A temperature fitted on the detector's own validation split; a probability only for content like that split", "face_video; image_detector whole-image path"],
        ["Exploratory calibration", "Temperature fitted on ASVspoof dev, but calibration error out of corpus rose after scaling", "speech (marked as not calibrated)"],
        ["Ranking score", "Only the order of scores is reliable; not a probability", "generated_image, generated_video; any detector on unlike sources"],
        ["Threshold finding", "indicates_manipulation if the score reaches the validation threshold, otherwise no indication within tested scope", "face_video, image_detector, speech"],
        ["Verdict label", "Module-specific wording with an uncertainty level; never report-level", "every completed module"],
    ], [3.8, 7.8, 4.8], size=9)
    B.h2("Software Testing", trace="CONTENT_MAP 15; TESTING_RECORD sections 9, 10, 12, 13")
    B.p("Testing verifies that the software behaves as specified; it says nothing about detection accuracy. <ref tab:tests> lists the latest recorded results.")
    B.table("tests", "Software verification (engineering, not accuracy).", ["Check", "Result"], [
        ["Full safe test suite, GPU hidden", "713 passed, 2 skipped (opt-in GPU tests), 13 deselected (data-dependent tests), exit 0"],
        ["Focused suites: verdict, explanation and pipeline; generated_image constraints; Evaluation Lab", "112 passed; 7 passed; 57 passed"],
        ["Opt-in GPU tests; wrapper parity on 11 non-sealed files", "5 passed; largest absolute difference 0.001919 (tolerance 0.02)"],
        ["Live browser demonstration in Microsoft Edge with real detectors, 8 inputs from non-sealed sources", "8 of 8 as specified, 0 page errors, no horizontal overflow at 390 px"],
        ["Dependency lock check", "exit 0"],
    ], [10.0, 6.4], size=10)
    B.p("Defects found by testing and fixed include a failed live analysis that left the MOCK report on screen and a report overview hidden under the sticky header. "
        "Two behaviours are documented and not changed: the speech module scores any sound (a 440 Hz tone receives a finding), and the generated-video score is 1.0 "
        "on a real LAV-DF clip and on a synthetic test pattern. Not run: the 13 data-dependent tests, a fresh clone, other browsers and screen readers.")


# ---------------------------------------------------------------------------------------------------------------
def ch9(B):
    B.h1("Limitations, Ethics and Licensing")
    B.box("This chapter consolidates the limitations of the evidence, the ethical position of the product and the licence terms that restrict how the data, "
          "weights and results may be used.")
    B.h2("Limitations", trace="CONTENT_MAP 13; E-032 (post hoc rates)")
    B.numbered([
        "Small development slices (15 identities, 12 speakers, 2 generator units) give wide intervals; every iteration verdict is exploratory.",
        "Sealed video intervals rest on 344 groups (44 identities and 300 single YouTube videos) with 744 real against 6,906 fake videos from two sources; speech on 42 speakers. Every sealed number is one run per frozen iteration.",
        "T3 uses one baseline family at one input size (224 px), with the baseline learning rate at the grid edge; the system is non-inferior at both levels, and the superiority rule is met at the 95 percent level only, not after Bonferroni.",
        "Part of the sealed speech gain removes a silence and level shortcut (Chapter 7), a level cue remains on In-the-Wild, and the development evidence for the shortcut rests on 12 speakers.",
        "Thresholds do not transfer: the face-video system detects 0.5896 of the family's fakes and 0.480 of FakeAVCeleb fakes, and in a post hoc re-analysis of sealed scores already in the ledger (not confirmatory) it flags about 0.11 of the real videos of the Celeb-DF family and about 0.14 of DFD; scores are not probabilities on other sources.",
        "Only base-size encoders fit the 12 GB card; published leaders use larger adapted encoders.",
        "Dataset artefacts inflate AUC: codecs in FakeAVCeleb, container and colour tags in DeepAction, file format and size in Community Forensics-Eval, a production cue in LAV-DF. Controls reduce but do not remove them.",
        "No audio-visual detection, no pulse signal, and every generated-video head is source-confounded or at chance.",
        "There is no directly comparable result for face video against a published detector, and edited images, social-media processing and other languages were not tested.",
        "Some pre-registrations of the post-freeze studies were hashed into result files but not committed before the runs, which is weaker than a commit.",
    ])
    B.h2("Ethics", trace="CONTENT_MAP 14 product ethics")
    B.p("The product never says that media is authentic, gives no overall verdict, and never lets an experimental signal decide. A valid Content Credential concerns "
        "the origin of a file relative to an operator-configured trust anchor, not the truth of its content. Every label carries its uncertainty because a wrong flag "
        "can harm the person shown in a file. Detection results describe a file, never a person's intent.")
    B.h2("Licensing", trace="CONTENT_MAP 14; DATA_LOG; D-038, D-070")
    B.p("<ref tab:lic> lists the licence terms that limit how the data, weights and results may be used.")
    B.table("lic", "Licence terms and what they restrict.", ["Item", "Terms", "Consequence"], [
        ["Celeb-DF, Celeb-DF++, FakeAVCeleb", "Research only; no publication of derived data (FakeAVCeleb form terms not recorded)", "No frame, crop or per-video score in any document"],
        ["LAV-DF; DF26", "Non-commercial; DF26 evaluation-only with no redistribution", "Aggregate results only; no frames"],
        ["Community Forensics", "CC BY-NC-SA 4.0", "No images in public deliverables; retrained heads are research artefacts, not installed"],
        ["In-the-Wild; diffusion test images", "Licence conflict or not established", "Recorded as open questions"],
        ["OpenAI CLIP ViT-L/14; WavLM base plus", "No licence stated (CLIP) or conditional (WavLM); treated as research-only", "Checkpoints cannot be published or used commercially"],
        ["SyncNet v2 weights", "Weights licence not stated by the source", "Used for an academic prototype only"],
        ["SigLIP, DINOv2, HuBERT, Whisper, YuNet, c2pa-python", "Permissive (Apache-2.0 or MIT)", "No added restriction"],
    ], [4.4, 6.0, 6.0], size=9)


# ---------------------------------------------------------------------------------------------------------------
def ch10(B):
    B.h1("Conclusion and Future Work")
    B.box("This chapter summarises what was achieved and what remains open.")
    B.h2("Conclusion", trace="CONTENT_MAP 2, 9")
    B.p("The project delivered a local system that analyses images, videos and speech, and a complete record of the path. Under the protocol the final face-video system improved, by the declared rule, over the first working version "
        "on the sealed Celeb-DF family (+0.1980 [+0.1688, +0.2242]) and the speech system improved on sealed In-the-Wild (+0.1633 [+0.1177, +0.2266]), although part of "
        "the speech gain removes a silence and level shortcut and is not all better detection of synthesis. Against a re-run Xception baseline the face-video system is "
        "non-inferior at both levels (margin 0.0100); the declared 95 percent superiority rule was met but does not survive the Bonferroni correction (98.33 percent "
        "interval [-0.0011, +0.0355]), and the baseline ran at 224 px.")
    B.p("The platform reports what each of its eight modules found within its tested scope, with cautious labels and no overall score. Later studies mostly added "
        "limits. On FakeAVCeleb the frozen system's AUCs met their pre-stated criteria but are not evidence of detection quality (codec metadata alone gives 0.9985 "
        "on the development half) and the thresholds do not transfer. On LAV-DF a near-perfect fused score rested on a production cue (its visual part is at chance "
        "within the produced videos), so no audio-visual detection is claimed. Modern generated video and images remain unsolved.")
    B.h2("Future Work", trace="CONTENT_MAP 13 future work; PB 4.1 to 4.5, 13.3, 13.9; D-083, D-086")
    B.bullets([
        "Score the frozen face system on the official Celeb-DF v2 test list, next to the Xception baseline.",
        "Parameter-efficient adaptation of a larger CLIP encoder; a larger frozen speech encoder; a lip-sync model of the full time-delay distribution with a false-alarm control.",
        "Generated-video training with real clips from many sources (the source-confound test as acceptance test); for modern images a threshold calibrated on diverse real sources and a portrait validation; a Gemini study once its blockers are resolved.",
    ])


# ---------------------------------------------------------------------------------------------------------------
def appendix_a(B):
    B.h1("Supporting Evidence for the Negative Results", kind="appendix",
         trace="Appendix restored in Phase B (report_audit.md marks it appendix). VIVA_PREP questions answered: 'Why not claim audio-visual detection from the fused LAV-DF AUC of 0.9935?'; "
               "'Why does the FF++-trained visual head have AUC 0.38 on LAV-DF, below chance?'; 'Why was rPPG dropped?'; 'The pulse check failed; how do you know the pipeline can see a pulse at all?'")
    B.p("This appendix holds the detail behind two negative results of Chapter 6, because an examiner asking why audio-visual detection and a pulse check were dropped "
        "needs the per-cell numbers. All values are exploratory (validation split of LAV-DF, or 432 real clips).")
    B.h2("LAV-DF per-Cell Results and the Production-Pipeline Control", plain="A.1", trace="E-010, E-011 and its post hoc control; VIVA_PREP: LAV-DF fusion and the sign reversal of the FF++ head")
    B.p("<ref tab:lavdf1> gives the AUC of the frozen face-video and speech heads for each kind of fake against the real originals. The visual head is below chance on "
        "every cell, including the audio-only fakes whose frames are untouched, so it responds to something other than the manipulation.")
    B.table("lavdf1", "LAV-DF validation, frozen heads (exploratory): AUC of each kind of fake against the real originals.", ["Kind of fake", "Visual head", "Audio head"], [
        ["video fake", "0.2788 [0.2488, 0.3087]", "0.5564 [0.5215, 0.5895]"],
        ["audio fake (frames untouched)", "0.2876 [0.2568, 0.3174]", "0.6804 [0.6464, 0.7119]"],
        ["both fake", "0.2692 [0.2393, 0.2986]", "0.6708 [0.6378, 0.7029]"],
    ], [5.6, 5.4, 5.4], size=10)
    B.p("<ref tab:lavdf2> gives heads trained on LAV-DF, whose fused score reached 0.9935 [0.9905, 0.9962]. Its AUCs are computed within the produced videos only, "
        "so that every clip went through the data set's own pipeline (a control run after the result, not a declared criterion). The visual head separates the real "
        "originals from the audio-only fakes at 0.9912 [0.9864, 0.9951] although their frames are untouched, and is at chance for its own label within the produced "
        "videos; the audio head is the only one above chance there.")
    B.table("lavdf2", "LAV-DF validation, heads trained on LAV-DF: AUC within the produced videos only (post hoc control).", ["Score", "Label asked", "AUC [95 percent]"], [
        ["visual head", "video manipulated", "0.5070 [0.4771, 0.5361]"],
        ["audio head", "audio manipulated", "0.8574 [0.8394, 0.8752]"],
        ["visual head", "audio manipulated (should be near 0.5)", "0.5029 [0.4743, 0.5320]"],
        ["audio head", "video manipulated (should be near 0.5)", "0.3330 [0.3059, 0.3602]"],
    ], [3.4, 7.0, 6.0], size=10)
    B.h2("Pulse Check and its Positive Control", plain="A.2", trace="I-7 S-1 pulse check and M-005; VIVA_PREP: why rPPG was dropped, and how the pipeline is known to see a pulse")
    B.p("The pulse check asked whether a pulse-band signal from the skin of 432 real clips exceeds the 95th percentile of a time-shuffled control in at least 0.6 of "
        "the clips. It did in 0.301 [0.260, 0.346] (0.280 on FaceForensics++, 0.288 on Celeb-DF v2 and 0.333 on LAV-DF), so the check failed: a pulse signal "
        "is not recoverable from these data. <ref tab:ppg> shows that the pipeline can see a clean sinusoid. A sinusoid of 1.2 Hz was injected into the skin pixels "
        "of the same clips after decoding. It shows the sensitivity of the pipeline to a clean sinusoid of that size; it does not show that a real pulse exists in these "
        "videos or survives compression.")
    B.table("ppg", "Positive control for the pulse check: fraction of the 432 clips in which an injected sinusoid is recovered.", ["Relative green amplitude", "Recovered [95 percent Wilson]"], [
        ["0 (control, reproduces the pulse check)", "130/432 = 0.301 [0.260, 0.346]"],
        ["0.002", "148/432 = 0.343 [0.299, 0.389]"],
        ["0.004", "236/432 = 0.546 [0.499, 0.593]"],
        ["0.008", "377/432 = 0.873 [0.838, 0.901]"],
        ["0.016", "420/432 = 0.972 [0.952, 0.984]"],
    ], [8.2, 8.2], size=10)
