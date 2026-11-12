"""Chapters 1 and 2: introduction and literature survey."""


def intro(B):
    B.h1("Introduction")
    B.box("This chapter presents the background and the motivation of the project, states the problem and its scope, lists the objectives, describes the way the work was "
          "organised and gives the outline of the report.")
    B.h2("Background", trace="DECISIONS D-001, D-002, D-011, D-018, D-019; RESEARCH R-001 to R-003, R-017, R-033")
    B.p("Deepfakes are images, videos and audio clips that a learned model has generated or altered so that they show a person doing or saying something that did not happen, "
        "or show a scene that never existed. Five kinds of manipulation matter for this project: a face in a video replaced by another face (face swap); a face whose expression and head "
        "movement are driven by another person (re-enactment); a face animated from a still image and a speech signal (audio-driven talking face); a whole image produced by a generative "
        "network, either a generative adversarial network (GAN) or a diffusion model; and speech that is synthesised from text or converted from another speaker. A video can also be "
        "manipulated in its picture and in its sound separately, so that one modality is real and the other is not.")
    B.p("The research community has built benchmark data sets for each of these cases. FaceForensics++ holds 1,000 real videos and 4,000 videos manipulated with four methods at three "
        "compression levels <c ffpp>; Celeb-DF contains 590 real videos of 59 celebrities and 5,639 face-swap videos <c celebdf>; the DeepFake Detection Challenge data set adds over "
        "100,000 clips of paid actors <c dfdc>; and for speech the ASVspoof 2019 database provides synthesised, converted and replayed attacks <c asvspoof>. Detectors trained on these sets "
        "reach high scores on test data drawn from the same set.")
    B.p("The scores fall when the test data come from another source. In the DeepfakeBench benchmark, a plain Xception network trained on FaceForensics++ at the c23 compression level "
        "reaches a mean AUC of 94.5 over the six tests inside FaceForensics++ and 77.2 over eight other data sets (AUC in percent, frame level, the benchmark’s own protocol) <c bench>. "
        "For speech, one study that re-implemented twelve architectures under one protocol found a lowest equal error rate of about 1.2 percent on the evaluation split of ASVspoof 2019 "
        "and a lowest of about 33.9 percent on its separate In-the-Wild set, reached by a different model <c inthewild>. On the DeepFake Detection Challenge, the best models on content "
        "that was not made for the challenge reached an average precision of 75.3 and a ROC-AUC of 73.4 (percent) <c dfdc>. These numbers come from different protocols and are context only: "
        "they are never compared with the results of this project.")
    B.h2("Motivation", trace="DECISIONS D-004, D-005, D-012; RESEARCH notes 1, 10, 12")
    B.p("Three observations from the literature shaped the project. First, reported numbers rarely share a protocol, so a table of published scores cannot show whether a new detector is "
        "better or worse than another; only a re-run of both under one protocol can. Second, binary face-forgery classifiers trained on FaceForensics++ learn information about the identity "
        "of the person in the picture <c identity>, so a split that lets a person appear on both sides of the boundary measures memorisation of faces. Third, a full-text search of eight "
        "detector papers read for this survey found no mention of seeds, standard deviations or confidence intervals, although the seed-to-seed spread in audio detection reaches several points of "
        "equal error rate <c inthewild>.")
    B.p("There is also a practical motivation. A person who receives a suspicious file needs a probability that is calibrated, so that 0.9 means what it says, and needs to know which "
        "modality carries the evidence: the picture, the sound or both. The system therefore returns a separate visual and audio score for a video instead of a single opaque number.")
    B.p("Finally, the project is a final-year project and its purpose is to show a documented path: a simple working version first, then at least seven iterations that each improve "
        "something for a stated reason, ending at results that can be compared with published detectors on a protocol that the project defines and runs itself. The goal is not the best "
        "model in the world. The report is written for a reader who asks “why this and not that”, so each choice is tied to a written decision and, where one exists, to an experiment.")
    B.h2("Problem Statement and Scope", trace="DECISIONS D-002, D-011, D-018, D-019, D-033; DOC_FACTS section 2")
    B.p("Given an image, a video with its audio track, or a standalone audio clip, the system returns the probability that the content is fake, and states which modality carries the "
        "evidence. Fake means a swapped, re-enacted or audio-driven face, a fully generated image (GAN or diffusion), synthesised or converted speech, or audio and video manipulated "
        "separately. <ref tab:scope> lists what is returned for each input.")
    B.table("scope", "Inputs, outputs and claims of the system.",
            ["Input", "What is returned", "What is claimed"],
            [["Still image", "One visual score. A face crop is scored if a face is found, otherwise the whole image", "A score calibrated on the validation data of its path (the face-crop path is not calibrated for still images); the path taken is reported"],
             ["Video with an audio track", "A visual score and an audio score, shown side by side", "Each score for its own modality; no combined score is claimed until the combining rule is decided and tested"],
             ["Video without audio", "A visual score; the audio score is reported as unavailable, with the reason", "A visual score only"],
             ["Standalone audio file", "One audio score", "A score for fake speech with an exploratory calibration: a ranking score, not a reliable probability"]],
            [3.4, 6.6, 6.4], ["left", "left", "left"], size=10)
    B.p("The system does not locate a manipulation in time, because temporal labels are not available in every data set. It makes no claim about kinds of content that its training and "
        "test data do not cover. The frozen web page of the working version states, in a fixed sentence next to every score, that the score is a calibrated probability from a model fitted on the training data of this project, "
        "valid only for the kinds of content in that data, and that it is not proof. Calibrated here means that a temperature was fitted on the validation split of each task; "
        "on content unlike that split the number is a score, not a probability, which the later measurements on other sources show and the analysis platform states.")
    B.p("The analysis platform built after the freeze keeps these scores but presents each one as the finding of a module within its tested scope, adds modules for lip synchronisation, "
        "Content Credentials and generated video, and gives no single overall probability that the media is fake.")
    B.h2("Objectives", trace="DECISIONS D-003 to D-006, D-016, D-025, D-041")
    B.p("The objectives of the project work are the following.")
    B.numbered([
        "Build the whole system from zero in one repository: data preparation, an evaluation harness, a detector for each of the three tasks (face video frames, generated images, speech audio), an inference library and a local web page.",
        "Define one evaluation protocol before any model exists, with group-disjoint splits, a development slice for ranking, sealed test sets and a guard that refuses test data unless a run is flagged as final.",
        "Compare candidate pretrained encoders under that protocol, with a rule that prefers the cheaper candidate when the evidence does not separate them.",
        "Develop the system in iterations, each starting from a written hypothesis, a success criterion and a failure criterion, and keep failed iterations in the record.",
        "Test remote photoplethysmography, the pulse visible in skin colour, as a candidate improvement and adopt it only if the evidence on the development slice supports it.",
        "State “comparable with published detectors” in a checkable way: non-inferior, on the same protocol, to a re-run of a published detector.",
        "Wrap the frozen detectors in an analysis platform that reports one finding per module with its tested scope and limitations, never a single overall probability that the media is fake, and add further modules (lip synchronisation, Content Credentials, generated video) only with the status that their evidence supports.",
        "Produce the software, an IEEE-format paper, this report and a presentation, all from the same logged evidence.",
    ])
    B.h2("Approach and Constraints", trace="DECISIONS D-007, D-009, D-016, D-017; PLAN; CLAUDE.md section 2")
    B.p("The project runs over seven calendar days, from 24 to 30 October 2026, on one desktop machine with one NVIDIA GeForce RTX 5070 GPU of 12 GB. The schedule was organised so that every "
        "iteration is complete by itself: hypothesis, run, result, decision. An iteration that "
        "is not finished by the end of Wednesday is reported as not run, not as a result. <ref fig:timeline> shows the plan.")
    B.figure("timeline", "fig06_timeline.png", "Plan of the seven days to submission, 24 to 30 October 2026. An iteration not finished by the end of Wednesday is reported as not run.", 15.5,
             "Gantt-style chart of the seven working days from Saturday 24 to Friday 30 October 2026", tag="schematic")
    B.p("Five working rules governed every step. Nothing unmeasured is written as a result: each number in a document comes from a logged experiment, and observations that are not "
        "measurements are labelled as such. The hypothesis, the success criterion and the failure criterion of every iteration and comparison are written before the run. The test data are "
        "never used for tuning: data are split once, by group, tuning uses only training and validation data, and the sealed test sets are touched once per frozen iteration. "
        "Candidates are compared on the same data, split, budget and metric, with several seeds, the mean, the standard deviation and a bootstrap interval; if the intervals of the "
        "difference include zero, the cheaper candidate is preferred and the report says so. Failed runs stay in the log.")
    B.h2("Organisation of the Report and Outline of the Next Chapter")
    B.p("Chapter 2 surveys the literature and ends with the research gap that the objectives address. Chapter 3 gives the mathematical background: the metrics, the group bootstrap, the "
        "paired and non-inferiority comparisons and the argument behind each lever of the iteration ladder. Chapter 4 describes the implementation and the experimental design, from the "
        "environment and the data to the web application, the analysis platform, the software tests, the test protocol, the sealed-test results and the experiments after the freeze. Chapter 5 gives the conclusions and the further work. The references and two appendices follow.")


def survey(B):
    B.h1("Literature Survey")
    B.box("This chapter presents the literature that the project builds on: the data sets and benchmarks, the visual, image, speech and audio-visual detectors, the evidence on the "
          "pulse signal, and the work on evaluation and adaptation. It ends with the research gap and the objective of the work.")
    B.h2("Face-video Datasets and Benchmarks", trace="RESEARCH R-001, R-002, R-003, R-017, R-053")
    B.p("FaceForensics++ is the standard training set for face forgery <c ffpp>. It contains 1,000 real videos and about 1.8 million fake images from 4,000 fake videos made with "
        "four methods (DeepFakes, Face2Face, FaceSwap and NeuralTextures) at three compression levels, raw, c23 and c40. Its fixed split uses 720, 140 and 140 videos for training, "
        "validation and test, and the faces are cropped with an enlargement factor of 1.3 before classification, a value that the preprocessing of this project reuses. With the in-dataset "
        "test videos, per-frame accuracy of an XceptionNet on cropped faces was 99.26, 95.73 and 81.00 percent at the three levels, which shows the sensitivity to compression.")
    B.p("Celeb-DF was built to be harder: 590 real videos of 59 celebrities and 5,639 face-swap videos made with an improved synthesis pipeline <c celebdf>. Nine existing "
        "detectors, run with their released models and without retraining, reached a frame-level AUC of 65.3 (Xception trained at c23) and lower for others, and Celeb-DF gave the lowest "
        "average AUC of the data sets compared. The DeepFake Detection Challenge data set contains over 100,000 clips of 3,426 paid and consenting actors, with a subject-disjoint "
        "public test set <c dfdc>; on content that was not made for the challenge the log loss of the best models was close to the value of a constant prediction.")
    B.p("Celeb-DF++ extends Celeb-DF with 22 recent generation methods, namely 8 face-swap methods including the original, 7 re-enactment methods and 7 talking-face methods, over the same 590 real "
        "videos of 59 identities, and defines three generalisation protocols <c celebdfpp>. A detector trained on FaceForensics++ lost on average 5.2 points of frame-level AUC on "
        "Celeb-DF++ compared with Celeb-DF. DeepfakeBench unifies 15 detectors, 9 data sets, one preprocessing pipeline and one set of metrics <c bench>; trained on FaceForensics++ c23, a "
        "plain Xception and an EfficientNet-B4 had the same mean cross-domain AUC (77.2) and were within one to two points of more elaborate detectors, which suggests that the backbone "
        "and the training recipe matter at least as much as the method name.")
    B.table("context", "Published results quoted as context. Each row names the protocol under which it was measured; none is a result of this project and no two rows are comparable.",
            ["Source", "Protocol of the source", "Quantity", "Value"],
            [["FaceForensics++ <c ffpp>", "In-dataset, XceptionNet, c23, per frame", "Accuracy (percent)", "95.73"],
             ["Celeb-DF <c celebdf>", "Released detectors, not retrained, Celeb-DF", "Frame AUC of Xception-c23 (percent)", "65.3"],
             ["DeepfakeBench <c bench>", "Train FF++ c23; plain Xception", "Mean AUC within FF++ / across 8 sets (percent)", "94.5 / 77.2"],
             ["LipForensics <c lips>", "Train FF++, cross-dataset, video level", "AUC on Celeb-DF / DFDC (percent)", "82.4 / 73.5"],
             ["Effort <c effort>", "Train FF++ c23, video level, 7 data sets", "Mean AUC (percent)", "91.7"],
             ["Ojha et al. <c ojha>", "Train ProGAN, test unseen diffusion and autoregressive models", "Mean AP, linear probe on CLIP / trained baseline", "95.00 / 75.51"],
             ["AASIST <c aasist>", "ASVspoof 2019 LA evaluation", "EER, best seed / mean of 3 seeds (percent)", "0.83 / 1.13"],
             ["Muller et al. <c inthewild>", "Train ASVspoof 2019, test In-the-Wild", "Lowest EER over 12 models (percent)", "about 33.9"],
             ["FakeAVCeleb <c fakeav>", "Own test set, three fake categories", "Best audio-only AUC / video-only AUC (percent)", "about 97.8 / 93.3"],
             ["LAV-DF <c lavdf>", "Own test split, identity-independent", "Classification AUC of its multimodal method (percent)", "99.0"]],
            [3.8, 5.0, 5.0, 2.6], ["left", "left", "left", "center"], size=10)
    B.h2("Visual Detectors and the Generalisation Problem", trace="RESEARCH R-004, R-017, R-018, R-021, R-034, R-035, R-044")
    B.p("Face X-ray predicts the blending boundary between a pasted face and its background and is trained on self-made blended real images, so that no fake from any generator is "
        "needed <c xray>. Trained on blended images only it reached a cross-dataset AUC on Celeb-DF of 74.76, against 36.19 for an Xception trained on FaceForensics++ (percent). "
        "Self-blended images take the idea further: a real face is copied, augmented and blended back with a deformed mask, and an EfficientNet-B4 trained on such images from real "
        "FaceForensics++ videos only reached a video-level AUC of 93.18 on Celeb-DF v2 in its own paper <c sbi>. The same paper states that the method fails on whole-image synthesis "
        "and reports one run per entry with no seeds or intervals.")
    B.p("LipForensics pretrains a network for lipreading and fine-tunes only its temporal part, with the feature extractor frozen <c lips>. In its ablation, the frozen extractor "
        "gave an accuracy of 87.5 and 90.4 percent on two unseen manipulation sets, against 83.2 and 84.6 when the whole network was fine-tuned; this is evidence that more trainable "
        "capacity does not always transfer better, which the fine-tuning iteration of this project tests directly. Dong et al. showed with a linear probe on frozen detector "
        "features that binary classifiers trained on FaceForensics++ encode identity <c identity>, which is the reason why the splits of this project are disjoint in identity as well "
        "as in source video.")
    B.p("Adapting a large frozen vision-language encoder has recently given strong cross-dataset numbers. The Effort detector decomposes the attention weights of a CLIP ViT-L/14 by "
        "singular value decomposition, freezes the dominant part and trains a small residual with 0.19 million parameters, and reached a mean video-level AUC of 91.7 percent over seven "
        "data sets when trained on FaceForensics++ c23 <c effort>. The paper reports no seeds or intervals, and its ablations were run on the same test sets as its headline, so the "
        "number is context and not a target. Kornblith et al. compared 16 networks on 12 image data sets as fixed feature extractors and as fine-tuned models and found that ImageNet "
        "accuracy correlates with transfer accuracy for both, although some regularisers that raise ImageNet accuracy hurt the transfer of frozen features <c kornblith>. The study "
        "concerns natural images and not forensic detection, so it motivates the encoder comparison without answering it.")
    B.h2("Detection of Fully Generated Images", trace="RESEARCH R-011, R-045, R-046, R-047")
    B.p("A ResNet-50 trained on ProGAN images only, with blur and JPEG augmentation, detected images from unseen CNN generators with a mean average precision above 90 over eleven generators "
        "<c cnnspot>. The paper also separates two ways of choosing the decision threshold: a threshold tuned on the test distribution (the “oracle”) gave an accuracy of 96.8 on "
        "StyleGAN images against 87.1 for the threshold fixed on training data, which is why the threshold of this project is chosen on validation data only. Ojha et al. replaced the trained "
        "network by a nearest-neighbour or a linear classifier in the frozen feature space of CLIP ViT-L/14 <c ojha>. Trained on ProGAN, the linear probe reached a mean AP of 99.31 on "
        "GAN-family test sets and 95.00 on unseen diffusion and autoregressive models, against 94.19 and 75.51 for the best trained network; their training set was far larger than the "
        "one used here. Cozzolino et al. obtained robust results with a linear support vector machine on CLIP features trained on a few pairs of real and generated images <c cozzolino>.")
    B.p("Shortcuts are a risk in these data sets. In GenImage the real images are JPEG files of varied size while the fakes are PNG files of fixed size, so a detector can learn "
        "compression and size; after removing both biases the cross-generator accuracy of a ResNet-50 rose from 71.68 to 82.74 percent <c fakejpeg>. The generated-image path of this project "
        "therefore checks content duplicates across splits and states the class balance of every test set.")
    B.h2("Speech Anti-spoofing", trace="RESEARCH R-012, R-013, R-028 to R-030, R-033")
    B.p("The ASVspoof 2019 database defines logical-access attacks (speech synthesis and voice conversion) and physical-access attacks (replay), with training, development and "
        "evaluation partitions that are disjoint in speakers and an evaluation set that contains attacks unseen in training <c asvspoof>. AASIST applies graph attention to the raw waveform "
        "and reached 0.83 percent EER on the logical-access evaluation set in its best seed and 1.13 percent as the mean of three seeds; the authors note that the headline is the best of "
        "three seeds and that one baseline varied between 1.19 and 2.06 percent across seeds <c aasist>. A RawNet2 front end was worse than a feature-based baseline when used alone and "
        "better in fusion with it <c rawnet2>.")
    B.p("Pretrained self-supervised encoders changed the picture. Replacing the front end of AASIST by wav2vec 2.0 lowered the EER on ASVspoof 2021 logical access from 11.47 to 6.15 percent "
        "(best seed) with nothing else changed <c w2vspoof>. The ASVspoof 2021 challenge showed the danger of a development partition: on the deepfake task the best EER fell to 0.10 percent "
        "in the progress phase and was 15.6 percent in the evaluation phase <c asv2021>. The In-the-Wild study re-implemented twelve architectures under one protocol, trained them on "
        "ASVspoof 2019 and tested them on 37.9 hours of public recordings of 58 speakers; many models were close to chance, and the standard deviation over three seeds reached 15 EER points "
        "<c inthewild>. The speech encoders in the shortlist of this project, wav2vec 2.0 <c wav2vec2>, WavLM <c wavlm>, HuBERT <c hubert> and the Whisper encoder <c whisper>, are the "
        "pretrained models that this literature suggests, used here frozen.")
    B.h2("Audio-visual Detection", trace="RESEARCH R-014, R-015, R-020, R-024, R-025")
    B.p("FakeAVCeleb contains 500 real and 19,500 fake videos with synthesised, lip-synchronised audio <c fakeav>. Audio-only detectors reached an AUC of about 97.8 percent and video-only "
        "detectors about 93.3 percent on its test set, whereas ensembles of unimodal detectors were at best 82.8 and often near 50, and published multimodal models 53.3 to 68.8, so naive "
        "combination was not reliable. Chugh et al. computed a modality dissonance score between the audio and the video stream and reached a video-level AUC of 91.54 percent on a DFDC "
        "subset <c dissonance>. Zhou and Lim compared independently trained streams combined by an OR rule with joint training; on FaceForensics++ and DFDC the independent streams "
        "reached a whole-sequence accuracy of 95.83 and 89.36 percent and joint late fusion 94.25 and 89.15 percent, so joint fusion alone was no better <c zhou>.")
    B.p("LAV-DF contains 136,304 videos of 153 identities, in which words of the transcript are replaced and only the changed segments are regenerated in audio and in video; its four label "
        "types (real, video modified, audio modified, both) and its identity-independent split make it suitable for per-modality questions <c lavdf>. AV-Deepfake1M is larger, with "
        "1,146,760 videos, and shows that in-dataset numbers of different data sets cannot be compared: the same Xception had an AUC of 83.58 on LAV-DF and 68.68 on AV-Deepfake1M "
        "<c avdf1m>. In the audio-visual detectors read for this survey, none handled a missing modality inside the model, although a still image or a silent video needs it.")
    B.h2("Physiological Signals", trace="RESEARCH R-005 to R-010, R-054, R-057, R-064, R-065")
    B.p("FakeCatcher extracts photoplethysmography-like signals from facial regions and classifies them with a support vector machine or a CNN, with in-dataset video accuracy of 96, 94.65, "
        "91.50 and 91.07 percent on four sets <c fakecatcher>; a follow-up addressed source attribution with biological signals <c hearts>. DeepFakesON-Phys fine-tuned a heart-rate "
        "estimation network as a detector and reached an in-dataset AUC of 99.9 percent on Celeb-DF v2 and 98.2 on DFDC Preview, with identity-disjoint development and test sets <c onphys>. "
        "DeepRhythm reached an accuracy of 98.0 percent on FaceForensics++ but 64.1 percent when tested on DFDC Preview, and its ablation shows that without motion magnification the "
        "plain map scores near 50 percent, so the discriminative signal may be an amplified artefact and not a pulse <c deeprhythm>.")
    B.p("Newer evidence is mixed. A study with ECG and photoplethysmography ground truth found that high-quality face swaps do carry a pulse: the heart-rate error was 1.80 beats per minute "
        "for genuine videos, 1.85 for autoencoder fakes and 3.24 for real-time DeepFaceLive fakes, and the pulse of a fake followed the driving video <c heart>. A reproduction of "
        "DeepFakesON-Phys on talking-face fakes of Celeb-DF++ scored an AUC of 62.2 percent against 99.9 in its original setting <c harraq>; the authors of that preprint later describe their "
        "rPPG-derived signal as not shown to be cardiac and note that real and fake videos differ in frame rate <c harraq2>. On fakes built from data sets with contact-sensor ground truth, "
        "the heart-rate error grew from 3.13 beats per minute for real videos to 5.70 for face swap and 12.51 for motion transfer <c cohface>. Pulse signals can also be forged on purpose "
        "<c lin>. The pulse extraction itself, the POS algorithm, was developed on uncompressed laboratory video, and its authors judged one compressed data set unsuitable because compression may remove or pollute the pulse <c pos>. These "
        "results motivate treating the pulse as an experiment with written thresholds and not as an adopted component.")
    B.h2("Evaluation Methodology, Calibration and Adaptation", trace="RESEARCH R-016, R-017, R-036, R-035")
    B.p("Guo et al. showed that modern networks are poorly calibrated and that temperature scaling, a single parameter fitted on a validation set, usually gives the best ECE among the "
        "calibration methods they compared; they define ECE with 15 equal-width bins <c calib>. The studies above show repeatedly that the threshold, the aggregation of frame scores "
        "and the number of seeds change the headline number, which is why the project fixes them in a protocol before any model exists.")
    B.p("Low-rank adaptation freezes the pretrained weights and trains a low-rank update, which reduced the trainable parameters of GPT-3 175B by about 10,000 times and the memory "
        "for training from 1.2 TB to 350 GB <c lora>; the study is on language models. In the Effort protocol (trained on ProGAN, mean accuracy at a fixed threshold), a linear probe scored "
        "81.02, full fine-tuning 86.22 and low-rank variants 91.06 to 93.53 <c effort>, although the rank was chosen on the test sets used for the headline. The encoders of the shortlist of "
        "this project are EfficientNet <c effnet>, Xception <c xception>, DINOv2 <c dinov2>, CLIP <c clip> and SigLIP <c siglip> for images, and the four speech encoders named above; "
        "faces are found with the YuNet detector <c yunet>.")
    B.h2("Research Gap and Objective of the Work")
    B.p("The survey leads to four observations. The numbers of published detectors come from protocols that differ in training data, splits, thresholds and aggregation, so they cannot "
        "stand in for a comparison. In the eight detector papers searched, uncertainty over groups and over seeds was not mentioned, while the measured seed spread of audio detectors is large. "
        "The identity of the person is a shortcut that a split must remove. The pulse signal has contradictory evidence on current generators. No complete path from a minimal working "
        "version to a documented improvement, with the hypothesis written before each run and every comparison paired, was found for the combination of images, video and audio.")
    B.p("The objective of the project work is therefore to build such a system and path: a multimodal detector for images, video and audio, developed from a working version through seven "
        "hypothesis-driven iterations, evaluated on group-disjoint splits with intervals over groups and seeds, and compared with a published detector family re-run under the same "
        "protocol, with “comparable” defined as non-inferior.")
    B.h2("Summary and Outline of the Next Chapter")
    B.p("This chapter reviewed the data sets, detectors and evidence in the literature, quoted their results only with their own protocols, and stated the research gap and the objective. "
        "Chapter 3 now gives the mathematical background of the metrics, the uncertainty estimates and the levers of the iteration ladder.")
