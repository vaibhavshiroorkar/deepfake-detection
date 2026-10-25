"""Chapter 4: implementation and experimentation. Only logged measurements appear as numbers; results that are not final are [pending]."""
import os
import ch4_results as R
HERE_L = __import__("pathlib").Path(os.path.dirname(os.path.abspath(__file__))) / "_labelled"
import ch4_iter as IT



def build(B):
    B.h1("Implementation and Experimentation")
    B.box("This chapter presents the environment, the data, the shared preprocessing, the evaluation harness, the encoder comparison, the working version, the measured iteration "
          "ladder, the web application and the test protocol. Development-slice results are exploratory; the sealed-test results are not yet run and are marked as pending.")
    environment(B)
    datasets(B)
    preprocessing(B)
    harness(B)
    sweep(B)
    working(B)
    ladder(B)
    IT.results(B)
    webapp(B)
    staircase(B)
    B.h2("Summary and Outline of the Next Chapter")
    B.p("This chapter described what was built, how it is tested and what the iterations found. The data, the group-disjoint splits, the shared preprocessing, the evaluation harness with its acceptance test, the "
        "design of the encoder comparison, the working version, the iteration ladder, the web application and the staircase test protocol exist and are documented. The encoder comparison, the seven "
        "iterations, the two further face-video iterations 1b and 1c, the pulse positive control and the re-run baseline have exploratory results on the development slice and on validation data, with their intervals and mechanical verdicts; the sealed-test results and the three "
        "confirmatory tests are marked as pending, because the sealed run follows the readiness checks and no sealed test set has been read. Chapter 5 states the conclusions that the present evidence "
        "supports and the further work.")


# ----------------------------------------------------------------------------------------------------------------------


def environment(B):
    B.h2("Environment", trace="EXPERIMENTS environment record (measured 2026-10-24); DECISIONS D-007, D-017")
    B.p("All experiments run on one desktop computer. The hardware and the software were measured on the machine itself and not copied from its description; <ref tab:env> lists them. "
        "The central processor is an AMD Ryzen 9 7900X, which is the measured model. About 1 GiB of the video memory is in use by desktop applications at idle, so about 11 GiB is usable.")
    B.table("env", "Environment of all experiments, as measured.",
            ["Item", "Measured value"],
            [["Operating system", "Windows 10 Pro, version 10.0.19045, 64 bit"],
             ["Processor", "AMD Ryzen 9 7900X, 12 cores, 24 threads, maximum clock 4701 MHz, 64 MiB L3 cache"],
             ["Memory", "2 x 16 GiB DDR5 configured at 5200 MT/s, 31.1 GiB usable"],
             ["Graphics processor", "NVIDIA GeForce RTX 5070, 12,227 MiB reported by the driver (11.94 GiB seen by PyTorch), compute capability 12.0 (sm_120)"],
             ["Driver", "NVIDIA 595.79 (WDDM), CUDA version reported by the driver 13.2"],
             ["Python", "CPython 3.13.15 managed by uv 0.12.12, virtual environment inside the repository"],
             ["Deep learning", "PyTorch 2.14.1 with CUDA 13.0 wheels (cuDNN 9.24.0), timm 1.0.30, transformers 5.20.2, safetensors 0.8.0, huggingface-hub 2.3.0"],
             ["Numerics and tools", "NumPy 2.5.4, SciPy 1.18.1, scikit-learn 1.9.1 (tests only), OpenCV 5.0.0.93 (headless), soundfile 0.14.0, Pillow 13.0.0"],
             ["Web", "Flask 3.1.3, Werkzeug 3.1.9"],
             ["Media and version control", "ffmpeg and ffprobe 9.0.1 (gyan.dev full build), git 2.51.1"],
             ["Disk", "Repository on drive D: (531 GB, 404 GB free when measured)"]],
            [4.0, 12.4], ["left", "left"], size=10)
    B.p("Python 3.13 was chosen over the newer system Python 3.14 because the machine-learning libraries support it better. The wheel index of PyTorch was queried directly: it offers "
        "builds of 2.14.1 for CUDA 12.6, 13.0 and 13.2, whereas the index for CUDA 12.8 stops at 2.11.0. The build for CUDA 13.0 was installed because the driver reports CUDA 13.2, which is "
        "newer, and because the build ships native kernels for sm_120.")
    B.p("The graphics processor was checked before any model work: PyTorch reports it, its list of architectures includes sm_120, a 4096 by 4096 float32 matrix product ran and differed from "
        "a float64 product on the processor by at most 6.0e-4 on the first eight rows, which is ordinary float32 rounding for sums of 4096 products, and a short training loop under "
        "bfloat16 and float16 autocast ran with finite loss. To size later plans, a deliberately trivial network was timed: three 3 by 3 convolution layers with 32, 64 and 128 channels, "
        "max pooling, global average pooling and a linear layer (93,506 parameters), on random tensors of 3 by 224 by 224 in batches of 256 with AdamW and the data already on the "
        "graphics processor. <ref tab:probe> gives the result. It is a throughput and memory probe and not a model proposal: it says nothing about the speed of a real backbone, "
        "and real data loading is the likely bottleneck.")
    B.table("probe", "Sizing probe: one training step of a trivial network, batch of 256 images of 3 by 224 by 224.",
            ["Precision", "Time per step", "Images per second", "Peak allocated memory"],
            [["float32", "134.5 ms", "1903", "5771 MiB"],
             ["bfloat16 autocast", "77.6 ms", "3300", "3787 MiB"],
             ["float16 autocast with gradient scaler", "83.4 ms", "3068", "3787 MiB"]],
            [6.0, 3.2, 3.6, 3.6], ["left", "center", "center", "center"], size=10)
    B.p("Mixed precision cut the peak memory by about a third and the step time by about 42 percent for this network, so it is the default assumption when sizing. The memory here is "
        "dominated by activations at 224 by 224 resolution and not by parameters. Finally, the download speed was measured once: 4.09 MB/s on a single connection from one server, "
        "which is about 27 hours for 404 GB. This figure, from one server and one moment, is why the data plan puts the datasets that the working version needs first, and why the "
        "approval-gated data sets were fetched on a faster connection and transferred on a drive.")


def datasets(B):
    B.h2("Datasets, Splits and Licences", trace="DECISIONS D-012, D-013, D-020 to D-022, D-026, D-035, D-039, D-045; DATA_LOG")
    B.p("Every data set serves a numbered requirement: training data for the working version and the comparison of candidates, cross-dataset test sets for the headline "
        "generalisation result, audio-visual data with labels for each modality, audio-only data, image data of two kinds (face-manipulated and fully generated) and data adequate for "
        "the pulse experiment. A data set is kept only if it serves one of them; easy availability is not a reason. <ref tab:data> lists the data sets in use.")
    B.table("data", "Data sets in use, their roles and their licences or terms of use. All are for research use only.",
            ["Data set", "Role", "Content as obtained", "Licence or terms"],
            [["FaceForensics++ c23", "Training and validation of the video task; in-dataset test", "5,000 videos: 1,000 originals and 1,000 for each of four methods; no audio track", "FaceForensics terms of use: non-commercial research and education"],
             ["Celeb-DF v2", "Development slice and sealed cross-dataset test", "590 real, 300 YouTube real, 5,639 fake videos; 59 identities; no audio track", "Request terms: non-commercial; no reproduction or publication of videos or of derived data"],
             ["Celeb-DF++", "Sealed test with 21 newer generators", "Test list: 178 real and 4,900 fake videos, 5,078 extracted", "Own terms of use, provided as it is; derived data not published"],
             ["DeepFakeDetection (DFD)", "Sealed cross-dataset test", "363 original and 3,068 manipulated videos, 28 actors", "Same terms as FaceForensics++"],
             ["ASVspoof 2019 LA", "Training, validation and in-dataset test of the audio task", "25,380 / 24,844 / 71,237 utterances; 20 / 20 / 67 speakers", "Open Data Commons Attribution License"],
             ["In-the-Wild", "Development slice and sealed test of the audio task", "54 speakers in the metadata, 31,779 clips", "CC BY-SA 4.0 on the dataset card"],
             ["ProGAN validation and test sets", "Training and validation (14 and 6 classes); in-dataset test", "8,000 images each, 20 LSUN classes, 200 real and 200 fake per class", "CC BY-NC-SA 4.0 on the dataset card"],
             ["CNN synthesis subset", "Development slice and sealed test of the image task", "56,003 images from 8 generator folders", "CC BY-NC-SA 4.0 on the dataset card"],
             ["Diffusion test set", "Sealed test of the image task", "10,000 images from unseen diffusion generators", "Image licence not established; code under MIT"],
             ["LAV-DF", "Audio-visual iteration", "136,304 videos of 153 identities, all with an audio stream; splits of 78,703 / 31,501 / 26,100 videos", "CC BY-NC 4.0; non-commercial research; VoxCeleb2 agreements apply"]],
            [3.2, 3.9, 5.2, 4.1], ["left", "left", "left", "left"], size=9, keep=False)
    B.p("All of these data sets are for research use only. Celeb-DF, Celeb-DF++ and FakeAVCeleb forbid the publication of derived data, so no frame, crop or per-video score of them "
        "appears in any document of this project. Downloads were checked by file counts against the documentation and by archive CRC or SHA-256 checks, and the FaceForensics++ and LAV-DF videos were also "
        "checked by decoding every file with ffprobe. The LAV-DF archive of 25,607,198,720 bytes matched the published hash and all 136,304 videos have an audio stream; "
        "neither the FaceForensics++ videos (0 of 20 sampled) nor the Celeb-DF v2 videos (0 of 250 sampled) have one, so the audio inside videos is trained and tested on LAV-DF only. "
        "DFDC, FaceForensics++ at c40, DeepfakeTIMIT and UBFC-rPPG were deferred, and the other data sets considered were dropped because no requirement needs them.")
    B.h3("Group-disjoint splits")
    B.p("A group is the source video together with the person identity where metadata exists; everything that shares a source or an identity shares a group. When identity metadata "
        "is missing the group falls back to the source video, and the manifest records that the identity is not known. Every sample belongs to exactly one of five splits, which "
        "<ref fig:splits> shows: <i>train</i>; <i>val</i>, held out from the training data set and used for the threshold, the temperature and early stopping; <i>dev_x</i>, a development slice "
        "carved out of a cross-dataset source and used to rank candidates and choose hyperparameters; <i>test_in</i>, the in-dataset test; and <i>test_x</i>, a data set never used in "
        "training. The development slice is not a final split and may be evaluated repeatedly; the two test splits are sealed. A data set family, such as Celeb-DF v2 together with "
        "Celeb-DF++, counts as one source, so a family used in training is never a cross-dataset test.")
    B.figure("splits", "fig02_splits.png", "Group-disjoint splits and the access rules. The development slice and the sealed test set of a cross-dataset source share no group, identity or file content.", 15.5,
             "Diagram of the training data set with train, validation and in-dataset test, and of a cross-dataset source with development slice and sealed test set", tag="schematic")
    B.p("The development slice is chosen by a seeded rule fixed before any extraction: the group names are sorted, a permutation is drawn with a fixed random seed and the first $k$ groups "
        "go to the slice, with every sample of a drawn group. <ref tab:slices> gives the slices that resulted. The slices are small, which makes intervals wide and ties likely; the "
        "cheaper-candidate rule exists for this reason.")
    B.table("slices", "Development slices and sealed test sets of the cross-dataset sources (real / fake counts where both exist).",
            ["Source", "Development slice", "Sealed test set"],
            [["Celeb-DF v2", "15 of 59 identity groups: 1,728 videos (146 real, 1,582 fake)", "4,801 videos (744 real including 300 YouTube videos, 4,057 fake); 44 identity groups plus the YouTube videos"],
             ["Celeb-DF++", "none; the 15 development identities are excluded", "2,849 fake videos, no real video; 44 identity groups"],
             ["Family total (Celeb-DF v2 and ++)", "1,728 videos", "744 real and 6,906 fake videos; about 44 independent identity groups"],
             ["In-the-Wild", "12 of 54 speakers", "the other 42 speakers"],
             ["CNN synthesis subset", "2 of 7 generator units: StarGAN (1,999 real, 1,999 fake) and Which-Face-Is-Real (1,000 real, 1,000 fake)", "48,805 images of the other 5 units after removing 1,200 duplicate real images"],
             ["DFD", "none", "28 actors, 3,431 videos"]],
            [3.6, 6.0, 6.8], ["left", "left", "left"], size=9)
    B.h3("Celeb-DF++ and the balance of its sealed test set")
    B.p("Celeb-DF++ is built on Celeb-DF, shares its 59 identities and counts as one family with it. Its test list holds 4,900 fakes of the 21 newer generators and 178 real "
        "videos; 5,078 videos were extracted. Two exclusions keep any source video or identity of the development slice from being counted twice. First, 2,079 samples were excluded from the "
        "sealed test set because their target identity (1,477 samples) or only their celebrity source identity (602 samples) is one of the 15 development identities. Second, 150 real "
        "videos of Celeb-DF++ whose content is byte-identical to Celeb-DF v2 videos were dropped, and Celeb-DF v2 keeps them under its own group identifier, so that each source video has "
        "one group. The exclusion by celebrity source identity, and not by target identity alone, is required and not optional: in a face swap named by target and source identity the face "
        "that is visible is the source identity, so a rule on the target alone would leave faces of the development identities in the sealed test set.")
    B.p("The result is that Celeb-DF++ alone has 2,849 fake videos and no real video in its sealed test set, in 44 identity groups. Its AUC is therefore computed together with the "
        "Celeb-DF v2 real videos, which are the same real sources, and the pooled family test set holds 744 real and 6,906 fake videos. Real videos are 9.7 percent of that pool. The "
        "consequence for the statistics is that the bootstrap resamples identities, so the fakes of Celeb-DF++ add generator coverage but almost no new independent groups: intervals on a "
        "pooled or per-generator AUC are governed by about 44 identity groups, and the estimates per generator, from 200 to 300 clips each, are wide. Balanced accuracy and the "
        "threshold carry more noise than the AUC on such an unbalanced set.")
    B.h3("Leakage check")
    B.p("A script proves that the splits are disjoint in group, identity and, for the image and the audio-visual sets, file content. With content hashing the check first failed on "
        "the image sets: real LSUN images occur byte for byte in the training set, the validation set and several test sets. The rule adopted is that any sample of a test set whose "
        "content hash equals the hash of a training or validation sample is excluded from the test set, and the exclusions are listed with the reason. After 1,200 samples of the "
        "CNN synthesis subset were excluded, the check passes on 80,803 image files, and it passes on 41,739 files of Celeb-DF, In-the-Wild and DFD together. The Celeb-DF family has "
        "one further pass after the exclusions above (9,378 files, no duplicate content in or across splits). The grouping of Celeb-DF fakes takes the first identity in the file name, "
        "assumed to be the target, as the resampling unit; the other identity is kept in the notes. The ProGAN validation set is split by LSUN class, 14 classes for training and 6 for "
        "validation, because a class is the coarsest source grouping available for images with no identity; the threshold and the temperature are therefore fitted on classes not seen in "
        "training.")


def preprocessing(B):
    B.h2("Shared Preprocessing", trace="DECISIONS D-028, D-033, D-034, D-044; EXPERIMENTS M-001, M-002")
    B.p("One package prepares inputs for training, evaluation and inference, so that the same function runs in all three. The preprocessing parameters form a configuration whose hash is "
        "stored in every checkpoint, and inference refuses a checkpoint whose hash differs from the one in use. This turns a silent mismatch between training and use into an error.")
    B.table("prep", "Preprocessing of each input type.",
            ["Input", "Steps", "Parameters"],
            [["Video", "Decode sequentially; take uniformly spaced frames; detect the faces with YuNet; crop the largest face enlarged by a factor; store as JPEG", "16 frames; factor 1.3; crop 224 by 224; JPEG quality 95; frames without a face left out and counted"],
             ["Image", "Face crop if a face is found, otherwise the whole image processed by the encoder’s own model-card rule", "the path taken is reported in the response"],
             ["Audio", "First samples at 16 kHz, repeat-padded if shorter; the Whisper encoder receives its fixed 30 second window with the 4 seconds first and only those frames are pooled", "64,600 samples (about 4 seconds)"]],
            [2.0, 8.4, 6.0], ["left", "left", "left"], size=10)
    B.p("When several faces are found the largest is used and the warning that several faces were found is returned. A clip with no face at all returns the warning that no face was "
        "detected. Carrying the box of a neighbouring frame or using a centre crop for a frame without a face was rejected, because it invents a face position. Sequential decoding is used because seeking "
        "is about four times slower on FaceForensics++.")
    B.p("The face detector was chosen by measurement. YuNet <c yunet>, UltraFace and SCRFD were compared on 7,488 sampled FaceForensics++ frames and, separately, on 300 frames from 300 "
        "distinct FaceForensics++ validation videos and 300 frames from 300 distinct Celeb-DF development-slice videos (frames only, no labels or scores). All three found a face in every "
        "frame and never disagreed on whether a face exists; the largest boxes agree with a mean intersection over union of 0.887 to 0.923, and the eight frames of lowest agreement per set "
        "show the same face in every tile. The differences that matter are speed and licence (<ref tab:det>).")
    B.table("det", "Face detector comparison (measurements; the machine was busy with 13 to 20 percent of the processor before each run, and the 200-video sample is warm in the file cache).",
            ["Detector", "Licence of the weights", "ms per frame (FaceForensics++ / Celeb-DF)", "Detections on 500 face-free images"],
            [["YuNet 2023mar", "MIT", "12.1 / 12.4", "6.0 percent"],
             ["UltraFace RFB-320", "not verified", "4.8 / 4.8", "20.2 percent"],
             ["SCRFD 500M", "non-commercial research only", "29.7 / 30.9", "4.6 percent"]],
            [3.6, 4.4, 4.8, 3.6], ["left", "left", "center", "center"], size=10)
    B.p("YuNet was chosen: it is small, MIT licensed, fast enough and runs on the processor through OpenCV; the other two were used only as benchmarks. The measurement does not "
        "include hard cases such as rotated or tiny faces, and the effect of a few percent of box shift on accuracy was not measured.")
    B.p("The throughput of the preparation was measured on 200 FaceForensics++ videos (16 frames, 224 pixels, JPEG 95, YuNet). Decoding takes about 73 percent of the time per video and "
        "detection about 24 percent. Decoding 16 frames took 0.74 s per video with OpenCV sequentially and with an ffmpeg pipe, and 2.88 s with seeking. The pipeline reached 1.25, 4.13, 5.91, 6.35 "
        "and 6.22 videos per second with 1, 4, 8, 16 and 24 worker processes; the growth stops after 8 workers and the cause was not found. This projects to about 13 minutes for the "
        "5,000 FaceForensics++ videos at 16 workers, with 222,786 bytes of crops per video. ASVspoof FLAC files load in 4.76 ms each, about 582 s single-threaded for all 122,299 files. "
        "These timings are caveated by the busy machine and by warm reads; cold-disk reads were not measured.")


def harness(B):
    B.h2("Evaluation Harness and Its Acceptance Test", trace="PROTOCOL; EXPERIMENTS V-001, V-002; DECISIONS D-024")
    B.p("The evaluation protocol was fixed, and its harness built and tested, before any model existed. The harness implements the metrics of Chapter 3, the threshold chosen on "
        "validation data, temperature scaling, the group bootstrap with the paired and multi-seed forms, the guard for the sealed splits and the ledger. It reads score files only and never "
        "imports a model. A score file holds one row per clip with the run, iteration, seed, modality, data set, split, sample identifier, group identifier, label and the probability of fake.")
    B.p("Three rules define how a clip and a threshold are obtained. The clip score is the mean of the frame probabilities. The threshold maximises balanced accuracy on the validation "
        "split of the training data set, taking the midpoints between adjacent distinct validation scores and, among equal maxima, the one closest to 0.5; the chosen threshold is then fixed "
        "for every test set. Temperature scaling is fitted on the same split. Uncertainty uses 2,000 resamples, a 95 percent percentile interval and a fixed base seed, with the group as "
        "the resampling unit.")
    B.p("The guard refuses any evaluation that reads a sealed split unless the run is flagged final and carries an iteration tag. The ledger, a file under version control, records every "
        "final evaluation: the date, the iteration, the data set, the split, the run identifiers and the hash of the score file; a second final evaluation of the same iteration and data "
        "set is refused unless a reason is given, and the repeat is then listed as a correction. A paired comparison on a sealed split re-analyses logged scores only, needs the same "
        "flag, and refuses unless a regular final evaluation of that iteration and data set is already in the ledger, so it never gives a first look at a sealed split.")
    B.p("<ref tab:accept> lists the acceptance checks and what was logged for them.")
    B.table("accept", "Acceptance checks of the evaluation harness and the data tools.",
            ["Check", "What is tested", "Logged result"],
            [["Metrics", "AUC, log loss and Brier score against scikit-learn on at least 20 random score sets; balanced accuracy, EER, precision, recall and ECE against hand-computed values", "passed; on 5 further random sets of 2,000 samples with many tied scores the AUC equals scikit-learn’s to 10 decimals and log loss and Brier differ by at most 1.1e-16"],
             ["Temperature scaling", "recovers a known temperature on synthetic logits within 2 percent", "passed"],
             ["Bootstrap coverage", "the group bootstrap interval covers the true AUC in at least 90 percent of 200 simulated data sets", "192 of 200 (96 percent)"],
             ["Paired bootstrap", "two identical runs give a difference of exactly zero with the interval [0, 0]", "passed"],
             ["Guard and ledger", "refuses a sealed split without the final flag, refuses a second final evaluation without a reason, accepts a first one", "passed"],
             ["Leakage check", "exits non-zero on a manufactured overlap and zero on clean splits, including across source families, with content hashing", "passed (Section 4.2)"],
             ["Manifests", "documented counts for FaceForensics++ c23, ASVspoof 2019 LA, ProGAN validation and test, Celeb-DF v2", "121,461 ASVspoof rows; 8,000 images per ProGAN set; 5,000 FaceForensics++ videos; 6,529 Celeb-DF v2 videos in 59 identity groups"],
             ["Development slice", "accepted by the score reader and not treated as final; leakage fails if a group or content is in both the slice and the sealed test", "passed"],
             ["End-to-end random scorer", "AUC about 0.5 with an interval containing 0.5", "every interval contained 0.5 and every AUC was within 0.06 of 0.5; in a one-off run on 30 other synthetic files 117 of 120 intervals contained 0.5"]],
            [3.0, 7.0, 6.4], ["left", "left", "left"], size=9)
    B.p("A full pass of the command line on 10,000 validation and 50,000 test rows takes 9 s, and 2,000 resamples on 50,000 samples in 5,000 groups take 1.0 to 2.1 s. After the development slices "
        "were added, the test suites of the data tools and of the evaluation harness had 67 and 89 tests, all of which pass.")


def sweep(B):
    B.h2("Design of the Encoder Comparison", trace="DECISIONS D-032, D-036, D-038, D-042; O007_SWEEP; EXPERIMENTS E-001, E-002, E-003")
    B.p("The first iteration is a measured comparison of pretrained encoders under one protocol. The form of the working version is fixed: a frozen pretrained encoder and a linear head. "
        "Which encoder is the open question, and it is asked separately for three tasks: face video frames (the visual branch of a video), generated images and speech audio. "
        "<ref tab:tasks> gives the data of each task.")
    B.table("tasks", "The three tasks of the comparison.",
            ["Task", "Train", "Validation", "Development slice (ranking)", "Input to the encoder"],
            [["Face video frames", "FaceForensics++ c23, official training split", "FaceForensics++ official validation split", "Celeb-DF v2, 15 identity groups", "16 face crops of 224 by 224 per video"],
             ["Generated images", "ProGAN validation set, 14 classes", "the other 6 classes", "2 of 7 CNN-synthesis generator units", "whole image, processed by the encoder’s own rule"],
             ["Speech audio", "ASVspoof 2019 LA train", "ASVspoof 2019 LA dev", "In-the-Wild, 12 speakers", "first 4 seconds at 16 kHz"]],
            [2.5, 3.4, 3.2, 3.6, 3.7], ["left", "left", "left", "left", "left"], size=9)
    B.p("The shortlist was chosen by criteria stated first: the licence is verified from the weights’ own repository, the weights download and load offline, full inference fits in "
        "12 GB, the feature tensor is stated exactly, and the families differ. DINOv3 was excluded on purpose, because its access gate is manual and its licence carries a duty to acknowledge it "
        "in publications. <ref fig:shortlist> and <ref tab:weights> give the shortlist.")
    B.figure("shortlist", "fig08_shortlist.png", "Shortlist of the encoder comparison, with parameters and feature sizes read from the extraction records.", 15.5,
             "Table of the shortlisted image and speech encoders with family, parameters, feature size and weights licence")
    B.table("weights", "Weights of the shortlisted encoders and their licence labels.",
            ["Candidate", "Checkpoint", "Source", "Label"],
            [["EfficientNet-B4 <c effnet>", "timm efficientnet_b4.ra2_in1k", "Hugging Face", "Apache-2.0 tag; ImageNet-trained, research use"],
             ["Xception <c xception>", "timm legacy_xception (original architecture)", "GitHub release of timm", "Apache-2.0 declared by timm; upstream licence not retrieved; research use"],
             ["DINOv2 small, base <c dinov2>", "facebook/dinov2-small, facebook/dinov2-base", "Hugging Face", "Apache-2.0"],
             ["CLIP ViT-L/14, OpenAI <c clip>", "openai/clip-vit-large-patch14", "Hugging Face", "no weights licence stated; card says research use; accepted as research-only"],
             ["CLIP ViT-L/14, LAION", "laion/CLIP-ViT-L-14-laion2B-s32B-b82K", "Hugging Face", "MIT tag; card says research use"],
             ["SigLIP base <c siglip>", "google/siglip-base-patch16-224", "Hugging Face", "Apache-2.0"],
             ["wav2vec 2.0 base <c wav2vec2>", "facebook/wav2vec2-base", "Hugging Face", "Apache-2.0"],
             ["WavLM base plus <c wavlm>", "microsoft/wavlm-base-plus", "Hugging Face", "no tag; upstream licence CC BY-SA 3.0; accepted as research-only"],
             ["Whisper small encoder <c whisper>", "openai/whisper-small", "Hugging Face", "Apache-2.0 tag; MIT upstream"],
             ["HuBERT base <c hubert>", "facebook/hubert-base-ls960", "Hugging Face", "Apache-2.0"],
             ["Small networks from scratch", "own code", "none", "reference only, excluded from the selection"]],
            [4.2, 5.0, 2.6, 4.6], ["left", "left", "left", "left"], size=9)
    B.p("The licence of two weights is conditional. The OpenAI CLIP repository states no licence and its card says use is for research; the WavLM card has no licence tag and "
        "points to an upstream licence that is CC BY-SA 3.0. Both are accepted as research-only, and this is a limitation: results obtained with these weights make no claim of commercial "
        "usability. The ImageNet-trained weights are treated as research use as well, because the data set terms say so. The SHA-256 of every weight file was recorded before use, and the script that "
        "fetches the weights recomputes it locally and compares it.")
    B.p("Features are extracted once per candidate and task. The encoder runs in evaluation mode with mixed precision and the preprocessing of its own model card. The stored feature is the pooled "
        "embedding before any classifier: the global average of the final map for the convolutional networks, the class token for DINOv2, the projected image embedding for the CLIP encoders, "
        "the attention-pooled output for SigLIP, and the mean and standard deviation over time of the last hidden layer for the speech encoders. Features are stored as 16-bit floats with an "
        "index to the sample identifiers, and the same crops and split files serve every candidate.")
    B.p("The head is a single linear layer on features standardised with the training mean and standard deviation, trained with class-weighted binary cross-entropy, AdamW with a "
        "learning rate of $10^{-3}$ and a weight decay of $10^{-4}$, batches of 1,024 features, at most 200 epochs and early stopping on the validation AUC with a patience of 10. Three "
        "seeds, 0, 1 and 2, change the initialisation and the batch order. Nothing else is tuned in this iteration.")
    B.p("The ranking metric is the clip-level AUC on the development slice, averaged over the seeds, with the 2,000-resample paired group bootstrap. The leader is the candidate with the highest "
        "mean; the tie set is every candidate whose paired interval against the leader includes zero (Equation <ref eq:tieset>); the cheapest member of the tie set wins. The from-scratch "
        "reference is trained on the same split for a fixed budget of 10 epochs with AdamW at $10^{-3}$ and no augmentation, is reported in the same table and is excluded from the "
        "selection. If the lower bound of the best candidate on the development slice is 0.55 or below, the sweep is declared inconclusive for that task. The exact sign-flip permutation "
        "test over groups is a sensitivity column beside the bootstrap, not a second rule. A cost record is kept for every candidate: parameters, floating-point operations of one forward "
        "pass, peak memory, and time per 1,000 inputs both for the model alone and end to end including decoding.")
    B.p("The hypotheses, success criteria and failure criteria were written before any feature extraction and are not edited afterwards. For face video frames, the hypothesis is that "
        "at least one self-supervised or vision-language encoder (the DINOv2 or CLIP families) reaches a higher development-slice AUC than the best supervised ImageNet network "
        "(EfficientNet or Xception); it is supported if the paired interval of that difference lies above zero and the lower bound of the best candidate is above 0.60, and it fails if "
        "the interval includes zero or if the lower bound is 0.55 or below. For generated images the hypothesis is that a CLIP-style frozen encoder beats every supervised network, with "
        "a lower bound above 0.65 as the success line. For speech the hypothesis is that wav2vec 2.0 or WavLM beats both the Whisper encoder and the from-scratch reference, "
        "with a lower bound above 0.60 as the success line. These are hypotheses and they have a direction that is uncertain by design: the literature also shows that face-forgery "
        "detectors learn identity cues <c identity> and that frozen features have been tested for generated images <c ojha> and not for face forgery.")
    R.sweep_results(B)


def working(B):
    B.h2("The Working Version (Iteration 0)", trace="DECISIONS D-032, D-040, D-044; ITERATIONS I-0")
    B.p("The working version is not an improvement; it is the baseline that everything else is compared with. It counts as working when all of the following hold: a file goes in "
        "through the local web page; the encoder of each task is the winner of the comparison under the cheapest rule, frozen, with a linear head trained on the training split; the score is "
        "calibrated by temperature scaling fitted on validation and the checkpoint metadata is checked at load; metrics with three seeds, standard deviation and bootstrap interval exist on "
        "the validation split and on the development slice without any sealed split touched; the page shows a real result with its source marked real and the warnings for no face and "
        "several faces; the visual and the audio score of a video are shown separately; and the tests pass. <ref fig:pipeline> shows the chain from a file to a calibrated score.")
    B.figure("pipeline", "fig01_pipeline.png", "From a file to a calibrated score. The encoders are frozen, the temperature and the threshold are fixed on validation data, and the scores are probabilities of fake.", 15.5,
             "Three rows of processing blocks for the visual branch of a video, the audio branch and an image, each ending in a score", tag="schematic")
    B.p("A checkpoint is a folder with one metadata file and one head file per task; no pickled object is used, because a pickle executes code when it is loaded. The metadata holds the "
        "checkpoint identifier, the iteration, the commit, the protocol, the seed, the library versions, the manifest and split identifiers and, per task, the encoder (name, revision, "
        "feature size, input size), the hash of the head file, the preprocessing hash and configuration, the calibration method with its temperature and the split it was fitted on, and the "
        "threshold with how it was chosen. The head file holds the mean and standard deviation of the standardisation and the weight vector and bias of the head. The loader refuses a "
        "folder without metadata, a folder with missing fields, a head file whose hash differs, a head whose width differs from the encoder’s and a preprocessing hash different from "
        "the one in use. The real adapter also checks that the encoder weights on disk are the hash-verified ones of the recorded revision, and a checkpoint without an audio head reports "
        "the audio modality as unavailable with the reason; no score is ever invented.")
    B.p("Scoring follows training exactly: 16 frames, the same detector, the same enlargement, a JPEG quality-95 round trip, the mean of the frame probabilities and temperature scaling. An "
        "image with a face goes through the video head and an image without a face through the generated-image head, with the path reported. Audio is scored from the first 64,600 "
        "samples at 16 kHz with the speech encoder’s own feature extractor, the mean and standard deviation over time, the audio head and temperature scaling. The code path was tested with a "
        "test double for the encoder and the detector and a made-up head; the acceptance run of the working version must score a training-split clip both ways to expose any mismatch "
        "between training-time and inference-time preprocessing.")
    B.p("After every iteration, including this one, the same diagnostic battery is run and its output fixes the order of what follows. It consists of: the AUC per data set and per generator "
        "or method with intervals; the gap between training and validation AUC, which separates bias (both low) from variance (a large gap); the gap between validation and the development "
        "slice, which indicates a distribution shift; the calibration curve with 15 bins, the ECE and the Brier score before and after temperature scaling; the score histograms of real and "
        "fake examples; the results split by compression level, face size and number of faces; the 20 worst errors, 10 false alarms and 10 misses, reviewed by eye and classified in "
        "writing; and the standard deviation over the three seeds next to the bootstrap half-width, which separates training noise from sampling noise. The diagnosis vocabulary is bias, "
        "variance, shift, calibration, aggregation and missing signal.")
    R.working_results(B)


def ladder(B):
    B.h2("The Iteration Ladder", trace="DECISIONS D-004, D-041, D-042; ITERATIONS; MATH_NOTES sections 10 to 17")
    B.p("After the working version the project has seven experimental iterations, I-1 to I-7, and a last step, I-8, that integrates and freezes. I-1 to I-6 take their lever from one "
        "of six families, and the lever of each slot is chosen after the diagnostic battery of the preceding iteration, not in advance; the primary candidate in <ref tab:ladder> is a "
        "starting guess for planning that is replaced without penalty when the battery points elsewhere. I-7 is the pulse experiment and I-8 runs no experiments. <ref fig:ladder> shows the "
        "ladder. The levers that were actually used and their results are given in the section on the results of the iterations. "
        "After the first pass over the ladder, two further iterations on the face-video task were added at the request of the project team, 1b (augmentation matched to the test perturbations, with mixup) and 1c "
        "(combining encoders), so the video task has the iterations 1, 1b and 1c.")
    B.figure("ladder", "fig03_ladder.png", "Iteration ladder. Each step from I-1 to I-6 starts from the diagnostic battery of the preceding step.", 15.5,
             "Row of nine boxes from the working version to integration and freeze, all feeding a diagnostic battery", tag="schematic")
    B.p("Every slot is written in one template: observation (numbers from the battery), diagnosis, mathematical hypothesis, change, prediction with sign and rough size, result and verdict. "
        "The prediction lines are hypotheses, and each is judged against its failure line after the run. Verdicts on the development slice are exploratory.")
    B.table("ladder", "Candidate lever, trigger, prediction and failure line of iterations I-1 to I-6 as planned before the runs. The predictions are hypotheses written before any run; the measured ladder is in the section on results.",
            ["Slot", "Lever", "Triggered when", "Prediction", "Failure line"],
            [["I-1", "Fine-tuning depth: linear probe to the last blocks, low-rank adapters or full fine-tuning", "Bias: training and validation AUC both low or close (gap under 0.02, AUC under about 0.90)", "Validation AUC up by about 0.02 to 0.05; development slice between slightly negative and +0.03", "Paired development interval includes zero or is below zero while validation AUC rises; or memory or time is outside the 12 GB budget"],
             ["I-2", "Frame aggregation: mean to top-k mean, maximum or learned attention pooling", "Frame-level AUC clearly above video-level AUC, a wide spread of frame scores, or many frames without a face", "Video AUC change from 0 to +0.02 when few frames are informative; negative when frame noise dominates", "Paired interval includes zero, or the real-video score histogram moves right (more false alarms)"],
             ["I-3", "Loss and class weighting: class-weighted, focal, label smoothing or pairwise surrogate", "Imbalance or saturation: balanced accuracy far below what the AUC implies, scores piled at 0 or 1, or ECE high after scaling", "AUC change within about 0.01; balanced accuracy and ECE change more", "AUC interval includes zero and the other two improve by less than their intervals"],
             ["I-4", "Augmentation matched to the test perturbations; mixup; weight decay and dropout", "Shift: validation AUC high, development slice or compression splits much lower (gap above about 0.10)", "Development AUC up by about 0.01 to 0.04 for a compression or resolution shift; in-dataset flat or down by up to 0.01", "Development interval includes zero, or gains appear only on the perturbation types used in training"],
             ["I-5", "Combining encoders: concatenate features or average scores", "Ties in the comparison and complementary errors (low error correlation)", "AUC +0.01 to +0.03 if the mean correlation is below 0.8; none as it tends to 1", "Interval includes zero: the cost in time, memory and licences is not justified"],
             ["I-6", "Late fusion of audio and visual scores with per-modality calibration", "Both unimodal heads above chance on LAV-DF and calibrated, with errors not strongly correlated", "Fused AUC at or above the better single modality; gain of 0 to +0.05; learned weights below 1 if dependent", "Fused lower bound below the better single modality, or the gain is within the interval"]],
            [1.1, 3.7, 4.2, 3.8, 3.6], ["left", "left", "left", "left", "left"], size=9, keep=False)
    B.p("Two further elements complete the ladder. For I-6, the unimodal heads are the ones frozen in the earlier iterations, the visual head trained on FaceForensics++ and the audio head on "
        "ASVspoof 2019, each with its own temperature; only the fusion layer is fitted, on the LAV-DF training split, with the choice made on its validation split and per-modality "
        "temperatures also refitted on LAV-DF and reported both ways. A video is labelled fake for the fused question if either modality is fake, and the label of each modality from "
        "the metadata is reported as well, one AUC per modality label.")
    B.h3("Fine-tuning budget of iteration I-1")
    B.p("The budget was derived from the measured time per step and not from a plan. Fine-tuning is done for the video task first. The top three encoders of the comparison that fit are "
        "fine-tuned at three learning rates, $10^{-5}$, $3\\times 10^{-5}$ and $10^{-4}$, with one seed, and the best rate of the best encoder is then repeated with two further seeds. "
        "Every run has the same budget of 1,800 optimiser steps, which is one pass over the 57,600 training frames, an effective batch of 32, AdamW with a cosine schedule and 5 percent "
        "warm-up, weight decay 0.05, mixed precision and no augmentation. A run has a cap of 10 minutes of training and 15 minutes of wall time including evaluation. An encoder enters "
        "only if its peak memory at batch 32 is at most 10 GiB and one budget takes at most 10 minutes. By the step-time measurement this admits DINOv2 small and base, Xception and "
        "EfficientNet-B4; the two CLIP ViT-L/14 models do not fit, because their peak memory at batch 32 exceeds the 12 GB card and spills into shared system memory, which makes a step "
        "take seconds, and they stay frozen (a low-rank adapter variant is a later option). The claim “easiest to fine-tune” is made for the video task and for the encoders that fit only, "
        "and the step times are re-checked against a clean re-run before the first run. The generated-image and the speech tasks keep their frozen heads in I-1.")
    B.h3("The pulse experiment (iteration I-7)")
    B.p("The pulse is a candidate improvement and not an adopted component. Its success and failure thresholds were written before any run and are not edited after the first run. "
        "A signal check, S-1, starts early on the processor, in parallel with I-1 to I-3. It extracts about 10 seconds of contiguous frames at the native frame rate, because the "
        "16-frame sampling cannot carry a pulse, and computes the spectral peak in 0.7 to 3 Hz and its signal-to-noise ratio (Equation <ref eq:snr>) on real clips. The negative control is "
        "the same clips with time-shuffled frames, pooled across clips with at least ten shuffles per clip, and the 95th percentile of the pooled shuffled values is the reference. <ref fig:rppg> shows the chain.")
    B.figure("rppg", "fig05_rppg.png", "Processing chain of the pulse signal check and its negative control.", 15.5,
             "Seven processing blocks from contiguous frames to peak and signal-to-noise ratio, with a time-shuffled control", tag="schematic")
    B.p("S-1 passes when at least 60 percent of the real clips have a peak SNR above the 95th percentile of the shuffled control and, if a data set with pulse ground truth is obtained, "
        "the mean absolute heart-rate error is at most 8 beats per minute. If it does not pass, I-7 reports exactly that, “pulse not recoverable from these data”, with the numbers and "
        "the negative control, and no separation or fusion experiment is run; this is an honest result of the slot. If it passes, three further tests follow: the AUC of rPPG features on an "
        "identity-disjoint split for at least two generator families, with success if the lower bound is above 0.55 and failure if the interval includes 0.5; the added value of the fused "
        "model with the pulse over the model without it, a paired group bootstrap on the development slice with three seeds, adopted only if the whole interval is above zero; and negative "
        "controls with time-shuffled and identity-mismatched signals, whose AUC intervals must include 0.5, otherwise the separation is an artefact. The decision to adopt or not is made on the "
        "development slice and never on a sealed test set. The prior expectation, a hypothesis and not a result, was that most of the videos are compressed, which attenuates the pulse, so that S-1 "
        "would pass only partially. S-1 was run and failed (130 of 432 real clips above the control, 0.301 [0.260, 0.346], against the required 0.6), so none of the three further tests was run; the numbers are in the section on the results of the iterations.")
    B.h3("Integration and freeze (iteration I-8)")
    B.p("I-8 runs no new experiments. It freezes the code, the preprocessing and the checkpoints, tags every frozen iteration in version control, evaluates each frozen iteration once on the "
        "sealed test sets under the final flag, runs the leakage check on the final splits, fills the paper, the report and the slides only from logged experiments, demonstrates the "
        "system end to end with the source set to real, and deletes the LAV-DF archive by hand. Anything not frozen before the sealed run is reported as not done. At the time of writing I-8 has not been run: the tools for it (the scoring scripts for the staircase, the checkpoint of the frozen system and its parity check) exist, and no sealed split has been read.")


def webapp(B):
    B.h2("The Web Application", trace="DECISIONS D-027, D-029, D-031, D-043")
    B.p("The software form is a local web page: the user uploads an image, a video or an audio file in a browser and sees the response of the inference library. It was chosen over a "
        "command line, which was not adopted as a second form. The page is served by Flask, chosen against FastAPI, the standard library server, Streamlit and Gradio, Bottle, by six criteria written "
        "before the choice: it installs on Python 3.13 without a compiled build, runs as one process with no database or separate front-end build, handles uploads of several hundred "
        "megabytes without code that has to be tested separately, gives plain HTML, CSS and a little JavaScript that can be restyled without a build step, adds few packages (four, on top of those "
        "already present), and returns the response structure unchanged as JSON. A job runs in a worker thread and the page polls a status route, which gives progress by stage without a second service.")
    B.p("Three safeguards were added and tested. The server binds to 127.0.0.1 only and has no option to change this. Uploads are limited to 512 MB, and an oversized upload returns an error in the "
        "response format. Uploaded files are deleted after every request, also after an aborted upload, and leftovers are removed at start.")
    B.p("The inference library takes a file and returns one structured response, so that the library and the page can be built in parallel and a development stand-in can never be mistaken for a "
        "result. <ref tab:response> lists its fields.")
    B.table("response", "Fields of the response of the inference library.",
            ["Field", "Content"],
            [["Schema version and request identifier", "version of the response format and a unique identifier of the request"],
             ["Input", "file name, media type (image, video or audio), MIME type, size, SHA-256, duration (none for images) and number of frames used (none for audio)"],
             ["Inference source", "always present: real or mock"],
             ["Model block", "checkpoint, configuration hash, preprocessing hash, commit, iteration, protocol version, calibration (method, temperature, split) and the threshold with how it was chosen; empty when the source is mock"],
             ["Result", "the fused score in [0, 1] with a decision (fake, real or undetermined); for an image the visual score, for audio the audio score; for a video the combining rule is not decided, so the decision is undetermined; per modality: availability with a reason, score and number of units used"],
             ["Score meaning", "the fixed sentence: calibrated probability that this content is fake, from a model fitted on the training data of this project, valid only for the kinds of content in that data, not proof"],
             ["Status", "ok, partial or failed, the processing time and the list of stages (validate, decode, detect, score, calibrate, fuse) with their times"],
             ["Warnings and errors", "warnings such as no face detected, several faces, short audio, low resolution; errors such as unsupported media type or unreadable file, in which case the status is failed and there is no result"]],
            [4.4, 12.0], ["left", "left"], size=10)
    B.p("A development stand-in is allowed under strict rules. It lives in a module whose name contains “mock”, always sets the source to mock and leaves the model block empty, and its scores are "
        "deterministic from the file hash so that tests are repeatable, but they carry no meaning. Every interface shows a visible banner whenever the source is not real, a test checks this, the evaluation "
        "harness refuses any run whose metadata says mock, and the document builds fail if a number cites a mock run. The real adapter refuses to run without a checkpoint whose metadata and "
        "preprocessing hash match. <ref fig:web> shows the page, with screenshots taken against the development stand-in; the banner and the undetermined decision for a video are part of the design "
        "and the scores shown mean nothing.")
    B.figure_pair("web", [str(HERE_L / "web_image.png"), str(HERE_L / "web_video.png")],
                  "The local web page against the development stand-in: (a) result for an image, (b) result for a video. The banner is part of the design and the scores shown are not measurements.", 8.0,
                  ["Screenshot of the web page with the result for an image and a purple mock banner", "Screenshot of the web page with the result for a video and a purple mock banner"],
                  tag="screenshots against the stand-in")
    R.screens(B)
    B.p("Tests with synthetic media cover the response contract, the stand-in and its banner, the real adapter stub, the web routes and, in a real browser, the page itself against the stand-in.")
    IT.software(B)


def staircase(B):
    B.h2("The Staircase Test Protocol", trace="DECISIONS D-041; PROTOCOL addendum; ITERATIONS evidence status")
    B.p("Rankings, hyperparameter choices and iteration verdicts made on the development slice and on the validation split are exploratory: they report sign, size and interval, "
        "and they choose the next step; they are never presented as confirmed performance. At the freeze each frozen iteration is evaluated once on the sealed test sets under the final flag, "
        "which gives a staircase of test scores, one per iteration, each touched once and recorded in the ledger. <ref fig:staircase> shows the protocol.")
    B.figure("staircase", "fig04_staircase.png", "Staircase test protocol. Verdicts on the development slice are exploratory; each frozen iteration is evaluated once on the sealed test sets and the pre-declared comparisons are the confirmatory claims.", 15.2,
             "Rising blocks for iterations I-0, I-1, I-2 and the final system, with arrows to the confirmatory claims", tag="schematic")
    B.p("The confirmatory claims are two comparisons, declared before any final run, both as paired group bootstrap comparisons on the sealed data under the comparison rules: the final "
        "system against the working version, and the final system against a re-run published baseline, where the second is the non-inferiority test of Equation <ref eq:noninf>. "
        "The first comparison is made on two sealed sets, face video on the Celeb-DF family and speech on In-the-Wild, so there are three tests in all; a Bonferroni interval at the 98.33 percent level is printed beside each as a robustness line. The margin "
        "$\\delta$ is stated before the comparison and chosen from the measured seed-to-seed spread of the baseline, so the baseline had to be trained before $\\delta$ existed; it was trained "
        "(Xception, three seeds, first at three learning rates and then with one more, 3e-4, that was selected) and $\\delta$ is 0.0100, the floor of the rule; the first pass gave 0.0143. Every other comparison in the report is labelled exploratory.")
    B.p("The standing limitations of this protocol are stated with it. The development slices have 15 identities, 12 speakers and 2 generator units, so intervals are wide and ties are "
        "likely; the sealed Celeb-DF family tests have about 44 independent identity groups; the threshold is chosen on validation data from the training distribution, which may be too easy, "
        "so it can transfer badly across data sets, and that would appear as a gap between the in-dataset and the cross-dataset balanced accuracy and be reported; and temperature scaling "
        "does not repair a shift between data sets.")
    IT.final_system(B)
    R.status_table(B)
    IT.limits(B)
