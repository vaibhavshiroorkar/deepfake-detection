"""Chapter 4 results of the working version (iteration 0): the final encoder comparison and the first working checkpoint.
Every number is copied from the logged entries of the encoder comparison and of the working version; they are
exploratory results on the development slice."""

import os
from pathlib import Path
HERE_L = Path(os.path.dirname(os.path.abspath(__file__))) / "_labelled"
EXPL = "exploratory, development slice"

TV = [  # name, AUC [interval], validation AUC, difference to the leader [interval], permutation p, in tie set
    ("CLIP ViT-L/14, OpenAI", "0.7464 [0.6721, 0.8306]", "0.9046", "leader", "n/a", "yes"),
    ("SigLIP base", "0.7396 [0.7031, 0.7832]", "0.8767", "−0.007 [−0.113, +0.091]", "0.930", "yes"),
    ("CLIP ViT-L/14, LAION", "0.7331 [0.6764, 0.7966]", "0.8635", "−0.013 [−0.104, +0.082]", "0.795", "yes"),
    ("DINOv2 small", "0.6933 [0.6639, 0.7314]", "0.7985", "−0.053 [−0.126, +0.010]", "0.220", "yes"),
    ("DINOv2 base", "0.6745 [0.6323, 0.7239]", "0.8170", "−0.072 [−0.171, +0.017]", "0.202", "yes"),
    ("EfficientNet-B4", "0.6471 [0.5967, 0.6987]", "0.7478", "−0.099 [−0.172, −0.023]", "0.110", "no"),
    ("Xception", "0.6162 [0.5744, 0.6633]", "0.7184", "−0.130 [−0.191, −0.071]", "0.007", "no"),
    ("Small network from scratch (reference, excluded)", "0.5166 [0.5033, 0.5324]", "not applicable", "not applicable", "n/a", "excluded"),
]
TI = [
    ("CLIP ViT-L/14, OpenAI", "0.9935 [0.9803, 0.9982]", "1.0000", "leader", "n/a", "yes"),
    ("CLIP ViT-L/14, LAION", "0.9563 [0.7760, 0.9975]", "1.0000", "−0.037 [−0.204, −0.001]", "n/a", "no"),
    ("DINOv2 base", "0.9023 [0.8496, 0.9316]", "0.9899", "−0.091 [−0.131, −0.067]", "n/a", "no"),
    ("SigLIP base", "0.8657 [0.7591, 0.9166]", "0.9975", "−0.128 [−0.221, −0.082]", "n/a", "no"),
    ("DINOv2 small", "0.8340 [0.7401, 0.8823]", "0.9685", "−0.159 [−0.240, −0.116]", "n/a", "no"),
    ("EfficientNet-B4", "0.6688 [0.6238, 0.7137]", "0.8990", "−0.325 [−0.357, −0.285]", "n/a", "no"),
    ("Xception", "0.6380 [0.4912, 0.7115]", "0.8140", "−0.355 [−0.489, −0.287]", "n/a", "no"),
    ("Small network from scratch (reference, excluded)", "0.5500 [0.5080, 0.5735]", "not applicable", "−0.4435 [−0.4723, −0.4247]", "n/a", "excluded"),
]
TA = [
    ("Whisper small encoder", "0.6380 [0.5338, 0.7700]", "0.9995", "leader", "n/a", "yes"),
    ("WavLM base plus", "0.6334 [0.5291, 0.7714]", "0.9924", "−0.005 [−0.044, +0.040]", "0.894", "yes"),
    ("HuBERT base", "0.6100 [0.5434, 0.7256]", "0.9942", "−0.028 [−0.107, +0.039]", "0.599", "yes"),
    ("wav2vec 2.0 base", "0.4975 [0.3451, 0.7092]", "0.9965", "−0.140 [−0.238, −0.050]", "0.170", "no"),
    ("Small log-mel network from scratch (reference, excluded)", "0.2458 [0.0706, 0.4404]", "not applicable", "−0.3922 [−0.6527, −0.1434]", "n/a", "excluded"),
]
HEAD = ["Candidate", "AUC on the development slice [95 percent interval]", "Validation AUC", "Difference to the leader [95 percent interval]", "Permutation p", "In tie set"]
W = [3.6, 3.5, 1.8, 3.6, 2.2, 1.7]
AL = ["left", "center", "center", "center", "center", "center"]

CAL = [  # task, candidate, EER, ECE before/after, Brier before/after
    ("Face video", "CLIP ViT-L/14, OpenAI", "0.3173", "0.2845 / 0.2874", "0.3458 / 0.3476"),
    ("Face video", "SigLIP base", "0.3356", "0.0312 / 0.0430", "0.2116 / 0.2122"),
    ("Face video", "CLIP ViT-L/14, LAION", "0.3270", "0.1768 / 0.1958", "0.2803 / 0.2883"),
    ("Face video", "DINOv2 small", "0.3582", "0.2048 / 0.2217", "0.2963 / 0.3050"),
    ("Face video", "DINOv2 base", "0.3788", "0.1118 / 0.1401", "0.2704 / 0.2815"),
    ("Face video", "EfficientNet-B4", "0.4062", "0.0636 / 0.0710", "0.2199 / 0.2189"),
    ("Face video", "Xception", "0.4137", "0.0492 / 0.0696", "0.2396 / 0.2428"),
    ("Generated images", "CLIP ViT-L/14, OpenAI", "0.0401", "0.0462 / 0.0050", "0.0334 / 0.0299"),
    ("Generated images", "CLIP ViT-L/14, LAION", "0.1196", "0.0222 / 0.0998", "0.1074 / 0.1242"),
    ("Generated images", "DINOv2 base", "0.1803", "0.1914 / 0.1534", "0.2272 / 0.2103"),
    ("Generated images", "SigLIP base", "0.2147", "0.2084 / 0.2345", "0.2463 / 0.2620"),
    ("Generated images", "DINOv2 small", "0.2446", "0.2074 / 0.1771", "0.2592 / 0.2450"),
    ("Generated images", "EfficientNet-B4", "0.3716", "0.2880 / 0.2463", "0.3284 / 0.3043"),
    ("Generated images", "Xception", "0.4069", "0.2868 / 0.1728", "0.3362 / 0.2785"),
    ("Speech audio", "Whisper small encoder", "0.4006", "0.1819 / 0.2875", "0.2796 / 0.3292"),
    ("Speech audio", "WavLM base plus", "0.3930", "0.3700 / 0.4420", "0.3893 / 0.4494"),
    ("Speech audio", "HuBERT base", "0.4183", "0.3293 / 0.4052", "0.3699 / 0.4247"),
    ("Speech audio", "wav2vec 2.0 base", "0.5020", "0.3482 / 0.4165", "0.3993 / 0.4418"),
]


def sweep_results(B):
    t1 = "E-001, ranking T-V, final"
    B.p("The comparison was run on 25 October 2026 and is final. All numbers in this section are exploratory, on the development slice: each is the mean of three head seeds, with the "
        "95 percent group bootstrap interval of 2,000 resamples, and a verdict on this slice reports sign, size and interval and is never presented as confirmed performance. The "
        "verdicts are mechanical evaluations of criteria written before the run.")
    B.h3("Face video frames")
    B.p("The slice has 1,728 clips (146 real, 1,582 fake) of 15 identity groups. <ref tab:tv> gives the ranking. The leader is the OpenAI CLIP ViT-L/14. The tie set, the "
        "candidates whose paired interval against the leader includes zero, holds five candidates: the two DINOv2 sizes, SigLIP base and the two CLIP models. The cheapest of them "
        "by end-to-end extraction time, 0.943 s per 1,000 inputs with 22.1 million parameters, is DINOv2 small, so the cost rule chooses it for the working version. The two "
        "supervised networks are outside the tie set by the bootstrap. The small network trained from scratch reaches 0.5166 [0.5033, 0.5324], which is close to chance, and is excluded from "
        "the selection.")
    B.table("tv", "Encoders on face video frames (exploratory, development slice).", HEAD, [list(r) for r in TV], W, AL, size=8, trace=t1 + "; values from the logged ranking block, exploratory (E-001)")
    B.p("The pre-declared criteria are both met. The highest candidate minus the highest supervised network is +0.0994 [+0.0228, +0.1723], an interval above zero, and the lower "
        "bound of the highest candidate, 0.6721, is above 0.60; the hypothesis is therefore supported on this slice. Two caveats stand with it. First, EfficientNet-B4 is outside the tie "
        "set by the bootstrap (difference −0.099 [−0.172, −0.023]) but its permutation p-value is 0.110, so it is borderline; the winner would not change if it were in the tie set, "
        "because DINOv2 small is cheaper end to end (0.943 against 1.732 s per 1,000 inputs). Second, every candidate loses between 0.10 and 0.16 AUC from the validation split to the "
        "development slice (for DINOv2 small 0.7985 against 0.6933), which is the generalisation gap that the hypothesis predicted. The three candidates that enter the "
        "fine-tuning iteration are DINOv2 small, SigLIP base and DINOv2 base; the two CLIP models do not fit the card.")
    B.h3("Generated images")
    B.p("The slice has 5,998 images, 2,999 real and 2,999 fake, from two generator units. <ref tab:ti> gives the ranking. The OpenAI CLIP ViT-L/14 leads and its tie set contains only itself, so "
        "the cost rule is not needed. The two leading CLIP models reach 1.0000 on the validation split, which therefore cannot separate them, and on the development slice the leader (0.9935) is separated from every other candidate by the paired bootstrap, on only two generator units.  The interval of its AUC, [0.9803, 0.9982], and all other intervals of this task must be read with a caveat: the slice has only two groups, so the "
        "bootstrap interval is not interpretable as a population interval, and the permutation test cannot reach 0.05 with fewer than six groups. The family comparison "
        "(+0.3247 [+0.2845, +0.3565]) and the lower bound above 0.65 meet the criteria, and the verdict is supported on this slice. The per-generator values are in the diagnostic discussion below.")
    B.table("ti", "Encoders on generated images (exploratory, development slice, two groups).", HEAD, [list(r) for r in TI], W, AL, size=8, trace="E-002, ranking T-I, final (exploratory)")
    B.h3("Speech audio")
    B.p("The slice has 6,991 clips (3,804 real, 3,187 fake) of 12 speakers. <ref tab:ta> gives the ranking. The Whisper encoder leads, and three candidates are tied: Whisper, WavLM and HuBERT. HuBERT "
        "is the cheapest by end-to-end time and is chosen. All four encoders reach a validation AUC of 0.99 to 1.00 on the training corpus and fall to 0.50 to 0.64 on the development slice, "
        "a gap of 0.36 to 0.50. The verdict is inconclusive: the highest lower bound, 0.5338, is at or below the 0.55 line fixed before the run, so frozen speech features did not "
        "transfer reliably to In-the-Wild. The hypothesis that WavLM or wav2vec 2.0 beats Whisper is not met (WavLM minus Whisper −0.0046 [−0.0437, +0.0399]); the difference to the "
        "reference network is above zero, but that network is itself below 0.5.")
    B.table("ta", "Encoders on speech audio (exploratory, development slice, 12 speakers; verdict inconclusive).", HEAD, [list(r) for r in TA], W, AL, size=8, trace="E-003, ranking T-A, final (exploratory)")
    B.p("Two further observations qualify the audio result. The score is not a calibrated probability on this slice: the expected calibration error rose after temperature scaling for "
        "every audio encoder (for HuBERT from 0.3293 to 0.4052), whereas it improved for the OpenAI CLIP on generated images (from 0.0462 to 0.0050). And the reference network is inverted: "
        "0.2458 [0.0706, 0.4404] is below 0.5. <ref tab:cal> gives the other calibration measures of all candidates.")
    B.table("cal", "Equal error rate and calibration on the development slice before and after temperature scaling (exploratory).",
            ["Task", "Candidate", "EER", "ECE before / after", "Brier score before / after"], [list(r) for r in CAL],
            [3.0, 4.6, 1.8, 3.5, 3.5], ["left", "left", "center", "center", "center"], size=8, trace="E-001, E-002 and E-003, calibration blocks (exploratory)")
    B.h3("Checks of the audio result and of the image result")
    B.p("Because an AUC below 0.5 can come from a labelling error, the mapping of labels was checked first, with the training and development data and the development slice only. The "
        "features of the development slice agree with the metadata of In-the-Wild on every label and speaker (6,991 items, no mismatch), and the features of the ASVspoof 2019 training "
        "and development sets agree with the protocol files (25,380 and 24,844 items, no mismatch). The network from scratch reaches an AUC of 0.99996 to 0.99997 in the training domain and "
        "0.219 to 0.286 on the development slice for its three seeds. There is therefore no label inversion: values below 0.5 indicate a sign-flipped recording-condition shortcut learned from the training corpus, which is a property of the shift between the corpora and not a worse model.")
    B.p("A second check measured simple recording features as the fake score (<ref tab:triv>). The sign of two of them reverses between the training corpus and In-the-Wild: the "
        "leading silence has an AUC of 0.375 on ASVspoof development and 0.719 on the slice, and the energy has 0.594 and 0.248. The energy value on the slice is of the same size as the network "
        "from scratch. With 12 speakers the intervals of these two include 0.5, so the sign supports the hypothesis that recording-condition cues of ASVspoof reverse in In-the-Wild, "
        "and the size does not establish it. Only the clip duration (0.622 [0.532, 0.670]) and the repeat-padded fraction (0.394 [0.348, 0.481]) exclude 0.5.")
    B.table("triv", "AUC of single recording features used as the fake score (exploratory; a value below 0.5 means that fake clips have the smaller value).",
            ["Feature", "ASVspoof training, 20 speakers", "ASVspoof development, 20 speakers", "In-the-Wild slice, 12 speakers"],
            [["Leading silence", "0.416 [0.350, 0.488]", "0.375 [0.299, 0.453]", "0.719 [0.425, 0.933]"],
             ["Clip duration", "0.479 [0.428, 0.531]", "0.467 [0.393, 0.540]", "0.622 [0.532, 0.670]"],
             ["Energy (RMS)", "0.572 [0.525, 0.618]", "0.594 [0.537, 0.649]", "0.248 [0.073, 0.546]"],
             ["Repeat-padded fraction", "0.530 [0.478, 0.581]", "0.543 [0.471, 0.614]", "0.394 [0.348, 0.481]"]],
            [3.6, 4.2, 4.4, 4.2], ["left", "center", "center", "center"], size=9, trace="M-004 b, trivial audio features (exploratory)")
    B.p("For the image task, the pooled AUC hides a difference between the two generator units of the slice. The OpenAI CLIP separates StarGAN almost perfectly (0.9988 [0.9983, 0.9992]) and "
        "reaches 0.9830 [0.9788, 0.9870] on the Which-Face-Is-Real unit; the LAION CLIP reaches 0.9978 and 0.7790 [0.7591, 0.8000]. The intervals of these per-unit values resample images "
        "inside one unit and are not population intervals. The file extension does not separate the classes in any split of this task.")


def working_results(B):
    B.h3("Results of the working version")
    B.p("The heads of the three winners were trained with three seeds and the checkpoint was built on 25 October 2026 from the head of seed 0, a choice fixed before the run so that no number of the "
        "development slice enters it. Temperature and threshold were fitted on the validation split of each task; the threshold is stored in the calibrated probability space. "
        "The temperatures are 0.907 (face video), 0.636 (generated images) and 0.486 (audio), and the thresholds 0.4774, 0.2073 and 0.1451. <ref tab:i0> gives the AUC and the equal error "
        "rate, and <ref tab:i0cal> the calibration; all development-slice values are exploratory and reproduce the comparison above.")
    B.table("i0", "Working version: AUC with 95 percent group bootstrap interval, mean of three seeds (development slice values exploratory).",
            ["Task", "Encoder", "Training AUC", "Validation AUC", "Development slice AUC", "Slice EER"],
            [["Face video", "DINOv2 small", "0.8367 [0.8267, 0.8475]", "0.7985 [0.7731, 0.8269]", "0.6933 [0.6639, 0.7314]", "0.3582"],
             ["Generated images", "CLIP ViT-L/14, OpenAI", "1.0000 [1.0000, 1.0000]", "1.0000 [1.0000, 1.0000]", "0.9935 [0.9803, 0.9982]", "0.0401"],
             ["Speech audio", "HuBERT base", "0.9997 [0.9996, 0.9999]", "0.9942 [0.9910, 0.9967]", "0.6100 [0.5434, 0.7256]", "0.4183"]],
            [2.6, 3.3, 3.2, 3.2, 3.2, 1.7], ["left", "left", "center", "center", "center", "center"], size=8, trace="E-004, working version, exploratory on the development slice")
    B.table("i0cal", "Working version: expected calibration error before and after temperature scaling, Brier score and balanced accuracy at the validation threshold.",
            ["Task", "ECE, validation", "ECE, slice", "Brier, slice", "Balanced accuracy, validation / slice"],
            [["Face video", "0.0325 / 0.0321", "0.2048 / 0.2217", "0.2963 / 0.3050", "0.750 / 0.617"],
             ["Generated images", "0.0050 / 0.0012", "0.0462 / 0.0050", "0.0334 / 0.0299", "0.998 / 0.955"],
             ["Speech audio", "0.0883 / 0.0078", "0.3293 / 0.4052", "0.3699 / 0.4247", "0.971 / 0.529"]],
            [3.0, 3.3, 3.3, 3.3, 3.5], ["left", "center", "center", "center", "center"], size=9, trace="E-004, calibration (exploratory)")
    B.p("The standard deviation over the three seeds on the development slice is 0.0107, 0.0006 and 0.0026 against bootstrap half-widths of 0.0338, 0.0089 and 0.0911, so it is the sampling of "
        "identities and speakers, not training noise, that limits the intervals. The interval of the image task rests on two units and that of the audio task on 12 speakers.")
    B.p("The checkpoint was tested in two ways. In a parity check, seven training items (one fake and one real video, two ProGAN images, three ASVspoof utterances) were scored from the stored "
        "features and by the real adapter from the raw file; the largest absolute difference of the calibrated score was 0.0019 (0.0002 for video, 0.0019 for images and 0.0005 for audio) "
        "against a tolerance of 0.02, so the two dependences named in advance, the JPEG round trip and the colour order, agree. In the page check, the local web page was started with the real "
        "adapter and the checkpoint, and in a real browser one image, one video with audio and one audio file, all training items, gave results with the source marked real, no stand-in banner and no page "
        "error; the video shows its visual and audio score separately with the verdict undetermined, and the upload folder was empty afterwards. These items are training data, so the check "
        "shows that the path works and says nothing about accuracy. The claim of the working version is limited to this: the path works, and metrics with intervals exist.")
    B.h3("Diagnostic battery of the working version")
    B.p("The battery was run once, with the following observations (exploratory). Per method, the validation AUC of the face task is 0.884 for Deepfakes, 0.834 for Face2Face, 0.778 for FaceSwap and 0.697 "
        "for NeuralTextures, and the slice has one kind of fake, 0.693. For generated images all six validation classes score 1.000, and the slice units score 0.999 and 0.983. For speech the "
        "validation attacks score 0.984 to 0.997, and the slice per speaker has a minimum of 0.503, a median of 0.692 and a maximum of 0.990, with the pooled value 0.610 below the median.")
    B.p("The training minus validation AUC is +0.038 for face video, 0.000 for images and +0.006 for audio, and validation minus slice is +0.105, +0.007 and +0.384. Temperature scaling "
        "fitted on validation did not improve the calibration on the slice for face video or audio and did for images, and all three temperatures are below one, so the validation scores are "
        "sharpened and the slice scores are then over-confident. For face video the calibrated scores of real and fake clips overlap strongly (at the 50th percentile 0.304 for real and 0.434 for "
        "fake); for audio real In-the-Wild speech scores as fake (0.971 against 0.994 at the 50th percentile). Clips with two or more faces score 0.762 on the slice against 0.692 for one face, and "
        "the face-height tertiles score 0.689, 0.696 and 0.697, so face size and number show no effect that needs a lever; compression could not be split, because FaceForensics++ c23 is one level "
        "and the compression of Celeb-DF is not recorded. Aggregation does not help: on the slice the frame level scores 0.656, the clip mean 0.685, the median 0.688, the mean of the three "
        "highest 0.663 and the maximum 0.662.")
    B.p("The 20 worst errors on the slice were reviewed by eye from local contact sheets (face video and images) and from the audio features. The false alarms of the face task are sharp, "
        "well lit and near-frontal real faces with smooth skin, and six of ten misses come from one identity group; for images the false alarms are real StarGAN crops, several with stretched "
        "border pixels, and nine of ten misses are photorealistic StyleGAN faces; for audio both kinds of error are short clips (median 1.9 and 1.2 s) with a repeat-padded fraction of 0.52 and 0.70 and "
        "saturated scores. The contact sheets are not shown, because the licences of the data sets forbid publishing derived data.")
    B.h3("Candidate diagnoses")
    B.p("The diagnoses below use the vocabulary of the battery and are candidates for the choice of the next lever; no lever is chosen here. For face video the first candidate is bias, since the "
        "training AUC is only 0.837 and the frozen feature separates even the training data moderately, followed by shift (0.105 from validation to slice) and calibration under shift; aggregation is not "
        "indicated. For generated images no bias or variance is visible and calibration is good, but a slice of two units cannot measure the shift to other generators. For audio the shift is "
        "dominant (0.384 at an in-domain AUC of 0.994), the checks above point to recording-condition cues that reverse in In-the-Wild, calibration worsens under shift, and the errors "
        "concentrate in short repeat-padded clips, so a window or padding effect is a candidate as a missing signal or a shortcut.")


def screens(B):
    B.figure_pair("webreal", [str(HERE_L / "real_image.png"), str(HERE_L / "real_video.png")],
                  "The local web page with the real checkpoint: (a) result for an image, (b) result for a video with audio. The video shows its visual and audio score separately and the verdict undetermined.", 8.0,
                  ["Screenshot of the web page with the result for an image and the real checkpoint", "Screenshot of the web page with the result for a video, with a visual and an audio score"],
                  tag="screenshots of the real checkpoint on training items", trace="E-004 page check on the real checkpoint (training items, no accuracy claim)")
    B.p("<ref fig:webreal> and <ref fig:webaudio> show the page with the real checkpoint, on training items, so they show that the path works and not how accurate it is. For an audio score the page "
        "adds a warning to the result: it says that on the development slice the calibration error of every audio encoder tried rose after temperature scaling, so the audio score is to be read as a "
        "ranking score and not as a reliable probability. The verdict for an audio file above the threshold still appears next to this warning.")
    B.figure("webaudio", "../screens/real_audio.png", "The local web page with the real checkpoint for an audio file, with the warning about the exploratory calibration of audio scores.", 8.5,
             "Screenshot of the web page with the result for an audio file and a warning about the calibration of audio scores", tag="screenshot of the real checkpoint on a training item",
             trace="E-004 page check on the real checkpoint (training item, no accuracy claim)")


def status_table(B):
    pend = lambda x: "<pend %s>" % x
    N = "E-014 (freeze)"
    B.p("<ref tab:status> lists the results that do not exist yet. They are all sealed-test results: each cell is filled only from a logged final evaluation, and the sealed test sets are not read before the freeze.")
    B.table("status", "Results that are still pending (sealed test sets; the sealed run follows the readiness checks).",
            ["Result", "Where it is measured", "Value"],
            [["AUC of the working version on its sealed test sets", "sealed in-dataset and cross-dataset tests, once, at the freeze", pend(N)],
             ["Sealed test AUC of each frozen iteration", "staircase, one evaluation per iteration", pend(N)],
             ["Final face-video system against the working version", "paired group bootstrap on the sealed Celeb-DF family test", pend(N)],
             ["Final speech system against the working version", "paired group bootstrap on the sealed In-the-Wild test", pend(N)],
             ["Final face-video system against the re-run baseline", "non-inferiority interval with the margin 0.0100, on the sealed Celeb-DF family test", pend(N)],
             ["Robustness line of the three tests", "interval at the 98.33 percent level (Bonferroni over three), from the same resamples", pend(N)],
             ["LAV-DF sealed test of the unimodal heads and the fusion", "exploratory, once", pend(N)]],
            [7.2, 6.4, 2.8], ["left", "left", "center"], size=9)
